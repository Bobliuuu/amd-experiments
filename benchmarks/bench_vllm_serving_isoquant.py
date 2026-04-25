#!/usr/bin/env python3
"""
End-to-end serving throughput benchmark for vLLM + IsoQuant backend.

Workflow:
1) Launch OpenAI-compatible vLLM server in a subprocess.
2) Send concurrent completion requests.
3) Measure aggregate output tokens/sec and request latency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from cache_utils import add_swa_args, print_swa_status, vllm_swa_warn


def _post_json(url: str, payload: Dict, timeout_s: float) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    models_url = f"{base_url.rstrip('/')}/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"vLLM server did not become ready in {timeout_s:.0f}s")


@dataclass
class RequestResult:
    latency_s: float
    output_tokens: int
    ok: bool
    error: str = ""


def _single_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> RequestResult:
    start = time.time()
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        body = _post_json(f"{base_url.rstrip('/')}/completions", payload, timeout_s)
        choices = body.get("choices", [])
        usage = body.get("usage", {})
        out_tokens = int(usage.get("completion_tokens", 0))
        if out_tokens == 0 and choices:
            out_tokens = max_tokens
        return RequestResult(time.time() - start, out_tokens, True)
    except Exception as e:
        return RequestResult(time.time() - start, 0, False, str(e))


def _make_prompts(n: int, prompt_len_tokens_hint: int) -> List[str]:
    seed = "The AMD MI300X attention benchmark sentence."
    return [(" ".join([seed] * max(1, prompt_len_tokens_hint // 8))) for _ in range(n)]


def _launch_server(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        args.python_bin, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--kv-cache-dtype", args.kv_cache_dtype,
    ]
    env = os.environ.copy()
    if args.attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    else:
        env.pop("VLLM_ATTENTION_BACKEND", None)
    env["VLLM_IQ_METHOD"] = args.iq_method
    env["VLLM_IQ_BITS"] = str(args.iq_bits)
    env["PYTHONPATH"] = f"{args.kernels_path}:{env.get('PYTHONPATH', '')}"

    return subprocess.Popen(
        cmd,
        env=env,
        cwd=args.server_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def run(args: argparse.Namespace) -> Dict:
    server = _launch_server(args)
    base_url = f"http://{args.host}:{args.port}/v1"
    try:
        try:
            _wait_for_server(base_url, args.server_start_timeout_s)
        except Exception:
            if server.stdout is not None:
                lines = []
                for _ in range(200):
                    line = server.stdout.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))
                raise RuntimeError("Failed to start vLLM server:\n" + "\n".join(lines))
            raise

        prompts = _make_prompts(args.num_requests, args.prompt_tokens)
        all_results: List[RequestResult] = []
        wall_start = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    _single_request,
                    base_url,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.request_timeout_s,
                )
                for prompt in prompts
            ]
            for fut in as_completed(futures):
                all_results.append(fut.result())
        wall_s = max(time.time() - wall_start, 1e-6)

        ok = [r for r in all_results if r.ok]
        fail = [r for r in all_results if not r.ok]
        total_output_tokens = sum(r.output_tokens for r in ok)
        tok_per_s = total_output_tokens / wall_s
        latencies = sorted(r.latency_s for r in ok)

        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            idx = min(int((p / 100.0) * (len(latencies) - 1)), len(latencies) - 1)
            return latencies[idx]

        return {
            "model": args.model,
            "backend": "ISOQUANT_ROCM",
            "kv_cache_dtype": args.kv_cache_dtype,
            "iq_method": args.iq_method,
            "iq_bits": args.iq_bits,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "prompt_tokens_hint": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "swa": args.swa,
            "swa_window": args.window if args.swa == "on" else None,
            "successful_requests": len(ok),
            "failed_requests": len(fail),
            "total_output_tokens": total_output_tokens,
            "elapsed_s": wall_s,
            "throughput_tokens_per_s": tok_per_s,
            "latency_p50_s": pct(50),
            "latency_p95_s": pct(95),
            "errors": [r.error for r in fail[:5]],
        }
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--kv-cache-dtype", default="iso3")
    p.add_argument(
        "--attention-backend",
        default="ISOQUANT_ROCM",
        help="Set VLLM_ATTENTION_BACKEND. Use empty string to unset.",
    )
    p.add_argument("--iq-method", choices=["iso", "planar"], default="iso")
    p.add_argument("--iq-bits", type=int, choices=[3, 4], default=3)
    p.add_argument("--kernels-path", default="/root/workspace/amd-experiments/kernels")
    p.add_argument(
        "--server-cwd",
        default="/tmp",
        help="Working directory for launched vLLM server process.",
    )
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-timeout-s", type=float, default=180.0)
    p.add_argument("--server-start-timeout-s", type=float, default=240.0)
    p.add_argument("--output-json", default="")
    p.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch vLLM server.",
    )
    add_swa_args(p)
    args = p.parse_args()
    if args.attention_backend == "":
        args.attention_backend = None
    print_swa_status(args.swa, args.window if args.swa == "on" else None)
    vllm_swa_warn(args.swa, args.max_model_len)
    return args


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

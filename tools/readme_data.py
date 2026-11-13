#!/usr/bin/env python3
"""results/*.json -> results/derived/readme_data.json (README and video numbers). Nothing is hand-typed."""
import json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
J = lambda p: json.load(open(os.path.join(ROOT, "results", p)))
NAME = {"turbo": "TurboQuant (TQ3)", "iso": "IsoQuant", "planar": "PlanarQuant", "rotor": "RotorQuant"}
ORDER = ["turbo", "iso", "planar", "rotor"]
d = {}
cd = [r for r in J("bench_compress_decompress.json") if r["bits"] == 3 and r["n_vectors"] == 4096]
cd.sort(key=lambda r: ORDER.index(r["method"]))
pf = {(r["method"], r["bits"]): r for r in J("bench_prefill_all_methods.json")["standalone"] if r["seq_len"] == 32768}
cs = {(r["method"], r["bits"]): r for r in J("bench_ppl_all_methods.json")}
d["methods"] = [{"method": NAME[r["method"]], "compression": r["compression_ratio"], "fmas": r["fmas_per_vec"],
                 "compress_gbs": r["compress_bw_gbs"], "decompress_gbs": r["decompress_bw_gbs"],
                 "prefill_ktps": pf[(r["method"], 3)]["compress_toks_per_sec"] / 1e3,
                 "cosine": cs[(r["method"], 3)]["cosine_sim_mean"], "cosine4": cs[(r["method"], 4)]["cosine_sim_mean"]} for r in cd]
m = {x["method"]: x for x in d["methods"]}
d["compression"] = cd[0]["compression_ratio"]
d["planar_prefill_x"] = m["PlanarQuant"]["prefill_ktps"] / m["TurboQuant (TQ3)"]["prefill_ktps"]
d["iso_compress"] = m["IsoQuant"]["compress_gbs"]
d["turbo_compress"] = m["TurboQuant (TQ3)"]["compress_gbs"]
d["planar_prefill"] = m["PlanarQuant"]["prefill_ktps"]
d["turbo_prefill"] = m["TurboQuant (TQ3)"]["prefill_ktps"]
d["planar_fmas"] = m["PlanarQuant"]["fmas"]
d["turbo_fmas"] = m["TurboQuant (TQ3)"]["fmas"]
d["cos_min"] = min(x["cosine"] for x in d["methods"])
d["cos_max"] = max(x["cosine"] for x in d["methods"])
d["cos4"] = min(x["cosine4"] for x in d["methods"])
ta = J("bench_triton_attention.json")["results"]
d["triton_curve"] = [{"seq_k": r["seq_k"], "series": "fused TQ3 vs fp16 SDPA", "speedup": r["speedup_vs_fp16"]} for r in ta]
d["triton_best"] = max(r["speedup_vs_fp16"] for r in ta)
d["triton_worst"] = min(r["speedup_vs_fp16"] for r in ta)
d["triton_32k"] = next(r["speedup_vs_fp16"] for r in ta if r["seq_k"] == 32768)
d["triton_1k"] = next(r["speedup_vs_fp16"] for r in ta if r["seq_k"] == 1024)
g = J("bench_tq_gqa_decode_sweep.json")["rows"]
d["gqa_min"] = min(r["speedup"] for r in g)
d["gqa_max"] = max(r["speedup"] for r in g)
v = J("bench_vllm_turboquant_ab_sweep_kv_heavy.json")["results"]
d["vllm"] = [{"backend": r["backend"], "tps": r["throughput_output_tps"]} for r in v]
d["vllm_min"] = min(r["throughput_output_tps"] for r in v)
d["vllm_max"] = max(r["throughput_output_tps"] for r in v)
n = J("bench_niah.json")
d["niah_min"] = min(r["pass_rate"] for r in n if r["bits"] == 3) * 100
d["niah_trials"] = n[0]["n_trials"]
d["niah_ctx_max"] = max(r["context_len"] for r in n)
os.makedirs(os.path.join(ROOT, "results", "derived"), exist_ok=True)
json.dump(d, open(os.path.join(ROOT, "results", "derived", "readme_data.json"), "w"), indent=1)
print({k: round(x, 3) for k, x in d.items() if isinstance(x, float)})

# Benchmarks (`benchmarks`)

This directory contains runnable benchmark entry points for throughput, latency, quality, and serving behavior.

## Run Location

Run all commands from repo root unless noted:

```bash
cd /root/workspace/amd-experiments
```

## Common Quick Runs

```bash
# Compression/decompression GB/s (fast sanity check)
python3 benchmarks/bench_compress_decompress.py --n-vectors 4096 --n-iters 50

# Prefill throughput by method
python3 benchmarks/bench_prefill.py --model mistralai/Mistral-7B-v0.1

# Decode throughput by batch/sequence (v2)
python3 benchmarks/bench_batch_decode_v2.py --model mistralai/Mistral-7B-v0.1

# Reconstruction quality (cosine + MSE)
python3 benchmarks/bench_quality.py --model mistralai/Mistral-7B-v0.1 --n-tokens 4096
```

## Full Set (Frequently Used)

- `bench_compress_decompress.py`: compress/decompress bandwidth
- `bench_prefill.py`: prefill token throughput
- `bench_batch_decode_v2.py`: batch decode scaling
- `bench_all_methods_decode.py`: decode throughput across methods
- `bench_ppl_all_methods.py`: PPL comparison at 3-bit and 4-bit
- `bench_quality.py`: layer-level quality metrics
- `bench_niah.py`: needle-in-a-haystack retrieval
- `bench_kernels.py`: TurboQuant kernel microbenchmarks
- `bench_tq_attention.py`: fused TQ attention vs SDPA
- `bench_triton_attention.py`: FP16 vs Python-TQ3 vs Triton-fused-TQ3
- `bench_vllm_serving.py`: vLLM serving throughput experiments

## Batch Run

To execute the broader benchmark suite:

```bash
bash run_all_benchmarks.sh mistralai/Mistral-7B-v0.1
```

## Outputs

Most scripts write JSON artifacts to `results/`.

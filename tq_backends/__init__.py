# Drop-in attention backends for AMD ROCm (TurboQuant TQ3, IsoQuant).
# Do NOT name this package `vllm` — it would shadow PyPI vLLM. Copy
# `attention/backends/rocm_flash_attn.py` into site-packages after `pip install vllm`
# (see scripts/install_turboquant_vllm_backend.sh and docs/vllm_turboquant_wiring.md).

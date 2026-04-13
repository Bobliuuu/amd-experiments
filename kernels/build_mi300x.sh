#!/bin/bash
# build_mi300x.sh — Build TurboQuant MI300X library, tests, and benchmarks
#
# Target: gfx942 (AMD Instinct MI300X), ROCm 7.2
# Usage:
#   ./build_mi300x.sh            # build all
#   ./build_mi300x.sh lib        # shared library only
#   ./build_mi300x.sh test       # validation suite
#   ./build_mi300x.sh bench      # benchmark binary
#   ./build_mi300x.sh clean      # remove build artifacts
#
# After build: run ./tq_validate_mi300x to verify 9/9 tests pass.

set -euo pipefail

# IMPORTANT: Use the exact arch string reported by torch.cuda.get_device_properties().
# The VF adds sramecc+:xnack- feature flags that must match the embedded binary.
# Without these flags, hipMemcpyToSymbol returns hipErrorNoBinaryForGpu (209).
#
# ROCm VERSION NOTE:
# This build uses system ROCm 7.2 hipcc (/opt/rocm/bin/hipcc).
# PyTorch on this system bundles its own libamdhip64.so compiled against ROCm 6.2.
# Because HIP fat-binary ABI is NOT backward-compatible across major versions,
# the shared library (libturboquant_mi300x.so) cannot be loaded via ctypes in
# the same process as PyTorch without error 209.
#
# The standalone binaries (tq_validate_mi300x, tq_bench_mi300x) work correctly
# because they use the system ROCm 7.2 runtime directly.
#
# For Python integration, turboquant_mi300x.py uses a pure-PyTorch implementation
# that is equivalent in quality (rotation via torch.matmul → rocBLAS → MFMA).
ARCH="gfx942:sramecc+:xnack-"
ROCM="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM}/bin/hipcc"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ──────────────────────────────────────────────────────────────────────────────
# Compiler flags
# ──────────────────────────────────────────────────────────────────────────────

COMMON_FLAGS=(
    -O3
    --offload-arch="${ARCH}"
    -mwavefrontsize64           # Enforce Wave64 on CDNA3 (default, explicit for safety)
    -DCDNA3
    -DAMD_MFMA_AVAILABLE        # MFMA intrinsics available on gfx942
    -DTARGET_MI300X
    -I"${ROCM}/include"
    -I"${SCRIPT_DIR}"
    -ffast-math
    -Wall
    -Wno-unused-result
)

echo "=== TurboQuant MI300X Build ==="
echo "ROCm:     ${ROCM}"
echo "hipcc:    ${HIPCC}"
echo "Target:   ${ARCH}"
echo "Flags:    ${COMMON_FLAGS[*]}"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Clean
# ──────────────────────────────────────────────────────────────────────────────

clean() {
    rm -f libturboquant_mi300x.so tq_validate_mi300x tq_bench_mi300x
    echo "Clean done."
}

# ──────────────────────────────────────────────────────────────────────────────
# Build: shared library
# ──────────────────────────────────────────────────────────────────────────────

build_lib() {
    echo "[1/3] Building libturboquant_mi300x.so ..."
    "${HIPCC}" "${COMMON_FLAGS[@]}" \
        -fPIC \
        --shared \
        -o "${SCRIPT_DIR}/libturboquant_mi300x.so" \
        "${SCRIPT_DIR}/turboquant_mi300x.hip.cpp" \
        -lm
    echo "      -> libturboquant_mi300x.so"
}

# ──────────────────────────────────────────────────────────────────────────────
# Build: validation test suite
# ──────────────────────────────────────────────────────────────────────────────

build_test() {
    echo "[2/3] Building tq_validate_mi300x ..."
    "${HIPCC}" "${COMMON_FLAGS[@]}" \
        -o "${SCRIPT_DIR}/tq_validate_mi300x" \
        "${SCRIPT_DIR}/turboquant_mi300x.hip.cpp" \
        "${SCRIPT_DIR}/turboquant_mi300x_test.cpp" \
        "${SCRIPT_DIR}/ref/ggml_turboquant.c" \
        -lm
    echo "      -> tq_validate_mi300x"
}

# ──────────────────────────────────────────────────────────────────────────────
# Build: micro-benchmark
# ──────────────────────────────────────────────────────────────────────────────

build_bench() {
    if [ -f "${SCRIPT_DIR}/tq_hip_benchmark_mi300x.cpp" ]; then
        echo "[3/3] Building tq_bench_mi300x ..."
        "${HIPCC}" "${COMMON_FLAGS[@]}" \
            -o "${SCRIPT_DIR}/tq_bench_mi300x" \
            "${SCRIPT_DIR}/turboquant_mi300x.hip.cpp" \
            "${SCRIPT_DIR}/tq_hip_benchmark_mi300x.cpp" \
            -lm
        echo "      -> tq_bench_mi300x"
    else
        echo "[3/3] tq_hip_benchmark_mi300x.cpp not found — skipping bench build."
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────────

MODE="${1:-all}"
cd "${SCRIPT_DIR}"

case "${MODE}" in
    clean)  clean ;;
    lib)    build_lib ;;
    test)   build_test ;;
    bench)  build_bench ;;
    all)
        build_lib
        build_test
        build_bench
        echo ""
        echo "=== Build complete ==="
        echo "Run validation:  ./tq_validate_mi300x"
        echo "Run benchmark:   ./tq_bench_mi300x <n_vectors> <n_iters>"
        ;;
    *)
        echo "Usage: $0 [all|lib|test|bench|clean]"
        exit 1
        ;;
esac

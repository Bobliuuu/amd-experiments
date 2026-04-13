"""Minimal single-kernel benchmark for rocprofv2 counter collection."""
import sys
sys.path.insert(0, "/root/workspace/amd-experiments/kernels")
import torch
from turboquant_mi300x import TurboQuantMI300X

device = "cuda"
N = 65536
import numpy as np
rng = np.random.default_rng(42)
x = torch.from_numpy(rng.standard_normal((N, 128)).astype("float32")).to(device)
torch.cuda.synchronize()

tq = TurboQuantMI300X(bits=3, rotation_seed=42)

# warmup
for _ in range(3):
    tq.compress_tensor(x)
torch.cuda.synchronize()

# measured iterations
for _ in range(10):
    tq.compress_tensor(x)
torch.cuda.synchronize()
print(f"TQ3 compress: 10 iters, {N} vectors done")

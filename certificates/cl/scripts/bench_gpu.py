"""GPU/CPU assessment (AGENTS.md): bounded real workload = singular values of the bipartite block of T for all
tuples with 2j_max = M exactly (multiset enumeration), float64.  Times: (a) building blocks (CPU, shared);
(b) numpy svd loop, 1 process; (c) torch CUDA svdvals batched by shape incl. packing/transfer/sync.
Usage: python bench_gpu.py M
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import grid, bip

M = int(sys.argv[1])
t = time.time()
blocks = []
for s2 in range(1, M + 1):
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    tj = (s2, a, b, c, d)
                    if max(tj) != M or sum(tj) % 2: continue
                    S, W = grid(tj)
                    if len(S) < 2: continue
                    B, _, _ = bip(S, W)
                    if B.size: blocks.append(B)
tb = time.time() - t
t = time.time()
cpu = [np.linalg.svd(B, compute_uv=False).min() for B in blocks]
tc = time.time() - t
out = [f"M={M}: {len(blocks)} blocks, build {tb:.2f}s, numpy svd (1 proc) {tc:.2f}s"]
try:
    import torch
    assert torch.cuda.is_available()
    dev = torch.device('cuda')
    torch.cuda.synchronize()
    t = time.time()
    groups = {}
    for i, B in enumerate(blocks):
        groups.setdefault(B.shape, []).append(i)
    gpu = [None] * len(blocks)
    for shp, idx in groups.items():
        X = torch.from_numpy(np.stack([blocks[i] for i in idx])).to(dev, dtype=torch.float64)
        sv = torch.linalg.svdvals(X)
        mins = sv.min(dim=1).values.cpu().numpy()
        for k, i in enumerate(idx): gpu[i] = mins[k]
    torch.cuda.synchronize()
    tg = time.time() - t
    err = max(abs(a - b) for a, b in zip(cpu, gpu))
    out.append(f"torch cuda svdvals batched by shape ({len(groups)} shapes), incl. transfer: {tg:.2f}s; max |diff| {err:.2e}")
    out.append(f"GPU: {torch.cuda.get_device_name(0)}, mem allocated peak {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
except Exception as e:
    out.append(f"torch/cuda unavailable: {e!r}")
print('\n'.join(out))

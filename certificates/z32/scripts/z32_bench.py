"""GPU/CPU measurement (AGENTS.md) on a bounded real workload: regime R1 and R2- boxes with n1 <= 150.
Compares (i) the per-parameter numpy implementation z32_box_np.best, (ii) the vectorised torch code on CPU,
(iii) the same on CUDA, timing end to end (construction, transfers, synchronisation, result collection)."""
import time, sys, os, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z32_box_np as bx
import z32_box_torch as zt

NM = 150.0
out = {}
for name in ('R1', 'R2-'):
    # (i) numpy per parameter set
    t = time.time(); mn = 1e9
    amin = {'R1': 1.5, 'R2-': 1.0}[name]
    for n2 in range(4, int(2 * NM) + 1):
        n = n2 / 2; a = n - 2
        while a >= amin - 1e-9:
            v = bx.b8(*bx.best(name, n, a)); mn = min(mn, v); a -= 1
    out[name + '_numpy'] = (time.time() - t, mn)
    for dev in ('cpu', 'cuda'):
        if dev == 'cuda':
            torch.cuda.synchronize()
        t = time.time()
        n, a, F0, F1, W, s2 = zt.regime_box(name, NM, torch.device(dev))
        m = float((8 * torch.sqrt(s2)).min().item())   # forces sync + transfer
        out[name + '_torch_' + dev] = (time.time() - t, m)
for k, v in out.items():
    print(f'{k:16s} {v[0]:8.2f} s   min 8sigma {v[1]:.10f}')

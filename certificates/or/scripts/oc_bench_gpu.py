"""GPU/CPU measurement (AGENTS.md) on a bounded real workload: all open tuples with 2jmax = Mlev exactly,
the work of oc_survey.py per tuple (T for P1; K_G and K for the three pairings => 7 bipartite blocks, smallest
nonzero singular value of each).  float64 throughout.
  CPU path : build + numpy SVD, single process (same code as the survey worker).
  GPU path : build on CPU (identical), group blocks by shape, torch.linalg.svdvals on CUDA float64 per shape
             batch, min nonzero per block, results back to host.  Timed end to end incl. packing/transfer/sync.
Checks that both paths give the same minima (max relative difference).  Usage: python oc_bench_gpu.py Mlev out.txt
"""
import sys, os, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_fast import build, smin_nz, is_open, pairings


def level_tuples(M):
    for s2 in range(3, M + 1, 2):
        for a in range(1, M + 1):
            for b in range(a, M + 1):
                for c in range(b, M + 1):
                    for d in range(c, M + 1):
                        if max(s2, d) == M and (s2 + a + b + c + d) % 2 == 0:
                            yield (s2, a, b, c, d)


def blocks_of(tj):
    out = []
    for k, P in enumerate(pairings(tj)):
        o = build(P, ('T', 'KG', 'K') if k == 0 else ('KG', 'K'))
        for key in (('T', 'KG', 'K') if k == 0 else ('KG', 'K')):
            out.append(o[key])
    return out


if __name__ == '__main__':
    M, out = int(sys.argv[1]), sys.argv[2]
    import torch
    L = []
    dev = torch.device('cuda')
    L.append(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}, "
             f"free/total memory {torch.cuda.mem_get_info()[0]/2**30:.1f}/{torch.cuda.mem_get_info()[1]/2**30:.1f} GiB")
    t0 = time.time()
    tups = [t for t in level_tuples(M) if is_open(t)]
    t_sel = time.time() - t0
    # CPU path
    t0 = time.time()
    cpu_res = []
    for t in tups:
        cpu_res.append([smin_nz(B)[0] for B in blocks_of(t)])
    t_cpu = time.time() - t0
    # GPU path
    t0 = time.time()
    allb = []
    for i, t in enumerate(tups):
        for j, B in enumerate(blocks_of(t)):
            allb.append((i, j, B))
    t_build = time.time() - t0
    t1 = time.time()
    groups = {}
    for i, j, B in allb:
        groups.setdefault(B.shape, []).append((i, j, B))
    gpu_res = [[None] * 7 for _ in tups]
    for shp, items in groups.items():
        if min(shp) == 0:
            for i, j, B in items:
                gpu_res[i][j] = None
            continue
        X = torch.from_numpy(np.stack([B for _, _, B in items])).to(dev)
        sv = torch.linalg.svdvals(X)
        smax = sv.max(dim=1).values
        thr = 1e-10 * torch.clamp(smax, min=1.0) * (shp[0] + shp[1])
        big = torch.where(sv > thr[:, None], sv, torch.full_like(sv, float('inf')))
        mn = big.min(dim=1).values.cpu().numpy()
        for (i, j, _), v in zip(items, mn):
            gpu_res[i][j] = None if not np.isfinite(v) else float(v)
    torch.cuda.synchronize()
    t_gpu_svd = time.time() - t1
    t_gpu = t_build + t_gpu_svd
    diff = 0.0
    for a, b in zip(cpu_res, gpu_res):
        for x, y in zip(a, b):
            if x is None or y is None:
                if not (x is None and y is None):
                    diff = max(diff, 1.0)
                continue
            diff = max(diff, abs(x - y) / x)
    L.append(f"workload: open tuples with 2jmax={M}: {len(tups)} tuples, {len(allb)} bipartite blocks, {len(groups)} shapes "
             f"(selection {t_sel:.1f}s, not counted)")
    L.append(f"CPU (numpy, 1 process, build+svd): {t_cpu:.2f}s")
    L.append(f"GPU (build on CPU {t_build:.2f}s + pack/transfer/svdvals/sync/collect {t_gpu_svd:.2f}s): {t_gpu:.2f}s")
    L.append(f"max relative difference CPU vs GPU minima: {diff:.2e}")
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

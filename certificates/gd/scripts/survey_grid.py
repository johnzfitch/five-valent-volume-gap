"""Labelled survey of the smallest nonzero |eigenvalue| q_min of the 5-valent AL operator, using the
normal form (grid.py): orbit 1 (alpha,beta)=(2,0), orbit 2 (2,2), orbit 3 (4,2).  All labelled spin tuples
(j1 = distinguished edge; pairs {j2,j3}, {j4,j5}) with max 2j_i = m for m = 1..M, reduced by the exact
symmetries: swap inside a pair (X, D are symmetric under it up to sign), and for orbit 2 the swap of pairs.
Orbit 3 is NOT symmetric under the pair swap (alpha != beta), so both orders are enumerated.
Singular values of the bipartite block (float64); zero threshold TOL*s_max*dim, TOL=1e-12; the largest
'zero' / q_min ratio is recorded.
Backends: numpy (multiprocessing) or torch (CUDA, batched eigvalsh grouped by shape, float64).
Usage: python survey_grid.py --M 16 --orbit 2 --backend numpy --workers 8 --out file.jsonl
       python survey_grid.py --bench 12 --orbit 2 --backend torch
"""
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import W_matrix

AB = {1: (2, 0), 2: (2, 2), 3: (4, 2)}
TOL = 1e-12


def tuples_exact_max(m, orbit):
    out = []
    pairs = [(p, q) for q in range(1, m + 1) for p in range(1, q + 1)]
    for j1 in range(1, m + 1):
        for P in pairs:
            for Q in pairs:
                if orbit == 2 and Q < P:
                    continue
                if orbit == 1 and Q < P:      # orbit 1: beta=0, D-pair only constrains the grid; keep both? see note
                    pass
                J = (j1, P[0], P[1], Q[0], Q[1])
                if max(J) != m or sum(J) % 2:
                    continue
                out.append(J)
    return out


def bip_block(S, W):
    ev = [i for i, (b, c) in enumerate(S) if ((b + c) // 2) % 2 == 0]
    od = [i for i, (b, c) in enumerate(S) if ((b + c) // 2) % 2 == 1]
    return W[np.ix_(ev, od)]


def analyse_sv(s, dim):
    s = np.sort(s)
    smax = s[-1] if s.size else 0.0
    thr = TOL * max(smax, 1.0) * max(dim, 1)
    nz = s[s > thr]
    z = s[s <= thr]
    nzero_block = z.size
    return (float(nz[0]) if nz.size else None, float(z[-1]) if z.size else 0.0, int(nzero_block))


_orb = None


def work(J):
    al, be = AB[_orb]
    S, W = W_matrix(J, al, be)
    if len(S) < 2:
        return J, len(S), None, 0.0
    B = bip_block(S, W)
    if B.size == 0:
        return J, len(S), None, 0.0
    s = np.linalg.svd(B, compute_uv=False)
    q, mz, _ = analyse_sv(s, len(S))
    return J, len(S), q, mz


def init(o):
    global _orb
    _orb = o


def run_numpy(tups, orbit, workers):
    with Pool(workers, initializer=init, initargs=(orbit,)) as pool:
        return list(pool.imap_unordered(work, tups, chunksize=16))


def run_torch(tups, orbit):
    import torch
    init(orbit)
    al, be = AB[orbit]
    groups = {}
    for J in tups:
        S, W = W_matrix(J, al, be)
        if len(S) < 2:
            continue
        B = bip_block(S, W)
        if B.size == 0:
            continue
        groups.setdefault(B.shape, []).append((J, len(S), B))
    res = []
    for shp, items in groups.items():
        T = torch.as_tensor(np.stack([x[2] for x in items]), device="cuda", dtype=torch.float64)
        sv = torch.linalg.svdvals(T).cpu().numpy()
        for (J, d, _), s in zip(items, sv):
            q, mz, _ = analyse_sv(s, d)
            res.append((J, d, q, mz))
    torch.cuda.synchronize()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--Mmin", type=int, default=1)
    ap.add_argument("--orbit", type=int, default=2)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", type=int, default=None)
    a = ap.parse_args()
    if a.bench:
        tups = tuples_exact_max(a.bench, a.orbit)
        t0 = time.time()
        if a.backend == "numpy":
            r = run_numpy(tups, a.orbit, a.workers)
        else:
            r = run_torch(tups, a.orbit)
        dt = time.time() - t0
        q = min(x[2] for x in r if x[2] is not None)
        print(json.dumps(dict(bench=a.bench, orbit=a.orbit, backend=a.backend, workers=a.workers,
                              tuples=len(tups), total_s=round(dt, 2), min_q=q)))
        return
    with open(a.out, "w") as fo:
        for m in range(a.Mmin, a.M + 1):
            tups = tuples_exact_max(m, a.orbit)
            t0 = time.time()
            r = run_numpy(tups, a.orbit, a.workers) if a.backend == "numpy" else run_torch(tups, a.orbit)
            dt = time.time() - t0
            good = [x for x in r if x[2] is not None]
            best = min(good, key=lambda x: x[2]) if good else None
            amb = max((x[3] / x[2] for x in good if x[3] > 0), default=0.0)
            # per-j1 minima (orbit 2,3 diagnostics)
            perj1 = {}
            for J, d, q, mz in good:
                if J[0] not in perj1 or q < perj1[J[0]][0]:
                    perj1[J[0]] = (q, J)
            line = dict(orbit=a.orbit, m=m, ntuples=len(tups), q_min=best[2] if best else None,
                        argmin=best[0] if best else None, max_zero_over_q=amb, seconds=round(dt, 2),
                        per_j1={str(k): v for k, v in sorted(perj1.items())})
            fo.write(json.dumps(line) + "\n"); fo.flush()
            print(m, len(tups), best[2] if best else None, best[0] if best else None, f"{dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()

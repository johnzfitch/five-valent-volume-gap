"""j1 = 3/2 (2s = 3): two zigzags.  For every tuple, over pairings with monotone imbalance pattern and both
end choices, compute
   q        = 8 * smallest nonzero sv of T
   sB'      = 8 * sigma_min(B')                      (B' = square reduction; lam(T) >= sigma_min(B'))
   plan     = 8 * min_k sigma_min(D_k) / (1 + min(||N1 D0^-1||, ...))   (the 1D-reducible bound)
            uses 1/sigma(B') <= max_k ||D_k^-1|| + ||D1^-1|| ||N1 D0^-1||  and the symmetric variant.
Usage: python oc_s32_plan.py M out.txt
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_reduce import zig_paths, monotone_ok, ker_dim
from oc_fast import region, pairings, smin_nz


def blocks(tj, end):
    B, rk, ck, rows, cols, Z = zig_paths(tj)
    kept = []; imb = []
    for R, C in Z:
        im = len(R) - len(C); imb.append(im); R2, C2 = list(R), list(C)
        if im == 1:
            R2 = R2[1:] if end == 'low' else R2[:-1]
        if im == -1:
            C2 = C2[1:] if end == 'low' else C2[:-1]
        kept.append((R2, C2))
    return B, kept, imb


def plan_bound(tj, end):
    B, kept, imb = blocks(tj, end)
    if not monotone_ok(imb):
        return None
    (R0, C0), (R1, C1) = kept
    D0 = B[np.ix_(R0, C0)]; D1 = B[np.ix_(R1, C1)]; N1 = B[np.ix_(R1, C0)]
    Bp = B[np.ix_(R0 + R1, C0 + C1)]
    sB = np.linalg.svd(Bp, compute_uv=False).min() if Bp.size else np.inf
    i0 = np.linalg.inv(D0) if D0.size else np.zeros((0, 0))
    i1 = np.linalg.inv(D1) if D1.size else np.zeros((0, 0))
    n0 = np.linalg.norm(i0, 2) if i0.size else 0.0
    n1 = np.linalg.norm(i1, 2) if i1.size else 0.0
    if D0.size and D1.size and N1.size:
        a = np.linalg.norm(N1 @ i0, 2); b = np.linalg.norm(i1 @ N1, 2)
        inv_bound = min(max(n0, n1) + n1 * a, max(n0, n1) + b * n0)
        th = (a, b) if max(n0, n1) + n1 * a <= max(n0, n1) + b * n0 else (b, a)
    else:
        inv_bound = max(n0, n1); th = (0, 0)
    smin_D = 1 / max(n0, n1) if max(n0, n1) > 0 else float('inf')
    return sB, 1 / inv_bound, th, imb, smin_D


if __name__ == '__main__':
    M, out = int(sys.argv[1]), sys.argv[2]
    t0 = time.time(); rows = []; nomono = 0; theta_max = 0; thw = None; sD_min = (9e9, None)
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    tj = (3, a, b, c, d)
                    if sum(tj) % 2 or len(region(tj)[0]) < 2:
                        continue
                    best = None
                    for P in pairings(tj):
                        for end in ('low', 'high'):
                            r = plan_bound(P, end)
                            if r is None:
                                continue
                            if best is None or r[1] > best[1]:
                                best = (r[0], r[1], r[2], r[3], P, end, r[4])
                    if best is None:
                        nomono += 1; continue
                    lam = smin_nz(blocks(tj, 'low')[0])[0]
                    rows.append((8 * best[1], 8 * best[0], 8 * lam, tj, best[4], best[5], best[3]))
                    if best[2][0] > theta_max:
                        theta_max = best[2][0]; thw = tj
                    if 8 * best[6] < sD_min[0]:
                        sD_min = (8 * best[6], tj)
    rows.sort()
    L = [f"j1=3/2, all tuples with 2j<={M}: {len(rows)} tuples, {nomono} without monotone pairing, {time.time()-t0:.1f}s",
         f"max over tuples of the coupling norm used at the chosen configuration (||N D0^-1|| or ||D1^-1 N||): {theta_max:.4f} at {thw}",
         f"min over tuples of 8*min_k sigma_min(D_k) at the chosen configuration: {sD_min[0]:.4f} at {sD_min[1]}",
         "smallest plan bounds (8*plan, 8*sigma(B'), q, tuple, pairing, end, imbalance):"]
    for r in rows[:15]:
        L.append(f"  {r[0]:.4f} {r[1]:.4f} {r[2]:.4f} {r[3]} {r[4]} {r[5]} {r[6]}")
    L.append(f"min ratio plan/q = {min(r[0]/r[2] for r in rows):.4f};  min ratio sigma(B')/q = {min(r[1]/r[2] for r in rows):.4f}")
    L.append(f"violations plan > sigma(B'): {sum(1 for r in rows if r[0] > r[1]*(1+1e-9))};  sigma(B') > q: {sum(1 for r in rows if r[1] > r[2]*(1+1e-9))}")
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

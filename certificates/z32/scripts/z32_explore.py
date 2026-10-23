"""Exploration: validate the zigzag decomposition against the full grid operator and evaluate the
tridiagonal two-chain bound
    sigma_min(B')^2 >= max_tau min( lam_min(D0'^T D0' - tau N'^T N'),  tau/(1+tau) lam_min(D1'^T D1') )
over admissible ordered pairings and all single-site deletions.
Usage: python z32_explore.py M
"""
import sys, os, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, pairings6, monotone, tuples32

TAUS = np.linspace(0.02, 0.74, 37)


def full_q(z):
    D0, N, D1 = z.mats()
    B = np.zeros((D0.shape[0] + N.shape[0], D0.shape[1] + D1.shape[1]))
    B[:D0.shape[0], :D0.shape[1]] = D0
    B[D0.shape[0]:, :D0.shape[1]] = N
    B[D0.shape[0]:, D0.shape[1]:] = D1
    if B.size == 0:
        return None, 0
    sv = np.linalg.svd(B, compute_uv=False)
    thr = 1e-10 * max(sv.max(), 1) * sum(B.shape)
    nz = sv[sv > thr]
    ker = sum(B.shape) - 2 * nz.size
    return (8 * nz.min() if nz.size else None), ker


def lmin(M):
    return np.linalg.eigvalsh(M)[0] if M.size else np.inf


def bound_cfg(z, keep):
    R0, C0, R1, C1 = keep
    D0, N, D1 = z.mats(R0, C0, R1, C1)
    G0 = D0.T @ D0; NN = N.T @ N; G1 = D1.T @ D1
    c1 = lmin(G1)
    if G0.size == 0:
        return c1, None
    best = (-1, None)
    for t in TAUS:
        v = min(lmin(G0 - t * NN), t / (1 + t) * c1)
        if v > best[0]:
            best = (v, t)
    return best


def tuple_bound(tj):
    zs = [Z32(P) for P in pairings6(tj)]
    S = [abs(z.imb[0]) + abs(z.imb[1]) for z in zs]
    Smin = min(S[k] for k in range(6) if monotone(zs[k].imb))
    best = (-1, None)
    for z, s in zip(zs, S):
        if s != Smin:
            continue
        for keep in z.deletions():
            v, t = bound_cfg(z, keep)
            if v > best[0]:
                best = (v, (z.tj, z.imb, t, [len(x) for x in keep]))
    return best, Smin, zs


if __name__ == '__main__':
    M = int(sys.argv[1])
    t0 = time.time(); res = []
    for tj in tuples32(M):
        z = Z32(tj)
        if len(z.R0) + len(z.C0) + len(z.R1) + len(z.C1) < 2:
            continue
        (v, info), Smin, zs = tuple_bound(tj)
        q, ker = full_q(zs[0])
        if ker != Smin:
            print('KERNEL MISMATCH', tj, ker, Smin)
        b8 = 8 * np.sqrt(max(v, 0))
        res.append((b8, q, tj, info))
    res.sort(key=lambda r: r[0])
    print(f'{len(res)} tuples, {time.time()-t0:.1f}s')
    for r in res[:25]:
        print(f'{r[0]:.4f}  q={r[1]:.4f}  {r[2]}  {r[3]}')
    print('violations bound>q:', sum(1 for r in res if r[1] is not None and r[0] > r[1] * (1 + 1e-9)))

"""Exploration: the no-deletion forms.
If an admissible ordered pairing has imbalances in {0,+1} (all tall), B has full column rank and
   lam(T)^2 = lam_min(B^T B) >= max_tau min( lam_min(D0^T D0 - tau N^T N), tau/(1+tau) lam_min(D1^T D1) )
on the FULL blocks (column form).  If all imbalances are in {0,-1}, use B^T (row form):
   lam(T)^2 >= max_tau min( tau/(1+tau) lam_min(D0 D0^T), lam_min(D1 D1^T - tau N N^T) ).
Reports per-regime minima of 8*bound and of the ingredients.
Usage: python z32_fullform.py M
"""
import sys, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, pairings6, monotone, tuples32
from z32_cover import label

TAUS = np.concatenate([np.linspace(0.005, 0.1, 20), np.linspace(0.11, 0.749, 60)])


def lmin(M):
    return np.linalg.eigvalsh(M)[0] if M.size else np.inf


def colform(z):
    D0, N, D1 = z.mats()
    G0 = D0.T @ D0; NN = N.T @ N; c1 = lmin(D1.T @ D1)
    if G0.size == 0:
        return c1, None
    best = (-1, None)
    for t in TAUS:
        v = min(lmin(G0 - t * NN), t / (1 + t) * c1)
        if v > best[0]:
            best = (v, t)
    return best


def rowform(z):
    D0, N, D1 = z.mats()
    H1 = D1 @ D1.T; NN = N @ N.T; c0 = lmin(D0 @ D0.T)
    if H1.size == 0:
        return c0, None
    best = (-1, None)
    for t in TAUS:
        v = min(t / (1 + t) * c0, lmin(H1 - t * NN))
        if v > best[0]:
            best = (v, t)
    return best


if __name__ == '__main__':
    M = int(sys.argv[1])
    worst = {}; cnt = collections.Counter(); tuple_best = []
    for tj in tuples32(M):
        zs = [Z32(P) for P in pairings6(tj)]
        if sum(len(x) for x in (zs[0].R0, zs[0].C0, zs[0].R1, zs[0].C1)) < 2:
            continue
        S = [abs(z.imb[0]) + abs(z.imb[1]) for z in zs]
        Smin = min(S[k] for k in range(6) if monotone(zs[k].imb))
        tb = (-1, None)
        for z, s in zip(zs, S):
            if s != Smin:
                continue
            forms = []
            if min(z.imb) >= 0:
                forms.append(('col', colform(z)))
            if max(z.imb) <= 0:
                forms.append(('row', rowform(z)))
            for f, (v, t) in forms:
                b8 = 8 * np.sqrt(max(v, 0))
                key = (label(z), f) if z.P[0] < z.P[2] else (('swapped',) + label(Z32((3,) + z.tj[3:5] + z.tj[1:3])), f)
                cnt[key] += 1
                if key not in worst or b8 < worst[key][0]:
                    worst[key] = (b8, z.tj, t)
                if b8 > tb[0]:
                    tb = (b8, key)
        tuple_best.append((tb[0], tj, tb[1]))
    for k in sorted(worst, key=str):
        print(k, cnt[k], f'worst {worst[k][0]:.3f} at {worst[k][1]} tau={worst[k][2]}')
    tuple_best.sort()
    print('smallest per-tuple best:')
    for r in tuple_best[:12]:
        print(f'  {r[0]:.4f} {r[1]} {r[2]}')

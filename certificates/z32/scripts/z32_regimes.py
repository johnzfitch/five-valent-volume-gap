"""Exploration: regime structure for j1=3/2 in the orientation n1 < n2.
For each tuple (2j<=M) and each of the 3 pairings oriented with n1<n2, record the zigzag end types,
imbalances, admissibility, and the tridiagonal bound for every deletion position.
Usage: python z32_regimes.py M
"""
import sys, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, pairings6, monotone, tuples32
from z32_explore import bound_cfg


def ends(R, C):
    if not R and not C:
        return '--'
    lo = 'C' if (C and (not R or C[0] <= R[0])) else 'R'
    hi = 'R' if (R and (not C or R[-1] >= C[-1])) else 'C'
    return lo + hi


def oriented(tj):
    out = []
    for P in pairings6(tj):
        z = Z32(P)
        n1, m1, n2, m2 = z.P
        if n1 < n2:
            out.append(z)
    return out


if __name__ == '__main__':
    M = int(sys.argv[1])
    stats = collections.Counter(); worst = {}
    for tj in tuples32(M):
        zs = [Z32(P) for P in pairings6(tj)]
        if sum(len(x) for x in (zs[0].R0, zs[0].C0, zs[0].R1, zs[0].C1)) < 2:
            continue
        S = [abs(z.imb[0]) + abs(z.imb[1]) for z in zs]
        Smin = min(S[k] for k in range(6) if monotone(zs[k].imb))
        for z, s in zip(zs, S):
            n1, m1, n2, m2 = z.P
            if n1 > n2 or s != Smin:
                continue
            key = (ends(z.R0, z.C0), ends(z.R1, z.C1), z.imb, 'corner' if min(m1, m2) < 1 else '')
            # best deletion
            best = (-1, None)
            for keep in z.deletions():
                v, t = bound_cfg(z, keep)
                if v > best[0]:
                    best = (v, keep, t)
            b8 = 8 * np.sqrt(max(best[0], 0))
            stats[key] += 1
            if key not in worst or b8 < worst[key][0]:
                worst[key] = (b8, z.tj, z.P, best[2])
    for k in sorted(stats):
        w = worst[k]
        print(k, stats[k], f'worst {w[0]:.3f} at {w[1]} P={tuple(str(x) for x in w[2])} tau={w[3]}')

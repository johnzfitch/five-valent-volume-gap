"""Exploration: which regimes are admissible per tuple (orientation n1<n2), and a greedy hitting set.
Regime label = (zz0 ends, zz1 ends, dn==1/2?, dm class, corner).
Usage: python z32_cover.py M
"""
import sys, os, collections
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, pairings6, monotone, tuples32
from z32_regimes import ends


def label(z):
    n1, m1, n2, m2 = z.P
    dn, dm = n2 - n1, m2 - m1
    dmc = 'dm<=-3/2' if dm <= Fr(-3, 2) else ('dm=-1/2' if dm == Fr(-1, 2) else ('dm=1/2' if dm == Fr(1, 2) else 'dm>=3/2'))
    corner = 'corner' if (m1, m2) in ((0, Fr(1, 2)), (Fr(1, 2), 0)) else ''
    return (ends(z.R0, z.C0), ends(z.R1, z.C1), 'dn=1/2' if dn == Fr(1, 2) else 'dn>=3/2', dmc, corner)


if __name__ == '__main__':
    M = int(sys.argv[1])
    sets = []
    cnt = collections.Counter()
    for tj in tuples32(M):
        zs = [Z32(P) for P in pairings6(tj)]
        if sum(len(x) for x in (zs[0].R0, zs[0].C0, zs[0].R1, zs[0].C1)) < 2:
            continue
        S = [abs(z.imb[0]) + abs(z.imb[1]) for z in zs]
        Smin = min(S[k] for k in range(6) if monotone(zs[k].imb))
        labs = frozenset(label(z) for z, s in zip(zs, S) if s == Smin and z.P[0] < z.P[2])
        sets.append((tj, labs))
        for l in labs:
            cnt[l] += 1
    print(len(sets), 'tuples')
    for l, c in sorted(cnt.items(), key=lambda x: -x[1]):
        only = sum(1 for _, s in sets if s == frozenset([l]))
        print(c, only, l)
    # greedy hitting set
    remaining = list(sets); chosen = []
    while remaining:
        c2 = collections.Counter()
        for _, s in remaining:
            for l in s:
                c2[l] += 1
        l, c = c2.most_common(1)[0]
        chosen.append((l, c))
        remaining = [(t, s) for t, s in remaining if l not in s]
    print('greedy cover:')
    for l, c in chosen:
        print('  ', c, l)

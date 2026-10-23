"""Exploration: regime structure for 2j1 = 5.  For every tuple (2j<=M) list the all-tall ordered pairings with
(chain types, dn, dm, m1, m2) and count distinct 'labels'; then a greedy hitting set of labels.
Usage: python zk_regimes.py M"""
import sys, os, collections
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zk_core import ZK, pairings6, tuplesK
M = int(sys.argv[1])
def cls(x, cuts):
    for c in cuts:
        if x <= c: return f'<={c}'
    return f'>{cuts[-1]}'
sets = []; cnt = collections.Counter()
for tj in tuplesK(5, M):
    z0 = ZK(tj)
    if sum(len(r) + len(c) for r, c in zip(z0.R, z0.C)) < 2:
        continue
    labs = set()
    for P in pairings6(tj):
        z = ZK(P)
        if min(z.imb) < 0:
            continue
        dn, dm = z.n2 - z.n1, z.m2 - z.m1
        corner = 'c' if min(z.m1, z.m2) < Fr(5, 2) else ''
        lab = (tuple(z.ends(k) for k in range(3)), 'dn' + cls(dn, [Fr(-3,2), Fr(-1,2), Fr(1,2), Fr(3,2)]),
               'dm' + cls(dm, [Fr(-5,2), Fr(-3,2), Fr(-1,2), Fr(1,2), Fr(3,2)]), corner)
        labs.add(lab)
    sets.append((tj, frozenset(labs)))
    for l in labs: cnt[l] += 1
print(len(sets), 'tuples;', len(cnt), 'labels')
rem = list(sets); chosen = []
while rem:
    c2 = collections.Counter()
    for _, s in rem:
        for l in s: c2[l] += 1
    l, c = c2.most_common(1)[0]; chosen.append((l, c))
    rem = [(t, s) for t, s in rem if l not in s]
print('greedy cover:', len(chosen))
for l, c in chosen: print('  ', c, l)

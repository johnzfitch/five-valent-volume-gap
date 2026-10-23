"""Exploration: analysis-relevant regime labels for 2j1 = 5 (all-tall ordered pairings).
Label = (chain types, dn class, dm class, corner signature), where
  dn class = exact dn if |dn| <= 5/2 else sign*'big';  dm class likewise;
  corner signature = for each zigzag and each of rows/cols, whether the lower range end is set by the corner bound
  (u >= s) rather than by m1 or m2+offset.
Usage: python zk_regimes2.py M"""
import sys, os, collections
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zk_core import ZK, pairings6, tuplesK

M = int(sys.argv[1])
H = Fr(1, 2)


def cl(x):
    return str(x) if abs(x) <= Fr(5, 2) else ('+big' if x > 0 else '-big')


def corner_sig(z):
    s = z.s; sig = []
    for k in range(z.K):
        dk = z.d[k]
        for off, lst in ((0, z.R[k]), (1, z.C[k])):
            if not lst:
                sig.append('-'); continue
            lo = lst[0]
            alt = max(z.m1, z.m2 + dk + off)
            sig.append('c' if lo > alt else '.')
    return ''.join(sig)


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
        labs.add((tuple(z.ends(k) for k in range(3)), 'dn' + cl(z.n2 - z.n1), 'dm' + cl(z.m2 - z.m1), corner_sig(z)))
    sets.append((tj, frozenset(labs)))
    for l in labs:
        cnt[l] += 1
print(len(sets), 'tuples;', len(cnt), 'labels')
rem = list(sets); chosen = []
while rem:
    c2 = collections.Counter()
    for _, s in rem:
        for l in s:
            c2[l] += 1
    l, c = c2.most_common(1)[0]; chosen.append((l, c))
    rem = [(t, s) for t, s in rem if l not in s]
print('greedy cover:', len(chosen))
for l, c in chosen:
    print('  ', c, l)

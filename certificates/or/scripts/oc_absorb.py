"""Non-monotone zigzag patterns: is the (wide before tall) pair absorbed (kernel drops), cancelled (kernel =
sum|imb|), or partially absorbed?  Tabulated against sign((n1-n2)(m1-m2)).  All labelled half-integer-s tuples
2j_i <= M, pairing (23|45) as given.  Usage: python oc_absorb.py M out.txt"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_reduce import zig_paths, monotone_ok, ker_dim
from oc_fast import region
from collections import Counter
M, out = int(sys.argv[1]), sys.argv[2]
c = Counter(); ex = {}
for tj in itertools.product(range(1, M + 1), repeat=5):
    if tj[0] % 2 == 0 or sum(tj) % 2 or len(region(tj)[0]) < 2:
        continue
    B, rk, ck, rows, cols, Z = zig_paths(tj)
    imb = [len(R) - len(C) for R, C in Z]
    if monotone_ok(imb):
        continue
    S = sum(map(abs, imb)); kd = ker_dim(B)
    a = sum(1 for x in imb if x == -1); t = sum(1 for x in imb if x == 1)
    dn = (tj[1] + tj[2] - tj[3] - tj[4]) / 2; dm = (abs(tj[1] - tj[2]) - abs(tj[3] - tj[4])) / 2
    kind = 'cancel' if kd == S else ('full absorption' if kd == S - 2 * min(a, t) else 'partial absorption')
    key = (kind, 'dn*dm>0' if dn * dm > 0 else ('dn*dm<0' if dn * dm < 0 else 'dn*dm=0'))
    c[key] += 1; ex.setdefault(key, (tj, imb, kd))
L = [f"non-monotone patterns, labelled half-integer-s tuples 2j_i<={M}:"]
for k in sorted(c):
    L.append(f"  {k[0]:18s} {k[1]:8s} {c[k]:6d}   e.g. {ex[k]}")
open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

"""Validate ls_core against [OR]'s oc_fast builder (imported read-only; itself validated against a brute-force
tensor product in [OR]) on all labelled tuples with 2j_i <= 7, against the [GD] per-level minima for
2j_max <= 12 (exhaustive over multisets of legs 2-5), and sparse vs dense on medium tuples.
Usage: python ls_validate.py > ../outputs/validate_out.txt
"""
import sys, os, itertools, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', '..', 'or', 'scripts'))
import ls_core as L
import oc_fast

t0 = time.time()
worst = 0.0; nchk = 0
for tj in itertools.product(range(1, 8), repeat=5):
    if sum(tj) % 2:
        continue
    if L.dim_inv(tj) < 2:
        continue
    o = oc_fast.build(tj, ('T',))
    l, _ = oc_fast.smin_nz(o['T'])
    q, ker = L.qmin(tj)
    if l is None and q is None:
        continue
    a = 8 * l; b = q[0]
    worst = max(worst, abs(a - b) / a); nchk += 1
print(f"[V1] ls_core vs oc_fast on {nchk} labelled tuples with 2j_i<=7: max rel diff {worst:.2e}")

GD = {2: 4.557960739, 3: 5.564315647, 4: 6.512488857, 5: 7.418030814, 6: 8.290425521, 7: 9.135988056,
      8: 9.959187326, 9: 10.76333088, 10: 11.55095264, 11: 12.32404776, 12: 13.08422255}


def tuples_at_level(M):
    for s2 in range(1, M + 1):
        for js in itertools.combinations_with_replacement(range(1, M + 1), 4):
            if max(s2, js[-1]) != M or (s2 + sum(js)) % 2:
                continue
            yield (s2,) + js


for M in range(2, 13):
    best = (1e99, None)
    for tj in tuples_at_level(M):
        if L.dim_inv(tj) < 2:
            continue
        q, _ = L.qmin(tj)
        if q is not None and q[0] < best[0]:
            best = (q[0], tj)
    print(f"[V2] 2j_max={M:2d}: min q = {best[0]:.9f} at {best[1]}   GD digest {GD[M]:.9f}   diff {abs(best[0]-GD[M]):.1e}")

for tj in [(3, 300, 301, 301, 301), (1, 599, 600, 600, 600), (9, 100, 101, 103, 105), (21, 60, 61, 70, 72)]:
    qd, kd = L.qmin(tj, dense_max=10 ** 6, want=3)
    qs, ks = L.qmin(tj, dense_max=0, want=3)
    print(f"[V3] {tj}: dense {np.round(qd, 8)} ker {kd} | sparse {np.round(qs, 8)} ker {ks}")
print(f"time {time.time() - t0:.1f}s")

"""Minimum of q over the other four spins at fixed j1 (exhaustive over multisets 2j2<=..<=2j5 <= Mmax),
to test whether L = j1 controls the gap linearly.  Uses the S4 symmetry of legs 2-5 ([DC] Prop. 1(d)).
Usage: python ls_fixed_j1.py Mmax 2j1 [2j1 ...] > out
"""
import sys, os, itertools, math, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ls_core as L
Mmax = int(sys.argv[1])
for s2 in map(int, sys.argv[2:]):
    t0 = time.time(); best = []
    for js in itertools.combinations_with_replacement(range(1, Mmax + 1), 4):
        if (s2 + sum(js)) % 2 or js[3] > s2 + js[0] + js[1] + js[2] or s2 > sum(js):
            continue
        tj = (s2,) + js
        if L.dim_inv(tj) < 2:
            continue
        q, _ = L.qmin(tj, dense_max=4000)
        if q is None:
            continue
        best.append((float(q[0]), tj))
    best.sort()
    print(f"2j1={s2}: {len(best)} tuples with 2j<= {Mmax}; 5 smallest q: " +
          "; ".join(f"{q:.4f} {tj}" for q, tj in best[:5]) + f"   q/(2j1) min = {best[0][0]/s2:.4f}  ({time.time()-t0:.0f}s)", flush=True)

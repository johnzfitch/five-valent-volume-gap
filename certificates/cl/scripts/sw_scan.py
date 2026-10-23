"""Conjecture SW (shifted windows of a Stark matrix), numerical scan.
For the Stark matrix H(n,m) (N sites) and every window [a,b] (0-based, b <= N-2):
   up  : sigma_min of rows [a+1,b+1] x cols [a,b] of (H - 1)
   down: sigma_min of rows [a,b]     x cols [a+1,b+1] of (H - 1)
Reports the minimum over N <= Nmax, 2m <= M2max, and checks the two reductions used in the text:
 (R1) for fixed a, the 'up' value is minimised at b = N-2 (rows above b+1 receive nothing);
 (R2) over a, the 'up' minimum at b = N-2 is attained at a = 0 (observed, not proved).
Usage: python sw_scan.py Nmax M2max
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_windows import stark_full

Nmax, M2 = int(sys.argv[1]), int(sys.argv[2])
best = (9e9, None); r1_ok = True; r2_fail = 0; cnt = 0
for m2 in range(0, M2 + 1):
    for N in range(2, Nmax + 1):
        _, H = stark_full(m2 + 2 * N, m2)
        A = H - np.eye(N)
        anch = []
        for a in range(0, N - 1):
            vals = []
            for b in range(a, N - 1):
                up = np.linalg.svd(A[a + 1:b + 2, a:b + 1], compute_uv=False).min()
                dn = np.linalg.svd(A[a:b + 1, a + 1:b + 2], compute_uv=False).min()
                cnt += 2
                vals.append(up)
                for v, tag in ((up, 'up'), (dn, 'down')):
                    if v < best[0]:
                        best = (v, (N, m2 / 2, a, b, tag))
            if min(vals) < vals[-1] - 1e-12:
                r1_ok = False
            anch.append(vals[-1])
        if anch and min(anch) < anch[0] - 1e-12:
            r2_fail += 1
print(f"SW scan: {cnt} windows, N<=%d, m<=%s" % (Nmax, M2 / 2))
print(f"  minimum sigma_min = {best[0]:.12f} at (N, m, a, b, orientation) = {best[1]}")
print(f"  (R1) fixed a: minimum over b attained at b=N-2: {r1_ok}")
print(f"  (R2) number of (N,m) where the anchored minimum is not at a=0: {r2_fail}")

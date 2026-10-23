"""Growth of q along explicit one-parameter families, to large spin.
Prints q, q/M (M = 2j_max), q*ln(M)/M, and the local exponent d ln q / d ln M.
Usage: python ls_families.py > ../outputs/families_out.txt
"""
import sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ls_core as L

FAM = {
    'G1  (1/2; j-1/2, j, j, j)      [GD per-level minimiser]': lambda a: (1, a - 1, a, a, a),
    'A3  (3/2; j-1/2, j, j, j)      [OR thin family]': lambda a: (3, a - 1, a, a, a),
    'A5  (5/2; j-1/2, j, j, j)': lambda a: (5, a - 1, a, a, a),
    'A7  (7/2; j-1/2, j, j, j)': lambda a: (7, a - 1, a, a, a),
    'E1  (1; j, j, j, j)            [integer j1]': lambda a: (2, a, a, a, a),
    'E2  (2; j, j, j, j)': lambda a: (4, a, a, a, a),
    'L1  (1/2; 1/2, j, j, j+1/2)?   [ladder-type]': lambda a: (1, 1, a, a, a + 1) if (1 + 1 + 3 * a + 1) % 2 == 0 else (1, 1, a, a + 1, a + 1),
    'K1  (j; 1/2, j, 1, 1)          [large-j1 minimiser]': lambda a: (a, 1, a, 2, 2) if (a + 1 + a + 4) % 2 == 0 else (a, 1, a + 1, 2, 2),
    'F   (j; j, j, j, j)            [fat]': lambda a: (a, a, a, a, a) if (5 * a) % 2 == 0 else (a, a, a, a, a + 1),
    'H   (j/2; j, j, j, j)': lambda a: ((a // 2) | 1 if a % 2 == 0 else a // 2 * 2, a, a, a, a),
}

LEVELS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
CAPDIM = 60000

t0 = time.time()
for name, f in FAM.items():
    print(f"== {name}")
    prev = None
    for M in LEVELS:
        tj = f(M)
        if sum(tj) % 2:
            continue
        nS = L.dim_inv(tj)
        if nS > CAPDIM:
            break
        q, ker = L.qmin(tj)
        if q is None:
            continue
        q = float(q[0]); Mx = max(tj)
        loc = '' if prev is None else f"  local exponent {math.log(q / prev[1]) / math.log(Mx / prev[0]):.3f}"
        print(f"  {tj}  dim {nS:6d} ker {ker:4d}  q = {q:12.4f}  q/M = {q / Mx:.4f}  q*lnM/M = {q * math.log(Mx) / Mx:.4f}{loc}", flush=True)
        prev = (Mx, q)
print(f"time {time.time() - t0:.1f}s")

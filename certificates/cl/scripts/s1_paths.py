"""j1 = 1: the class V_0 = {b = c} is one diagonal; T^2 on V_0 = A+ A+^T + A- A-^T with A+- the
bipartite blocks of the two paths (d in {0,1}) and (d in {-1,0}).  Check
  lam(T)^2 >= g+^2 + g-^2,  g+- = sigma_min(A+-^T restricted to V_0)  (0 if V_0 is the majority in that path)
and report how often V_0 is the majority in both paths (bound useless there).
Usage: python s1_paths.py M
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import grid
from explore_T0 import nzmin

M = int(sys.argv[1])
worst = (9e9, None); bad = []; n = 0
for j2 in range(1, M + 1):
    for j3 in range(j2, M + 1):
        for j4 in range(1, M + 1):
            for j5 in range(j4, M + 1):
                if (j2, j3) > (j4, j5): continue
                tj = (2, j2, j3, j4, j5)
                if sum(tj) % 2: continue
                S, W = grid(tj)
                if len(S) < 2: continue
                lt, _ = nzmin(W, S)
                if lt is None: continue
                n += 1
                V0 = [i for i, (b, c) in enumerate(S) if b == c]
                g = []
                for dd in (2, -2):
                    V1 = [i for i, (b, c) in enumerate(S) if b - c == dd]
                    A = W[np.ix_(V0, V1)]
                    if len(V1) < len(V0):
                        g.append(0.0); continue
                    sv = np.linalg.svd(A, compute_uv=False)
                    g.append(sv.min() if len(sv) == len(V0) else 0.0)
                bnd = np.sqrt(g[0] ** 2 + g[1] ** 2)
                if bnd > lt * (1 + 1e-9): print('VIOLATION', tj, lt, bnd)
                if bnd == 0: bad.append(tj)
                elif bnd / lt < worst[0]: worst = (bnd / lt, tj, 8 * lt, 8 * bnd)
print('tuples', n, 'bound zero in', len(bad), bad[:10])
print('worst ratio bound/true', worst)

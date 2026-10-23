"""Chained bound vs true q along the 2j1=7 family (7, a, a, a+2, a+3) with a fine log-grid tau search
(each tau_k in 10^[-4.5, 0], 3 coordinate sweeps from several starts), all all-tall ordered pairings.
Usage: python zk_s72_family.py out.txt"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zk_core import ZK, pairings6
out = sys.argv[1]; L = []
grid = list(10 ** np.linspace(-6, 0, 61))
for a in [int(x) for x in sys.argv[2].split(",")]:
    tj = (7, a, a, a + 2, a + 3)
    best = (-math.inf, None)
    for P in pairings6(tj):
        z = ZK(P)
        if min(z.imb) < 0:
            continue
        D, N = z.mats(); K = z.K
        G = [d.T @ d for d in D]; NN = [None] + [n.T @ n for n in N[1:]]
        def fb(t):
            v = []
            for k in range(K):
                ck = 1.0 if k == 0 else t[k-1] / (1 + t[k-1])
                Mk = ck * G[k]
                if k + 1 < K and NN[k+1].size and Mk.size:
                    Mk = Mk - t[k] * NN[k+1]
                if Mk.size:
                    v.append(np.linalg.eigvalsh(Mk)[0])
            return min(v)
        for st in (1e-3, 1e-2, 0.05, 0.2):
            t = [st] * (K - 1); cur = fb(t)
            for _ in range(3):
                for j in range(K - 1):
                    for g in grid:
                        tt = list(t); tt[j] = g; v = fb(tt)
                        if v > cur: cur, t = v, tt
            if cur > best[0]: best = (cur, P, z.imb, t)
    q = 8 * np.linalg.svd(ZK(tj).full(), compute_uv=False).min()
    v = best[0]
    L.append(f'a={a}: chained 8*sqrt(bound) = {8*math.sqrt(v) if v > 0 else float("nan"):.4f} (raw {v:.4g}), true q = {q:.3f}, via {best[1]} imb {best[2]}')
    print(L[-1], flush=True)
open(out, 'w').write('\n'.join(L) + '\n')

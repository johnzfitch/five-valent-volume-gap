"""Exact chained certificate for given tuples with a fine float tau search (log grid 1e-6..1), all all-tall pairings.
Usage: python zk_cert_one.py out.txt "7,27,27,29,30" ["..."]"""
import sys, os, math
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zk_core import ZK, pairings6
from zk_exact_cert import bound, psd
out = sys.argv[1]; L = []
grid = list(10 ** np.linspace(-6, 0, 61))
for spec in sys.argv[2:]:
    tj = tuple(int(x) for x in spec.split(','))
    best = (-math.inf, None, None)
    for P in pairings6(tj):
        z = ZK(P)
        if min(z.imb) < 0: continue
        D, N = z.mats(); K = z.K
        G = [d.T @ d for d in D]; NN = [None] + [n.T @ n for n in N[1:]]
        def fb(t):
            v = []
            for k in range(K):
                ck = 1.0 if k == 0 else t[k-1] / (1 + t[k-1]); Mk = ck * G[k]
                if k + 1 < K and NN[k+1].size and Mk.size: Mk = Mk - t[k] * NN[k+1]
                if Mk.size: v.append(np.linalg.eigvalsh(Mk)[0])
            return min(v)
        for st in (1e-3, 1e-2, 0.05, 0.2):
            t = [st] * (K - 1); cur = fb(t)
            for _ in range(3):
                for j in range(K - 1):
                    for g in grid:
                        tt = list(t); tt[j] = g; v = fb(tt)
                        if v > cur: cur, t = v, tt
            if cur > best[0]: best = (cur, z, t)
    v, z, t = best
    taus = [Fr(x).limit_denominator(10 ** 7) for x in t]
    sig2 = None; ok = True
    for (lv, dg, o2) in bound(z, taus):
        if lv == math.inf: continue
        if lv <= 0:
            ok = False; break
        s = Fr(lv * (1 - 1e-6)).limit_denominator(10 ** 9)
        if not psd(dg, o2, s):
            ok = False; break
        sig2 = s if sig2 is None else min(sig2, s)
    q = 8 * np.linalg.svd(ZK(tj).full(), compute_uv=False).min()
    if ok:
        L.append(f'{tj}: certified q >= {8*math.sqrt(float(sig2)):.4f} (true q = {q:.3f}) via {z.tj} taus {[str(x) for x in taus]}')
    else:
        L.append(f'{tj}: NOT certified; best float chained bound sigma^2 >= {v:.4g} (<= 0 means the inequality gives nothing); true q = {q:.3f}; best pairing {z.tj}')
    print(L[-1])
open(out, 'w').write('\n'.join(L) + '\n')

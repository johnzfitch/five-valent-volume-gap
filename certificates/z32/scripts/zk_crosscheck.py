"""Cross-check zk_core (general s) against [OR]'s oc_fast (read-only import), and the coupling identity:
N_k^T N_k - kappa_k D_{k-1}^T D_{k-1} is diagonal with entries Lambda_k (exact formula), on full blocks.
Usage: python zk_crosscheck.py 2s M out.txt"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zk_core import ZK, pairings6, tuplesK
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'or', 'scripts')))
from oc_fast import build
s2, M, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
n = 0; worst = 0.0; worstid = 0.0
for tj in tuplesK(s2, M):
    for P in pairings6(tj):
        z = ZK(P)
        Bm = z.full()
        r = build(P, kinds=('T',))
        if r is None or Bm.size == 0:
            continue
        s1 = np.sort(np.linalg.svd(Bm, compute_uv=False)); s2v = np.sort(np.linalg.svd(r['T'], compute_uv=False))
        L = max(len(s1), len(s2v))
        s1 = np.concatenate([np.zeros(L - len(s1)), s1]); s2v = np.concatenate([np.zeros(L - len(s2v)), s2v])
        worst = max(worst, float(np.abs(s1 - s2v).max() / max(1.0, s2v.max())))
        D, N = z.mats()
        for k in range(1, z.K):
            if D[k - 1].shape[1] == 0:
                continue
            Q = N[k].T @ N[k] - float(z.kappa[k]) * (D[k - 1].T @ D[k - 1])
            lam = np.array([float(z.Lam(k, c)) for c in z.C[k - 1]])
            scale = max(1.0, np.abs(D[k - 1].T @ D[k - 1]).max())
            worstid = max(worstid, float(np.abs(Q - np.diag(lam)).max() / scale))
        n += 1
msg = (f'zk_core vs oc_fast, 2j1={s2}, all ordered pairings 2j<={M}: {n} cases; max rel. singular-value diff {worst:.2e}; '
       f'max rel. deviation of N^T N - kappa D^T D from diag(Lambda) {worstid:.2e}')
print(msg); open(out, 'w').write(msg + '\n')

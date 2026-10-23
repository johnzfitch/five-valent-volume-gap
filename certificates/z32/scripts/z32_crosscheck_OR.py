"""Cross-check of z32_core (this folder) against the open-regions grid builder oc_fast (imported read-only from
research/dichotomy_open_regions_20261021/scripts, not modified): for every j1=3/2 tuple with 2j<=M and every
ordered pairing, the singular values of B = [[D0,0],[N,D1]] equal those of oc_fast's bipartite block of T.
Usage: python z32_crosscheck_OR.py M out.txt"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, pairings6, tuples32
OR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'or', 'scripts')
sys.path.insert(0, os.path.abspath(OR))
from oc_fast import build
M, out = int(sys.argv[1]), sys.argv[2]
n = 0; worst = 0.0
for tj in tuples32(M):
    for P in pairings6(tj):
        z = Z32(P); D0, N, D1 = z.mats()
        B = np.zeros((D0.shape[0] + N.shape[0], D0.shape[1] + D1.shape[1]))
        B[:D0.shape[0], :D0.shape[1]] = D0; B[D0.shape[0]:, :D0.shape[1]] = N; B[D0.shape[0]:, D0.shape[1]:] = D1
        r = build(P, kinds=('T',))
        if r is None:
            continue
        A = r['T']
        s1 = np.sort(np.linalg.svd(B, compute_uv=False)) if B.size else np.array([])
        s2 = np.sort(np.linalg.svd(A, compute_uv=False)) if A.size else np.array([])
        k = min(len(s1), len(s2))
        s1 = np.concatenate([np.zeros(len(s2) - len(s1)), s1]) if len(s2) > len(s1) else s1
        s2 = np.concatenate([np.zeros(len(s1) - len(s2)), s2]) if len(s1) > len(s2) else s2
        if s1.size:
            worst = max(worst, float(np.max(np.abs(s1 - s2)) / max(1.0, s2.max())))
        n += 1
msg = f'z32_core vs oc_fast ([OR]), all ordered pairings of j1=3/2 tuples 2j<={M}: {n} cases, max relative singular-value difference {worst:.2e}'
print(msg); open(out, 'w').write(msg + '\n')

"""Validate grid.py (closed-form nonnegative grid matrix) against the full recoupling operator (hv_core, copy of
the closed folder's validated core) for sigma = (alpha, beta, beta, 0), all labelled tuples with 2j_i <= M,
(alpha,beta) in {(2,0),(2,2),(4,2),(2,4),(1,3)}.  Reports max relative spectral discrepancy and kernel agreement."""
import sys, itertools, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hv_core import Vertex
from grid import spectrum
M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
worst = 0.0; n = 0; kerbad = 0
for J in itertools.product(range(1, M + 1), repeat=5):
    if sum(J) % 2: continue
    v = Vertex(J)
    if v.dim == 0: continue
    for al, be in [(2, 0), (2, 2), (4, 2), (2, 4), (1, 3)]:
        A = v.Qsigma({(0, 1, 2): al, (0, 1, 3): be, (0, 2, 3): be})
        s1 = np.sort(np.linalg.svd(A, compute_uv=False))
        s2 = spectrum(J, al, be)
        assert len(s1) == len(s2), (J, len(s1), len(s2))
        sc = max(s1.max(), 1.0)
        worst = max(worst, np.abs(s1 - s2).max() / sc)
        if (s1 < 1e-9 * sc).sum() != (s2 < 1e-9 * sc).sum(): kerbad += 1
        n += 1
print(f"labelled tuples 2j_i<=%d, 5 (alpha,beta) pairs: {n} cases; max |spec diff|/max(s_max,1) = {worst:.2e}; kernel-count mismatches: {kerbad}" % M)

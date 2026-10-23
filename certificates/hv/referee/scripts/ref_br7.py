"""Referee: reproduce BR II Table 10 value lambda_V = 0.000597922 (7-vertex) with the referee's
operator code (float64 after exact construction), to pin the BR normalisation Z=1, lambda=sqrt(q)."""
import sys, itertools
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from ref_exact import Space, dot_matrix, apply_pm
twoj = (2, 2, 2, 2, 2, 3, 5)
sig_list = [2, 2, 2, -4, 0, 0, 0, 0, -2, -2, 2, 2, 4, -2, 4, 4, 0, 0, 0, -2]
triples = list(itertools.combinations(range(6), 3))   # BR order 123,124,...,456 == lexicographic
sigma = dict(zip(triples, sig_list))
sp = Space(twoj, 0); d = len(sp.states)
dots = {}
def dot(I, J):
    k = (min(I, J), max(I, J))
    if k not in dots: dots[k] = np.array([[float(x) for x in row] for row in dot_matrix(sp, *k)])
    return dots[k]
def C(I, J): return dot(I, I) + dot(J, J) + 2 * dot(I, J)
A = np.zeros((d, d))
for (I, J, K), s in sigma.items():
    if s:
        a, b = C(I, J), C(J, K); A += s * (a @ b - b @ a)
sp1 = Space(twoj, 2)
Jp = np.zeros((len(sp1.states), d))
for c, st in enumerate(sp.states):
    for leg in range(7):
        r = apply_pm(twoj, st, leg, True)
        if r: Jp[sp1.index[r[1]], c] += float(r[0])
u, s, vt = np.linalg.svd(Jp)
null = vt[np.sum(s > 1e-9 * s[0]):].T
R = np.linalg.lstsq(null, A @ null, rcond=None)[0]
res = np.linalg.norm(null @ R - A @ null) / np.linalg.norm(A @ null)
ev = np.linalg.eigvals(R)
q = np.sort(np.abs(ev.imag))
print("smallest 8 |q| (float64):", ", ".join(f"{x:.6e}" for x in q[:8]), " max q =", f"{q[-1]:.4e}")
nz = q[q > 1e-11 * q[-1]]
print(f"dim Inv = {null.shape[1]}, invariance residual = {res:.1e}, max|Re ev| = {np.abs(ev.real).max():.1e}")
print(f"smallest nonzero q = {nz[0]:.10g}, lambda_V = sqrt(q) = {np.sqrt(nz[0]):.9g}  (BR II: 0.000597922)")

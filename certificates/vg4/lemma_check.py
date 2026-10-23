"""Randomised stress test of Lemma 1 (the abstract Jacobi-matrix lemma) on chains that are
NOT volume chains: random log-concave positive sequences c_1..c_{n-1}.

Checks, for each chain:
  (i)  smallest positive eigenvalue lam >= 1 / (sqrt(rho_plus * rho_minus) * S),
       rho_plus = max(1, max_k c_{k+1}/c_k), rho_minus = max(1, max_k c_k/c_{k+1}), S = sum 1/c_k;
  (ii) for even n: lam <= 2 / S_odd  (S_odd = sum over odd-indexed c_k);
  (iii) for even n: the explicit inverse formula |B^{-1}|_{ij} = (1/c_{2j-1}) prod_{l=j}^{i-1} c_{2l}/c_{2l+1}.
Also shows the lemma is sharp in the sense that log-concavity cannot be dropped
(dimerised chain example).
Run: python lemma_check.py -> lemma_check_out.txt
"""
import os, numpy as np
from scipy.linalg import eigvalsh_tridiagonal

rng = np.random.default_rng(20261021)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def lam_pos(c):
    n = len(c) + 1
    w = np.sort(eigvalsh_tridiagonal(np.zeros(n), c))
    return w[(n + 1) // 2]

worst_i = np.inf; worst_ii = 0; worst_iii = 0; N = 0
for trial in range(400000):
    n = int(rng.integers(2, 60))
    m = n - 1
    # random concave log-sequence: decreasing increments
    inc = np.sort(rng.normal(0, rng.uniform(0.01, 1.0), size=max(m - 1, 0)))[::-1]
    x = np.concatenate([[0.0], np.cumsum(inc)])
    x -= x.max()
    if x.min() < -np.log(1e6):   # keep dynamic range <= 1e6 so float64 LAPACK is reliable
        continue
    c = np.exp(x)
    lam = lam_pos(c)
    S = np.sum(1 / c)
    r = c[1:] / c[:-1] if m > 1 else np.array([1.0])
    rp = max(1.0, r.max()); rm = max(1.0, (1 / r).max())
    worst_i = min(worst_i, lam * S * np.sqrt(rp * rm))
    if n % 2 == 0:
        So = np.sum(1 / c[0::2])
        worst_ii = max(worst_ii, lam * So / 2)
        if n <= 20:
            mm = n // 2
            Bm = np.zeros((mm, mm))
            for i in range(mm):
                Bm[i, i] = c[2 * i]
                if i + 1 < mm:
                    Bm[i + 1, i] = c[2 * i + 1]
            Ei = np.abs(np.linalg.inv(Bm))
            F = np.zeros_like(Ei)
            for i in range(mm):
                for j in range(i + 1):
                    F[i, j] = (1 / c[2 * j]) * np.prod([c[2 * l + 1] / c[2 * l + 2] for l in range(j, i)])
            L = np.tril(np.ones_like(F, dtype=bool))
            worst_iii = max(worst_iii, np.max(np.abs(Ei - F)[L] / F[L]), np.max(Ei[~L], initial=0.0) / Ei.max())
    N += 1
log(f"{N} random log-concave chains (2<=n<60, dynamic range <= 1e6):")
log(f"  (i)  min of lam*S*sqrt(rho+ rho-) = {worst_i:.6f}   (lemma claims >= 1)")
log(f"  (ii) max of lam*S_odd/2 (even n) = {worst_ii:.6f}   (lemma claims <= 1)")
log(f"  (iii) max rel. error of explicit inverse formula on i>=j (and upper-triangle size relative to max) = {worst_iii:.2e}")
# log-concavity cannot be dropped: dimerised chain c = (1, M, 1, M, ..., 1)
for M in (10, 100, 1000):
    c = np.array([1.0 if k % 2 == 0 else M for k in range(9)])
    lam = lam_pos(c); S = np.sum(1 / c)
    log(f"  dimerised (non log-concave) chain n=10, M={M}: lam*S = {lam*S:.3e}  (adjacent ratio M; lemma bound 1/(M S) = {1/(M*S):.3e})")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lemma_check_out.txt"), "w").write("\n".join(out) + "\n")

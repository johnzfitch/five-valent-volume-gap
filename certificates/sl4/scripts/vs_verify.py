"""Exhaustive checks of the proof steps on all sorted tuples with 2 j_max <= N (default 80).

Exact (Fractions): factorisation c_k^2 = h_k^2 w_k^2 (hydrogen h(A1,M), w^2 = f/16, f = (k^2-beta^2)(A2^2-k^2));
Lemma F f(kappa) >= X^2-1 (X = 2 j4 + 1) with equality exactly on B1 = B2 = 0; C0 f2 > f1; C2 f_L > f_2; C3 f_L^2 >= f_{L-1} f_2.
Float: lambda (LAPACK) >= Lemma-S bound sqrt(a_min b_min) * lambda_h >= sqrt(mu); report min lambda^2/mu per d and the
equality cases of lambda^2 = mu.
"""
import sys, math, time
from fractions import Fraction as Fr
import numpy as np
from scipy.linalg import eigh_tridiagonal
from vs_core import sorted_tuples, kdata, chain2_exact

N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
t0 = time.time()
cnt = 0; bad = []; eqF = []; eq_lam = []; minr = {}; minS = 9.0
for J in sorted_tuples(N):
    kd = kdata(J)
    kmin2, kmax2, d = kd
    if d < 2:
        continue
    cnt += 1
    j1, j2, j3, j4 = [Fr(x, 2) for x in J]
    A1, B1, A2, B2 = j1 + j2 + 1, abs(j1 - j2), j3 + j4 + 1, abs(j3 - j4)
    M, beta = max(B1, B2), min(B1, B2)
    ks = [Fr(kmin2, 2) + s for s in range(1, d)]
    c2 = chain2_exact(J)
    f = [(k * k - beta * beta) * (A2 * A2 - k * k) for k in ks]
    h2 = [(A1 * A1 - k * k) * (k * k - M * M) / (4 * k * k - 1) for k in ks]
    # factorisation and range
    if Fr(kmin2, 2) != M or Fr(kmax2, 2) != A1 - 1:
        bad.append(("range", J))
    if any(c2[i] != h2[i] * f[i] / 16 for i in range(d - 1)):
        bad.append(("factor", J))
    X = 2 * j4 + 1
    if f[0] < X * X - 1:
        bad.append(("F", J))
    if f[0] == X * X - 1:
        eqF.append(J)
    L = d - 1
    if L >= 2:
        if not f[1] > f[0]: bad.append(("C0", J))
        if not f[-1] > f[1] and L >= 2:
            if L > 2 or not f[-1] >= f[1]: bad.append(("C2", J))
    if L >= 3:
        if not f[-1] * f[-1] >= f[-2] * f[1]: bad.append(("C3", J))
    # float part
    c = np.sqrt(np.array([float(x) for x in c2]))
    w = np.linalg.eigvalsh(np.diag(c, 1) + np.diag(c, -1)) if d <= 3 else eigh_tridiagonal(np.zeros(d), c, eigvals_only=True)
    w = np.abs(w); lam = w[w > 1e-9 * w.max()].min()
    g = np.sqrt(np.array([float(x) for x in f])) / 4
    la = 0.0; amin = 1.0; s = 1
    while 2 * s <= L:                       # a_i, links 2s-1 -> 2s
        la += 2 * (math.log(g[2 * s - 1]) - math.log(g[2 * s - 2])); amin = min(amin, math.exp(la)); s += 1
    lb = 2 * math.log(g[0]); bmin = math.exp(lb); s = 1
    while 2 * s + 1 <= L:                   # b_i, links 2s -> 2s+1
        lb += 2 * (math.log(g[2 * s]) - math.log(g[2 * s - 1])); bmin = min(bmin, math.exp(lb)); s += 1
    lam_h = 1.0 if d % 2 == 0 else 2.0
    bnd = math.sqrt(amin * bmin) * lam_h
    mu = float(j4) * (float(j4) + 1) / 4
    if lam < bnd * (1 - 1e-11): bad.append(("lemmaS>lam", J, lam, bnd))
    if bnd * bnd < mu * (1 - 1e-11): bad.append(("bound<mu", J))
    minS = min(minS, bnd * bnd / mu)
    r = lam * lam / mu
    if d not in minr or r < minr[d][0]: minr[d] = (r, J)
    if abs(r - 1) < 1e-11: eq_lam.append(J)
lines = [f"N = {N}: {cnt} tuples with d >= 2; violations: {len(bad)} {bad[:5]}",
         f"Lemma F equality cases: {len(eqF)}; all with B1 = B2 = 0: {all(J[0]==J[1] and J[2]==J[3] for J in eqF)}",
         f"min (Lemma-S bound)^2/mu = {minS:.12f}",
         f"lambda^2 = mu (to 1e-11) in {len(eq_lam)} tuples; all of the form (1/2,1/2,j,j): {all(J[0]==1 and J[1]==1 and J[2]==J[3] for J in eq_lam)}",
         "min lambda^2/mu by d (d<=12): " + "; ".join(f"d={dd}: {minr[dd][0]:.4f} at {minr[dd][1]}" for dd in sorted(minr) if dd <= 12),
         f"wall time {time.time()-t0:.0f} s"]
print("\n".join(lines))
open(f"../outputs/vs_verify_N{N}.txt", "w").write("\n".join(lines) + "\n")

"""Referee check 2: exhaustive exact test of the sharp law and of every inequality the proof uses.

For every sorted doubled-spin tuple J1<=J2<=J3<=J4<=N with d >= 2 (chain from Lemma P, which r1 checked against
exact 6j and brute force):
  * EXACT: lambda^2 >= mu (mu = j4(j4+1)/4) via the LDL pivots of M - mu I, M = B^T B (odd sites x even sites),
    in Fractions.  Strict iff all pivots > 0; equality iff the first m-1 pivots are > 0 and the last is 0.
  * EXACT: Lemma F (f_1 >= X^2-1, equality iff J1=J2 and J3=J4), C0, C2, C3, and directly the two facts used in
    section 3: every a_i = prod_{l<=i} f_{2l}/f_{2l-1} >= 1 (2i <= L) and every b_i/b_1 = prod_{l<i} f_{2l+1}/f_{2l} >= 1
    (2i-1 <= L).  Also log-concavity f_s^2 >= f_{s-1} f_{s+1}.
  * FLOAT: eigenvalues of the hydrogen chain h (Lemma H) equal N-1-2i; the Lemma-S bound sqrt(amin bmin) lam_H is
    <= lambda (LAPACK).
Usage: python r2_exhaustive.py N [workers]
"""
import sys, time
from fractions import Fraction as Fr
from multiprocessing import Pool
import numpy as np
from scipy.linalg import eigh_tridiagonal


def one(J):
    J1, J2, J3, J4 = J
    j1, j2, j3, j4 = [Fr(x, 2) for x in J]
    A1, B1, A2, B2 = j1 + j2 + 1, abs(j1 - j2), j3 + j4 + 1, abs(j3 - j4)
    M, beta, K = max(B1, B2), min(B1, B2), j1 + j2
    d = int(K - M) + 1
    L = d - 1
    ks = [M + s for s in range(1, d)]
    h2 = [(A1 * A1 - k * k) * (k * k - M * M) / (4 * k * k - 1) for k in ks]
    f = [(k * k - beta * beta) * (A2 * A2 - k * k) for k in ks]
    c2 = [h2[i] * f[i] / 16 for i in range(L)]
    mu = j4 * (j4 + 1) / 4
    X = 2 * j4 + 1
    flags = []
    # Lemma F
    if f[0] < X * X - 1: flags.append("F")
    eqF = (f[0] == X * X - 1)
    if eqF != (J1 == J2 and J3 == J4): flags.append("Feq")
    if L >= 2 and not f[1] > f[0]: flags.append("C0")
    if L >= 3 and not f[-1] > f[1]: flags.append("C2")
    if L >= 3 and not f[-1] ** 2 >= f[-2] * f[1]: flags.append("C3")
    for s in range(1, L - 1):
        if f[s] ** 2 < f[s - 1] * f[s + 1]: flags.append("logconc"); break
    # a_i, b_i exact
    p = Fr(1); i = 1
    while 2 * i <= L:
        p *= f[2 * i - 1] / f[2 * i - 2]
        if p < 1: flags.append("a<1"); break
        i += 1
    p = Fr(1); i = 2
    while 2 * i - 1 <= L:
        p *= f[2 * i - 2] / f[2 * i - 3]
        if p < 1: flags.append("b<b1"); break
        i += 1
    # exact LDL of B^T B - mu
    m = d // 2
    diag = []; off2 = []
    for i in range(1, m + 1):
        x = c2[2 * i - 2] + (c2[2 * i - 1] if 2 * i - 1 < L else 0)
        diag.append(x)
        if i < m: off2.append(c2[2 * i - 1] * c2[2 * i])
    piv = diag[0] - mu; status = None
    pivs = [piv]
    for i in range(1, m):
        if pivs[-1] <= 0: break
        pivs.append(diag[i] - mu - off2[i - 1] / pivs[-1])
    if len(pivs) == m and all(x > 0 for x in pivs): status = "strict"
    elif len(pivs) == m and all(x > 0 for x in pivs[:-1]) and pivs[-1] == 0: status = "equal"
    else: status = "VIOLATION"
    # float parts
    c = np.sqrt(np.array([float(x) for x in c2]))
    ev = np.abs(eigh_tridiagonal(np.zeros(d), c, eigvals_only=True))
    lam = ev[ev > 1e-9 * ev.max()].min()
    hv = np.sort(eigh_tridiagonal(np.zeros(d), np.sqrt(np.array([float(x) for x in h2])), eigvals_only=True))
    herr = float(np.max(np.abs(hv - np.arange(-(d - 1), d, 2))))
    g2 = [float(x) / 16 for x in f]
    amin = 1.0; pa = 1.0; i = 1
    while 2 * i <= L:
        pa *= g2[2 * i - 1] / g2[2 * i - 2]; amin = min(amin, pa); i += 1
    bmin = g2[0]; pb = g2[0]; i = 2
    while 2 * i - 1 <= L:
        pb *= g2[2 * i - 2] / g2[2 * i - 3]; bmin = min(bmin, pb); i += 1
    bnd = np.sqrt(amin * bmin) * (1.0 if d % 2 == 0 else 2.0)
    sviol = lam < bnd * (1 - 1e-10)
    return J, d, status, flags, herr, sviol, lam * lam / float(mu)


def tuples(N, J4):
    for J3 in range(1, J4 + 1):
        for J2 in range(1, J3 + 1):
            for J1 in range(1, J2 + 1):
                if (J1 + J2 + J3 + J4) % 2: continue
                if max(J2 - J1, J4 - J3) > J1 + J2 - 2: continue   # need d >= 2: kmax-kmin >= 1 (doubled: >= 2)
                yield (J1, J2, J3, J4)


def work(J4):
    res = dict(n=0, viol=[], eq=[], flags=[], herr=0.0, sviol=[], minr={})
    for J in tuples(None, J4):
        J, d, st, fl, he, sv, r = one(J)
        res["n"] += 1
        if st == "VIOLATION": res["viol"].append(J)
        if st == "equal": res["eq"].append(J)
        if fl: res["flags"].append((J, fl))
        res["herr"] = max(res["herr"], he)
        if sv: res["sviol"].append(J)
        if d not in res["minr"] or r < res["minr"][d][0]: res["minr"][d] = (r, J)
    return res


if __name__ == "__main__":
    N = int(sys.argv[1]); W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    with Pool(W) as P:
        rs = P.map(work, list(range(N, 0, -1)), chunksize=1)
    n = sum(r["n"] for r in rs)
    viol = [x for r in rs for x in r["viol"]]; eq = sorted(x for r in rs for x in r["eq"])
    flags = [x for r in rs for x in r["flags"]]; sv = [x for r in rs for x in r["sviol"]]
    herr = max(r["herr"] for r in rs)
    minr = {}
    for r in rs:
        for dd, v in r["minr"].items():
            if dd not in minr or v[0] < minr[dd][0]: minr[dd] = v
    expected_eq = sorted((1, 1, J, J) for J in range(1, N + 1))
    lines = [f"N = {N} (2 j_max <= N): {n} sorted tuples with d >= 2",
             f"EXACT sharp law: violations {len(viol)} {viol[:5]}",
             f"EXACT equality cases: {len(eq)}; equal to {{(1/2,1/2,j,j)}}: {eq == expected_eq}",
             f"EXACT lemma flags (F, Feq, C0, C2, C3, log-concavity, a_i>=1, b_i>=b_1): {len(flags)} {flags[:5]}",
             f"FLOAT hydrogen chain spectrum max |ev - integer| = {herr:.2e}",
             f"FLOAT Lemma-S bound above LAPACK lambda: {len(sv)} {sv[:5]}",
             "min lambda^2/mu by d (d<=10): " + "; ".join(f"d={dd}: {minr[dd][0]:.6f} at {minr[dd][1]}" for dd in sorted(minr) if dd <= 10),
             f"wall {time.time()-t0:.0f} s, {W} workers"]
    print("\n".join(lines))
    open(f"../outputs/r2_exhaustive_N{N}.txt", "w").write("\n".join(lines) + "\n")

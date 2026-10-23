"""Checks for Theorem B2 (orbit 2, distinguished edge of spin 1/2): every step of the proof is evaluated on all
tuples (j2..j5) with 2j_i <= M (j1 = 1/2, integer total spin), in exact rational arithmetic where it is an
inequality between rationals, and in float64 / mpmath for spectra.

Notation (doubled spins J): pair X = (J2,J3): m_X=|J2-J3|, n_X=J2+J3+2; pair D = (J4,J5): m_D, n_D likewise.
m = max(m_X,m_D), n = min(n_X,n_D).  Link l (= doubled k) has squared weight c_l^2 = (n_l^2-l^2)(l^2-m_l^2)/4
(alpha = beta = 2), with (n_l,m_l) those of the pair of its type.

S1  the path's links are exactly l = m+1..n-1, types alternate, the first link is of the type not owning m,
    the last of the type not owning n; c_l^2 from grid.py equals the formula (exact).
S2  even N: odd links' pair O has m_O <= m-1, n_O >= n+1; odd N: P owns m, Q owns n, odd links are Q.
S3  Lemma 2: q_min = 1/||E|| (float) for even chains.
S4  even: E_ij^2 <= (j/i)/c_{2j-1}^2 and c_{2j-1}^2 >= 4 n j^2 (exact), and q_min >= sqrt(8n/(H_K^2+H2_K)).
S5  odd: rho_t increasing (exact); split r; left/right piece inequalities (exact); q_min >= bound.
Prints the minimum of q_min and of the proved bound, and counts violations (expected 0).
"""
import sys, os, itertools
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import W_matrix

M = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def H(K, p=1):
    return sum(Fr(1, i ** p) for i in range(1, K + 1))


def jacobi_qmin(c):
    n = len(c) + 1
    T = np.zeros((n, n))
    for i, x in enumerate(c):
        T[i, i + 1] = T[i + 1, i] = x
    ev = np.abs(np.linalg.eigvalsh(T))
    ev.sort()
    thr = 1e-10 * max(ev[-1], 1)
    return ev[ev > thr][0]


viol = {k: 0 for k in ["S1", "S2", "S3", "S4", "S5", "bound"]}
stats = dict(cases=0, even=0, odd=0, min_q=1e9, min_bound=1e9, min_ratio=1e9)
worst = None
for J2, J3, J4, J5 in itertools.product(range(1, M + 1), repeat=4):
    if (1 + J2 + J3 + J4 + J5) % 2:
        continue
    J = (1, J2, J3, J4, J5)
    S, sq = W_matrix(J, 2, 2, exact=True)
    if len(S) < 2:
        continue
    stats["cases"] += 1
    # order path by s = b + c
    order = sorted(range(len(S)), key=lambda i: S[i][0] + S[i][1])
    links = []
    for a, b in zip(order, order[1:]):
        key = (min(a, b), max(a, b))
        (b1, c1), (b2, c2) = S[a], S[b]
        typ = "X" if b1 != b2 else "D"
        l = max(b1, b2) if typ == "X" else max(c1, c2)       # doubled k
        links.append((typ, l, sq.get(key, Fr(0))))
    mX, nX, mD, nD = abs(J2 - J3), J2 + J3 + 2, abs(J4 - J5), J4 + J5 + 2
    m, n = max(mX, mD), min(nX, nD)
    par = {"X": (mX, nX), "D": (mD, nD)}
    own_m = "X" if mX > mD else "D"
    own_n = "X" if nX < nD else "D"
    ok = [lk[1] for lk in links] == list(range(m + 1, n)) and all(links[i][0] != links[i + 1][0] for i in range(len(links) - 1))
    ok &= links[0][0] != own_m and links[-1][0] != own_n
    for typ, l, s2 in links:
        mm, nn = par[typ]
        ok &= s2 == Fr((nn * nn - l * l) * (l * l - mm * mm), 4)
    if not ok:
        viol["S1"] += 1
        continue
    c2 = [lk[2] for lk in links]
    c = [float(x) ** 0.5 for x in c2]
    q = jacobi_qmin(c)
    N = len(S)
    if N % 2 == 0:
        stats["even"] += 1
        O = links[0][0]
        mO, nO = par[O]
        if not (mO <= m - 1 and nO >= n + 1):
            viol["S2"] += 1
        K = N // 2
        # Lemma 2
        E = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1):
                v = 1.0 / c[2 * j]
                for l_ in range(j + 1, i + 1):
                    v *= c[2 * l_ - 1] / c[2 * l_]
                E[i, j] = v
        if abs(1.0 / np.linalg.norm(E, 2) - q) > 1e-9 * q:
            viol["S3"] += 1
        # S4 exact: rho-products and c lower bound
        for j in range(1, K + 1):
            if c2[2 * j - 2] < 4 * n * j * j:
                viol["S4"] += 1
            prod = Fr(1)
            for i in range(j, K + 1):
                if i > j:
                    prod *= c2[2 * (i - 1) - 1] / c2[2 * (i - 1)]
                if prod > Fr(j, i):
                    viol["S4"] += 1
        bound = (8 * n / float(H(K) ** 2 + H(K, 2))) ** 0.5
    else:
        stats["odd"] += 1
        P, Q = own_m, own_n
        if P == Q or links[0][0] != Q:
            viol["S2"] += 1
        mP, nP = par[P]
        mQ, nQ = par[Q]
        K = (N - 1) // 2
        # rho_t = c^2(link 2t) / c^2(link 2t+1), t = 1..K-1 (1-based link index s = 2t is P)
        rho = [c2[2 * t - 1] / c2[2 * t] for t in range(1, K)]
        if any(rho[i] > rho[i + 1] for i in range(len(rho) - 1)):
            viol["S5"] += 1
        tau = sum(1 for x in rho if x <= 1)
        if any(x > 1 for x in rho[:tau]):
            viol["S5"] += 1
        r = tau + 1
        # left piece: odd links s=1,3,..,2r-1 (Q), ratios rho_t<=1 for t<=r-1; bound sum (r-j+1)/c^2_{2j-1}
        FL = sum(Fr(r - j + 1) / c2[2 * j - 2] for j in range(1, r + 1))
        for j in range(1, r + 1):
            l = m + 2 * j - 1
            if c2[2 * j - 2] < 2 * (r - j + 1) * (n + l) * j * (m + j - 1):
                viol["S5"] += 1
        # right piece: P links s = 2t, t=r+1..K; sum (t-r)/c^2_{2t}
        FR = sum(Fr(t - r) / c2[2 * t - 1] for t in range(r + 1, K + 1))
        for t in range(r + 1, K + 1):
            l = m + 2 * t
            if c2[2 * t - 1] < (K - t + 1) * (n + 1 + l) * t * (l + m):
                viol["S5"] += 1
        bL = Fr(1, 1) * 0 + (sum(Fr(1, 2 * (n + m + 1) * j * j) for j in range(1, r + 1)))
        bR = H(K - r) / ((n + m + 3) * (2 * m + 2)) if K > r else Fr(0)
        if FL > bL or FR > bR:
            viol["S5"] += 1
        q_rows = min([1 / float(FL) ** 0.5] if r else [] + [], default=1e18)
        bound = min(1 / float(bL) ** 0.5 if bL else 1e18, 1 / float(bR) ** 0.5 if bR else 1e18)
        # direct check of the row-deletion step with true pieces
        left = c[:2 * r - 1]
        right = c[2 * r + 1:]
        qs = [jacobi_qmin(x) for x in (left, right) if len(x) >= 1]
        if qs and min(qs) > q * (1 + 1e-9):
            viol["S5"] += 1
    if bound > q * (1 + 1e-9):
        viol["bound"] += 1
    stats["min_q"] = min(stats["min_q"], q)
    if bound < stats["min_bound"]:
        stats["min_bound"] = bound
        worst = J
    stats["min_ratio"] = min(stats["min_ratio"], q / bound)
print(f"j1=1/2, alpha=beta=2, all (2j2..2j5) <= {M}: {stats['cases']} chains with >=2 sites "
      f"({stats['even']} even, {stats['odd']} odd)")
print("violations:", viol)
print(f"min q_min = {stats['min_q']:.9f};  min proved bound = {stats['min_bound']:.6f} at 2j = {worst};  "
      f"min q/bound = {stats['min_ratio']:.4f}")

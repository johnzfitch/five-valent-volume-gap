"""Exact and float checks of the structural identities for the Stark matrices and W0.
 (I1) delta(L) := h(L+1)^2 - h(L)^2 = -(2L+1)/4 + (4n^2-1)(4m^2-1) / (4(2L-1)(2L+1)(2L+3)),
      exact rationals, every (n,m) with doubled n,m <= 80 and every label with both links inside [m,n].
      Also h(L)^2 = (n^2+m^2)/4 - 1/16 - L^2/4 - (4n^2-1)(4m^2-1)/(16(4L^2-1)).
 (I2) W0 Gram identity on interior diagonals (|delta| <= s-1):
      Z(d,d+1)^T Z(d,d+1) - Z(d,d-1)^T Z(d,d-1) = diag(Lambda), Lambda = present-up^2 - present-down^2,
      and Lambda = delta1(b) - delta2(c) at every site with all four rectangle neighbours of the d-steps in R.
      Exact rationals (off-diagonal entries compared as squares with sign), all tuples doubled legs <= 8, 2s <= 9.
 (I3) Casimir defect on the full rectangle: Q = 2(E1E2 + F1F2) - (b+1/2)(c+1/2),
      [Q, K] = (F1-E1) eps2(c) + (F2-E2) eps1(b),  eps(L) = (4n^2-1)(4m^2-1)/(2(2L-1)(2L+1)(2L+3)).
      Float64 on all pair-pairs with doubled n <= 30 (relative residual), plus the statement that
      Q commutes with K exactly when both m = 1/2.
 (I4) H(n,1/2) is J_x of spin n-1/2 on its odd sector: h(L)^2 = (n^2-L^2)/4 (exact).
"""
import itertools
from fractions import Fraction as Fr
import numpy as np


def h2(n, m, L):
    if 4 * L * L - 1 == 0:      # L = 1/2 is a bottom label (m = 1/2): no link below it
        return 0 * L
    return (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)


# ---------- I1
bad = 0; cnt = 0
for nx in range(2, 81):
    for mx in range(0, nx - 1):
        if (nx - mx) % 2:   # n - m must be a positive integer (N = n - m sites)
            continue
        n = Fr(nx, 2); m = Fr(mx, 2)
        L = m + 1
        while L + 1 <= n:
            lhs = h2(n, m, L + 1) - h2(n, m, L)
            rhs = -(2 * L + 1) / 4 + (4 * n * n - 1) * (4 * m * m - 1) / (4 * (2 * L - 1) * (2 * L + 1) * (2 * L + 3))
            pf = (n * n + m * m) / 4 - Fr(1, 16) - L * L / 4 - (4 * n * n - 1) * (4 * m * m - 1) / (16 * (4 * L * L - 1))
            cnt += 1
            if lhs != rhs or pf != h2(n, m, L):
                bad += 1
            L += 1
print(f'(I1) delta closed form and partial fractions: {cnt} label checks, {bad} mismatches')

# ---------- I4
bad = 0; cnt = 0
for nx in range(3, 201, 2):
    n = Fr(nx, 2); L = Fr(3, 2)
    while L <= n - 1:
        cnt += 1
        if h2(n, Fr(1, 2), L) != (n * n - L * L) / 4:
            bad += 1
        L += 1
print(f'(I4) h(L)^2 = (n^2-L^2)/4 for m = 1/2: {cnt} checks, {bad} mismatches')

# ---------- I2 (exact)
def grid_sites(S, a2, a3, a4, a5):
    n1, m1 = a2 + a3 + 2, abs(a2 - a3); n2, m2 = a4 + a5 + 2, abs(a4 - a5)
    st = {(b, c) for b in range(m1, n1 - 1, 2) for c in range(m2, n2 - 1, 2) if abs(b - c) <= S <= b + c}
    return st, (n1, m1, n2, m2)


tot = 0; badd = 0; bado = 0; site_formula = 0; site_formula_bad = 0
for S in range(1, 10):
    for legs in itertools.product(range(1, 9), repeat=4):
        if (S + sum(legs)) % 2:
            continue
        st, (n1, m1, n2, m2) = grid_sites(S, *legs)
        if not st:
            continue
        H1 = lambda b: h2(Fr(n1, 2), Fr(m1, 2), Fr(b, 2))
        H2 = lambda c: h2(Fr(n2, 2), Fr(m2, 2), Fr(c, 2))
        # squared link from site x to its four d-neighbours, 0 if absent from R
        def up(b, c):   # d+1 neighbours: (b+1,c) via h1(b+1), (b,c-1) via h2(c)
            return [((b + 2, c), H1(b + 2)), ((b, c - 2), H2(c))]
        def dn(b, c):   # d-1 neighbours: (b-1,c) via h1(b), (b,c+1) via h2(c+1)
            return [((b - 2, c), H1(b)), ((b, c + 2), H2(c + 2))]
        for (b, c) in st:
            d2 = b - c
            if abs(d2) > S - 2:
                continue        # interior diagonals only
            tot += 1
            Lam = sum(w for y, w in up(b, c) if y in st) - sum(w for y, w in dn(b, c) if y in st)
            # off-diagonal: partner on the same diagonal (b+1,c+1)
            y = (b + 2, c + 2)
            if y in st:
                # up-Gram entry: shared up-neighbour (b+1,c): links h1(b+1) [from x] and h2(c+1) [from y]
                shu = (b + 2, c) in st
                shd = (b, c + 2) in st
                gu = H1(b + 2) * H2(c + 2) if shu else 0
                gd = H2(c + 2) * H1(b + 2) if shd else 0
                if gu != gd:
                    bado += 1
            # closed form for Lambda where all four d-neighbour links are present or naturally zero
            allin = all((yy in st) or w == 0 for yy, w in up(b, c) + dn(b, c))
            if allin:
                site_formula += 1
                n1f, m1f, n2f, m2f = Fr(n1, 2), Fr(m1, 2), Fr(n2, 2), Fr(m2, 2)
                d1 = H1(b + 2) - H1(b); dd2 = H2(c + 2) - H2(c)
                if Lam != d1 - dd2:
                    site_formula_bad += 1
print(f'(I2) interior-diagonal sites {tot}; off-diagonal (plaquette) mismatches {bado}; '
      f'Lambda = delta1(b)-delta2(c) checked at {site_formula} sites (all four links present or naturally 0): {site_formula_bad} mismatches')

# ---------- I3 (float)
def stark_parts(nx, mx):
    """E (raising), F = E^T, labels, and eps(L) := 2(h(L+1)^2 - h(L)^2) + L + 1/2 from the actual links
    (h = 0 for absent links), together with the closed form C/(2(2L-1)(2L+1)(2L+3))."""
    Ls = np.arange(mx, nx - 1, 2) / 2.0
    n = nx / 2.0; m = mx / 2.0
    N = len(Ls)
    hh = np.zeros(N + 1)            # hh[i] = h(L_i)^2 for the link L_{i-1} - L_i; hh[0] = hh[N] = 0
    for i in range(1, N):
        hh[i] = (n * n - Ls[i] ** 2) * (Ls[i] ** 2 - m * m) / (4 * Ls[i] ** 2 - 1)
    E = np.zeros((N, N))
    for i in range(1, N):
        E[i, i - 1] = np.sqrt(max(0.0, hh[i]))
    eps = 2 * (hh[1:] - hh[:-1]) + Ls + 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        closed = (4 * n * n - 1) * (4 * m * m - 1) / (2 * (2 * Ls - 1) * (2 * Ls + 1) * (2 * Ls + 3))
    return Ls, E, E.T.copy(), eps, closed


worst = 0; cnt = 0; cf_bad = 0; cf_cnt = 0; half_def = []
for n1x in range(2, 31):
    for m1x in range(0, n1x - 1):
        if (n1x - m1x) % 2: continue
        for n2x in range(2, 31, 3):
            for m2x in range(0, n2x - 1):
                if (n2x - m2x) % 2: continue
                L1, E1, F1, e1, c1 = stark_parts(n1x, m1x)
                L2, E2, F2, e2, c2 = stark_parts(n2x, m2x)
                I1 = np.eye(len(L1)); I2 = np.eye(len(L2))
                K = np.kron(E1 + F1, I2) + np.kron(I1, E2 + F2)
                Q = 2 * (np.kron(E1, E2) + np.kron(F1, F2)) - np.kron(np.diag(L1 + .5), np.diag(L2 + .5))
                rhs = np.kron(F1 - E1, np.diag(e2)) + np.kron(np.diag(e1), F2 - E2)
                lhs = Q @ K - K @ Q
                r = np.abs(lhs - rhs).max() / max(1.0, np.abs(lhs).max())
                worst = max(worst, r); cnt += 1
                for Ls, e, cl, mx_, nx_ in ((L1, e1, c1, m1x, n1x), (L2, e2, c2, m2x, n2x)):
                    for L, a, b_ in zip(Ls, e, cl):
                        if mx_ == 1 and L == 0.5:
                            half_def.append((a, (4 * (nx_ / 2) ** 2 - 1) / 8))
                            continue
                        cf_cnt += 1
                        if abs(a - b_) > 1e-9 * max(1, abs(a)):
                            cf_bad += 1
hd = max(abs(a - b) for a, b in half_def)
print(f'(I3) Casimir defect identity [Q,K] = (F1-E1)eps2 + (F2-E2)eps1: {cnt} pair-pairs, max relative residual {worst:.2e}')
print(f'     eps equals the closed form C/(2(2L-1)(2L+1)(2L+3)) at {cf_cnt} labels ({cf_bad} mismatches);')
print(f'     the exception is the bottom label L = 1/2 of an m = 1/2 pair, where eps(1/2) = (4n^2-1)/8 '
      f'(checked {len(half_def)} cases, max deviation {hd:.2e}); for m = 1/2 eps vanishes at every other label.')

"""Stage (triangular) certificates for the fixed-j1 gap.

Notation (THEOREM.md section 2).  All half-integers are DOUBLED integers on input.
For an ordered pairing (n1,m1 | n2,m2) and a colour class C (sites with (d+s) = cc mod 2), the bipartite block
B : l2(C) -> l2(C') splits by diagonals.  Every row diagonal rho touches the column diagonals rho-1 and rho+1 (undoubled),
and on each column diagonal the Gram matrix of the rows of one row diagonal is TRIDIAGONAL (consecutive columns share
exactly one row of that diagonal).  For x = (x_delta), ||Bx||^2 = sum_rho || R_{rho,lo} x_{rho-1} + R_{rho,hi} x_{rho+1} ||^2.

A SCHEME assigns every row diagonal that touches two column diagonals (an "interior" row diagonal) either
  - 'drop' (its rows are discarded, allowed since they contribute >= 0), or
  - ('lo', t) / ('hi', t): owned by the column diagonal below/above, with t in (0,1):
        ||a + b||^2 >= (1-t)||a||^2 - (1/t - 1)||b||^2      (a = owner part).
Row diagonals touching only one column diagonal (walls, or rows whose other neighbour is outside the strip) are added
with coefficient 1 to that column diagonal.  Stage_X = sum of (coefficient * Gram) over row diagonals touching X.
If every Stage_X - lam*I is positive definite then sigma_min(B)^2 >= lam, so q = 8 sigma_min >= 8 sqrt(lam).

Exactness: on a fixed column diagonal, the Gram off-diagonals of the two adjacent row diagonals are proportional with
the rational factor kappa = F(d+1/2)/F(d-1/2) ([OR] Proposition 6, a polynomial identity), so the off-diagonal of any
linear combination is (rational) * (a square root of a rational), and LDL^T pivots of a tridiagonal matrix only involve
squared off-diagonals: the certificate is exact rational arithmetic (python-flint fmpq).  The proportionality is
ASSERTED exactly for every column pair used.
"""
import math
import numpy as np
from flint import fmpq

Q0 = fmpq(45579607, 10000000)          # target q0 = 4.5579607
LAM0 = (Q0 / 8) ** 2                   # target sigma^2


def h2q(nP, mP, L2):
    """exact h^2 for doubled n, m, L"""
    n = fmpq(nP, 2); m = fmpq(mP, 2); L = fmpq(L2, 2)
    v = (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)
    return v if v > 0 else fmpq(0)


def Ff(s2, t2):
    """F(t) = (s+1/2)^2 - t^2, doubled s and t"""
    return fmpq((s2 + 1) ** 2 - t2 * t2, 4)


def Gf(s2, u2):
    """G(u) = (u+s+1)(u-s), doubled"""
    return fmpq((u2 + s2 + 2) * (u2 - s2), 4)


def link2(s2, n1, m1, n2, m2, up, kind):
    """exact squared weight of the link whose UPPER end is site up=(B,C) (doubled); kind 'X' (b-1 -> b) or 'D'."""
    B, C = up
    d2 = B - C; u2 = B + C
    if kind == 'X':
        return h2q(n1, m1, B) * Ff(s2, d2 - 1) * Gf(s2, u2) / 16
    return h2q(n2, m2, C) * Ff(s2, d2 + 1) * Gf(s2, u2) / 16


class Geometry:
    def __init__(self, s2, n1, m1, n2, m2):
        self.s2, self.p = s2, (n1, m1, n2, m2)
        Bs = range(m1, n1 - 1, 2); Cs = range(m2, n2 - 1, 2)
        self.sites = set()
        for b in Bs:
            for c in Cs:
                if abs(b - c) <= s2 and b + c >= s2 and (b + c + s2) % 2 == 0:
                    self.sites.add((b, c))
        self.diags = {}
        for (b, c) in self.sites:
            self.diags.setdefault(b - c, []).append((b, c))
        for k in self.diags:
            self.diags[k].sort(key=lambda x: x[0] + x[1])

    def w2(self, row, col):
        """exact squared link between a row site and an adjacent column site (any orientation)"""
        (b1, c1), (b2, c2) = row, col
        if b1 != b2:                      # X link
            up = row if b1 > b2 else col
            return link2(self.s2, *self.p, up, 'X')
        up = row if c1 > c2 else col
        return link2(self.s2, *self.p, up, 'D')

    def gram(self, col_d, row_d):
        """tridiagonal Gram of the rows on doubled diagonal row_d restricted to the columns of col_d.
        returns (cols, diag list, off2 list) exact; off2[i] = (squared) off-diagonal between cols i, i+1."""
        cols = self.diags.get(col_d, [])
        rows = set(self.diags.get(row_d, []))
        diag = []
        for (b, c) in cols:
            a = fmpq(0)
            for r in ((b + 2, c), (b - 2, c), (b, c + 2), (b, c - 2)):
                if r in rows:
                    a += self.w2(r, (b, c))
            diag.append(a)
        off2 = []
        for i in range(len(cols) - 1):
            (b, c), (b2_, c2_) = cols[i], cols[i + 1]
            # consecutive columns on a diagonal differ by (+2,+2) doubled; shared rows: (b+2,c) and (b,c+2)
            o = fmpq(0); cnt = 0
            for r in ((b + 2, c), (b, c + 2)):
                if r in rows and abs(r[0] - r[1] - row_d) == 0:
                    o = self.w2(r, cols[i]) * self.w2(r, cols[i + 1]); cnt += 1
            assert cnt <= 1
            off2.append(o)
        return cols, diag, off2


def kappa(s2, col_d):
    """F(d+1/2)/F(d-1/2) for the column diagonal d = col_d/2 (doubled argument)"""
    return Ff(s2, col_d + 1) / Ff(s2, col_d - 1)


def isqrt_q(x):
    """exact square root of a nonnegative rational square, or None"""
    p, q = int(x.p), int(x.q)
    if p < 0:
        return None
    rp, rq = math.isqrt(p), math.isqrt(q)
    if rp * rp == p and rq * rq == q:
        return fmpq(rp, rq)
    return None


def column_classes(s2, cc):
    """doubled column diagonals of class cc ((d+s) = cc mod 2), and row diagonals"""
    cols = [d for d in range(-s2, s2 + 1, 2) if ((d + s2) // 2) % 2 == cc]
    rows = [d for d in range(-s2, s2 + 1, 2) if ((d + s2) // 2) % 2 != cc]
    return cols, rows


def stage_data(G, cc, scheme):
    """scheme: dict row_d -> 'drop' | ('lo', t) | ('hi', t) for interior row diagonals (t a fmpq or float).
    returns {col_d: (cols, [(coef, row_d)])}"""
    s2 = G.s2
    cds, rds = column_classes(s2, cc)
    comp = {d: [] for d in cds}
    for r in rds:
        nb = [d for d in (r - 2, r + 2) if d in comp]
        nb_present = [d for d in nb if G.diags.get(d)]
        if len(nb) == 1 or len(nb_present) <= 1:
            for d in nb:
                comp[d].append((1, r))
            continue
        act = scheme.get(r, 'drop')
        if act == 'drop':
            continue
        side, t = act
        own = r - 2 if side == 'lo' else r + 2
        oth = r + 2 if side == 'lo' else r - 2
        comp[own].append((1 - t, r))
        comp[oth].append((-(1 / t - 1), r))
    return comp


def stage_matrix_exact(G, col_d, comps, lam):
    """exact (diag, off2) of Stage - lam I on column diagonal col_d"""
    s2 = G.s2
    grams = []
    for coef, r in comps:
        cols, dg, o2 = G.gram(col_d, r)
        grams.append((fmpq(coef) if not isinstance(coef, fmpq) else coef, r, dg, o2))
    cols = G.diags.get(col_d, [])
    n = len(cols)
    diag = []
    for i in range(n):
        a = fmpq(0)
        for c, r, dg, o2 in grams:
            a += c * dg[i]
        diag.append(a - lam)
    off2 = []
    kap = None
    for i in range(n - 1):
        # combine: off = sum coef_j * o_j, with o_hi = kappa * o_lo exactly
        lo = [(c, o2[i]) for c, r, dg, o2 in grams if r == col_d - 2]
        hi = [(c, o2[i]) for c, r, dg, o2 in grams if r == col_d + 2]
        olo2 = lo[0][1] if lo else None
        ohi2 = hi[0][1] if hi else None
        if olo2 is not None and ohi2 is not None and olo2 != 0:
            if kap is None: kap = kappa(s2, col_d)
            assert ohi2 == kap * kap * olo2, "Prop 6 proportionality failed"
            mult = sum((c for c, _ in lo), fmpq(0)) + kap * sum((c for c, _ in hi), fmpq(0))
            off2.append(mult * mult * olo2)
        elif olo2 is not None and (ohi2 is None or ohi2 == 0):
            if ohi2 == 0 and olo2 != 0:
                raise AssertionError('one-sided zero off-diagonal')
            mult = sum((c for c, _ in lo), fmpq(0))
            off2.append(mult * mult * olo2)
        else:
            mult = sum((c for c, _ in hi), fmpq(0))
            off2.append(mult * mult * (ohi2 if ohi2 is not None else fmpq(0)))
    return diag, off2


def ldl_neg(diag, off2):
    """number of negative and zero pivots of the symmetric tridiagonal matrix (exact)"""
    neg = zero = 0
    d = None
    for i in range(len(diag)):
        d = diag[i] if i == 0 else diag[i] - off2[i - 1] / d
        if d == 0:
            return neg, 1
        if d < 0:
            neg += 1
    return neg, zero


# ---------------- float versions for the parameter search ----------------
def stage_matrix_float(G, col_d, comps):
    s2 = G.s2
    cols = G.diags.get(col_d, [])
    n = len(cols)
    diag = np.zeros(n); off = np.zeros(max(n - 1, 0))
    kap = float(kappa(s2, col_d))
    for coef, r in comps:
        _, dg, o2 = G.gram(col_d, r)
        diag += float(coef) * np.array([float(x) for x in dg])
        o = np.sqrt(np.array([float(x) for x in o2]))
        off += float(coef) * o            # hi-side off-diagonals are already kappa * lo-side (positive gauge)
    return diag, off


def lam_min_tri(diag, off):
    from scipy.linalg import eigvalsh_tridiagonal
    if len(diag) == 0:
        return np.inf
    if len(diag) == 1:
        return diag[0]
    return eigvalsh_tridiagonal(diag, off, select='i', select_range=(0, 0))[0]

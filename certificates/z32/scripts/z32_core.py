"""Core tools for j1 = 3/2 (2s = 3): the two zigzag chains, the coupling block, and the
tridiagonal (column-Gram) matrices used in the proof of Conjecture Z_{3/2}.

Independent implementation; imports nothing from other research folders.

Spins are passed DOUBLED: tj = (3, A, B, C, D), pairing (AB|CD) in the order given
(pair 1 = (A,B) carries the b label, pair 2 = (C,D) the c label).

Exact closed forms (4-valent links, as in [DC] section 1 / [OR] section 1):
  X-link (b-1,c)-(b,c), upper end (b,c):  x^2 = h1(b)^2 F(d-1/2) G(u) / 16
  D-link (b,c-1)-(b,c), upper end (b,c):  y^2 = h2(c)^2 F(d+1/2) G(u) / 16
  h_P(L)^2 = (n_P^2-L^2)(L^2-m_P^2)/(4L^2-1),  F(t) = 4 - t^2,  G(u) = (u+5/2)(u-3/2).

Zigzag 0: rows R_b = (b, b+3/2) (d=-3/2), columns C_b = (b, b+1/2) (d=-1/2).
Zigzag 1: rows R'_b = (b, b-1/2) (d=+1/2), columns C'_b = (b, b-3/2) (d=+3/2).
B (rows x columns) is [[D0, 0], [N, D1]] in the zigzag order.
"""
from fractions import Fraction as Fr
import numpy as np

H = Fr(1, 2)


def pars(tj):
    s2, A, B, C, D = tj
    assert s2 == 3
    a, b, c, d = (Fr(x, 2) for x in (A, B, C, D))
    return a + b + 1, abs(a - b), c + d + 1, abs(c - d)


def in_region(P, b, c):
    n1, m1, n2, m2 = P
    return (m1 <= b <= n1 - 1) and (m2 <= c <= n2 - 1) and abs(b - c) <= Fr(3, 2) and b + c >= Fr(3, 2)


def Pf(n, m, L):
    return (n * n - L * L) * (L * L - m * m)


def xlink2(P, b, c):
    """squared X-link between (b-1,c) and (b,c) (upper end (b,c))."""
    n1, m1, n2, m2 = P
    d, u = b - c, b + c
    return Pf(n1, m1, b) / (4 * b * b - 1) * (4 - (d - H) ** 2) * (u + Fr(5, 2)) * (u - Fr(3, 2)) / 16


def dlink2(P, b, c):
    """squared D-link between (b,c-1) and (b,c) (upper end (b,c))."""
    n1, m1, n2, m2 = P
    d, u = b - c, b + c
    return Pf(n2, m2, c) / (4 * c * c - 1) * (4 - (d + H) ** 2) * (u + Fr(5, 2)) * (u - Fr(3, 2)) / 16


class Z32:
    """Exact zigzag data for one ordered pairing."""

    def __init__(self, tj):
        self.tj = tj
        P = self.P = pars(tj)
        n1, m1, n2, m2 = P
        bs = [m1 + k for k in range(int(n1 - m1))]
        self.R0 = [b for b in bs if in_region(P, b, b + Fr(3, 2))]
        self.C0 = [b for b in bs if in_region(P, b, b + H)]
        self.R1 = [b for b in bs if in_region(P, b, b - H)]
        self.C1 = [b for b in bs if in_region(P, b, b - Fr(3, 2))]
        # squared links, keyed by (row b, col b)
        self.D0 = {}
        for b in self.R0:
            if b in self.C0:
                self.D0[(b, b)] = dlink2(P, b, b + Fr(3, 2))          # q_b : C_b - R_b
            if b + 1 in self.C0:
                self.D0[(b, b + 1)] = xlink2(P, b + 1, b + Fr(3, 2))  # p_b : R_b - C_{b+1}
        self.D1 = {}
        for b in self.R1:
            if b in self.C1:
                self.D1[(b, b)] = dlink2(P, b, b - H)                 # q'_b : C'_b - R'_b
            if b + 1 in self.C1:
                self.D1[(b, b + 1)] = xlink2(P, b + 1, b - H)         # p'_b : R'_b - C'_{b+1}
        self.N = {}
        for b in self.R1:
            if b in self.C0:
                self.N[(b, b)] = dlink2(P, b, b + H)                  # n+_b : R'_b - C_b
            if b - 1 in self.C0:
                self.N[(b, b - 1)] = xlink2(P, b, b - H)              # n-_b : R'_b - C_{b-1}
        self.imb = (len(self.R0) - len(self.C0), len(self.R1) - len(self.C1))

    # ---------- float matrices ----------
    def mats(self, R0=None, C0=None, R1=None, C1=None):
        R0 = self.R0 if R0 is None else R0
        C0 = self.C0 if C0 is None else C0
        R1 = self.R1 if R1 is None else R1
        C1 = self.C1 if C1 is None else C1

        def M(rows, cols, L):
            A = np.zeros((len(rows), len(cols)))
            ci = {c: j for j, c in enumerate(cols)}
            for i, r in enumerate(rows):
                for (rr, cc), w in L.items():
                    if rr == r and cc in ci:
                        A[i, ci[cc]] = float(w) ** 0.5
            return A
        return M(R0, C0, self.D0), M(R1, C0, self.N), M(R1, C1, self.D1)

    def deletions(self):
        """All square reductions: list of (R0,C0,R1,C1) kept lists (one deletion per imbalanced zigzag)."""
        out0, out1 = [], []
        for (R, C, imb, out) in ((self.R0, self.C0, self.imb[0], out0), (self.R1, self.C1, self.imb[1], out1)):
            if imb == 0:
                out.append((list(R), list(C)))
            elif imb == 1:
                for j in range(len(R)):
                    out.append((R[:j] + R[j + 1:], list(C)))
            elif imb == -1:
                for j in range(len(C)):
                    out.append((list(R), C[:j] + C[j + 1:]))
            else:
                raise ValueError('imbalance', imb)
        return [(a[0], a[1], b[0], b[1]) for a in out0 for b in out1]


def pairings6(tj):
    s, a, b, c, d = tj
    return [(s, a, b, c, d), (s, c, d, a, b), (s, a, c, b, d), (s, b, d, a, c), (s, a, d, b, c), (s, b, c, a, d)]


def monotone(imb):
    return not (imb[0] == -1 and imb[1] == 1)


def tuples32(M):
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    if (3 + a + b + c + d) % 2 == 0:
                        yield (3, a, b, c, d)

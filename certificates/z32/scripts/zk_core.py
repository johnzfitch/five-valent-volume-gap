"""Exact zigzag data for general half-integer j1 = s (K = s + 1/2 zigzags).  Doubled input (2s, A, B, C, D), pairing
(AB | CD) in the given order.

Zigzag k: rows R^k_b = (b, b - d_k) on d_k = -s + 2k, columns C^k_b = (b, b - d_k - 1) on d_k + 1.
Links (squared), with the closed forms x^2 = h1(b)^2 F(d-1/2) G(u)/16 (X, upper end (b,c)),
y^2 = h2(c)^2 F(d+1/2) G(u)/16 (D, upper end (b,c)),  F(t) = (s+1/2)^2 - t^2,  G(u) = (u+s+1)(u-s):
  q^k_b : C^k_b - R^k_b        (D-link, upper end R^k_b)
  p^k_b : R^k_b - C^k_{b+1}    (X-link, upper end C^k_{b+1})
  n+^k_b: R^k_b - C^{k-1}_b    (D-link, upper end C^{k-1}_b)
  n-^k_b: R^k_b - C^{k-1}_{b-1}(X-link, upper end R^k_b)
Coupling identity ([OR] Prop. 6): N_k^T N_k = kappa_k D_{k-1}^T D_{k-1} + diag(Lambda_k) on full blocks,
  kappa_k = F(d_k - 1/2)/F(d_k - 3/2),
  Lambda_k(c) = n+^k_c^2 + n-^k_{c+1}^2 - kappa_k (q^{k-1}_c^2 + p^{k-1}_{c-1}^2).
"""
from fractions import Fraction as Fr
import numpy as np

H = Fr(1, 2)


class ZK:
    def __init__(self, tj):
        s2, A, B, C, D = tj
        self.tj = tj
        self.s = s = Fr(s2, 2)
        self.K = K = s2 // 2 + 1
        a, b_, c_, d_ = (Fr(x, 2) for x in (A, B, C, D))
        self.n1, self.m1, self.n2, self.m2 = a + b_ + 1, abs(a - b_), c_ + d_ + 1, abs(c_ - d_)
        n1, m1, n2, m2 = self.n1, self.m1, self.n2, self.m2
        self.F = lambda t: (s + H) ** 2 - t * t
        self.G = lambda u: (u + s + 1) * (u - s)
        bs = [m1 + k for k in range(int(n1 - m1))]
        self.d = [-s + 2 * k for k in range(K)]
        inreg = lambda b, c: (m2 <= c <= n2 - 1) and abs(b - c) <= s and b + c >= s
        self.R = [[b for b in bs if inreg(b, b - self.d[k])] for k in range(K)]
        self.C = [[b for b in bs if inreg(b, b - self.d[k] - 1)] for k in range(K)]
        P1 = lambda L: (n1 * n1 - L * L) * (L * L - m1 * m1)
        P2 = lambda L: (n2 * n2 - L * L) * (L * L - m2 * m2)
        F, G = self.F, self.G

        def xl(b, c):   # X-link squared, upper end (b,c)
            return P1(b) / (4 * b * b - 1) * F(b - c - H) * G(b + c) / 16

        def dl(b, c):   # D-link squared, upper end (b,c)
            return P2(c) / (4 * c * c - 1) * F(b - c + H) * G(b + c) / 16
        self.q = []; self.p = []; self.np_ = [None]; self.nm = [None]
        for k in range(K):
            dk = self.d[k]
            Ck = set(self.C[k])
            self.q.append({b: dl(b, b - dk) for b in self.R[k] if b in Ck})
            self.p.append({b: xl(b + 1, b - dk) for b in self.R[k] if (b + 1) in Ck})
        for k in range(1, K):
            dk = self.d[k]
            Cm = set(self.C[k - 1])
            self.np_.append({b: dl(b, b - dk + 1) for b in self.R[k] if b in Cm})
            self.nm.append({b: xl(b, b - dk) for b in self.R[k] if (b - 1) in Cm})
        self.imb = [len(self.R[k]) - len(self.C[k]) for k in range(K)]
        self.kappa = [None] + [F(self.d[k] - H) / F(self.d[k] - 3 * H) for k in range(1, K)]

    def ends(self, k):
        R, C = self.R[k], self.C[k]
        if not R and not C:
            return '--'
        lo = 'C' if (C and (not R or C[0] <= R[0])) else 'R'
        hi = 'R' if (R and (not C or R[-1] >= C[-1])) else 'C'
        return lo + hi

    def Lam(self, k, c):
        """Lambda_k at column C^{k-1}_c (exact)."""
        v = self.np_[k].get(c, 0) + self.nm[k].get(c + 1, 0)
        v -= self.kappa[k] * (self.q[k - 1].get(c, 0) + self.p[k - 1].get(c - 1, 0))
        return v

    def mats(self):
        """float blocks D_k (rows x cols of zigzag k) and N_k (rows k x cols k-1)."""
        K = self.K
        D, N = [], [None]
        for k in range(K):
            ri = {b: i for i, b in enumerate(self.R[k])}; ci = {b: j for j, b in enumerate(self.C[k])}
            M = np.zeros((len(ri), len(ci)))
            for b, w in self.q[k].items():
                M[ri[b], ci[b]] = float(w) ** 0.5
            for b, w in self.p[k].items():
                M[ri[b], ci[b + 1]] = float(w) ** 0.5
            D.append(M)
        for k in range(1, K):
            ri = {b: i for i, b in enumerate(self.R[k])}; ci = {b: j for j, b in enumerate(self.C[k - 1])}
            M = np.zeros((len(ri), len(ci)))
            for b, w in self.np_[k].items():
                M[ri[b], ci[b]] = float(w) ** 0.5
            for b, w in self.nm[k].items():
                M[ri[b], ci[b - 1]] = float(w) ** 0.5
            N.append(M)
        return D, N

    def full(self):
        D, N = self.mats()
        K = self.K
        rows = sum(d.shape[0] for d in D); cols = sum(d.shape[1] for d in D)
        B = np.zeros((rows, cols))
        r0 = 0
        cstart = np.cumsum([0] + [d.shape[1] for d in D])
        for k in range(K):
            B[r0:r0 + D[k].shape[0], cstart[k]:cstart[k + 1]] = D[k]
            if k > 0:
                B[r0:r0 + N[k].shape[0], cstart[k - 1]:cstart[k]] = N[k]
            r0 += D[k].shape[0]
        return B


def pairings6(tj):
    s, a, b, c, d = tj
    return [(s, a, b, c, d), (s, c, d, a, b), (s, a, c, b, d), (s, b, d, a, c), (s, a, d, b, c), (s, b, c, a, d)]


def tuplesK(s2, M):
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    if (s2 + a + b + c + d) % 2 == 0:
                        yield (s2, a, b, c, d)

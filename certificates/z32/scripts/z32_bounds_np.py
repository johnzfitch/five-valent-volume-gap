"""Vectorised float64 version of z32_bounds.regime_bounds (exploration and scanning only).
The certified evaluation uses z32_bounds (exact) or z32_certify_scan (outward-rounded arb balls)."""
import numpy as np

H, K = 0.5, 1.5


def P(n, m, L):
    return (n * n - L * L) * (L * L - m * m)


def Lam(c, n1, m1, n2, m2):
    L1 = P(n1, m1, c + 1) * c * (c + 2) / ((2 * c + 1) * (2 * c + 3)) - P(n1, m1, c) * (2 * c + 3) / (4 * (2 * c + 1))
    L2 = P(n2, m2, c + H) * (2 * c + 3) * (2 * c - 1) / (16 * c * (c + 1)) - P(n2, m2, c + K) * c / (4 * (c + 1))
    return L1 + L2


q0 = lambda n2, m2, b: 3 / 16 * P(n2, m2, b + K) * b / (b + 1)
p0 = lambda n1, m1, b: 3 / 16 * P(n1, m1, b + 1) * (2 * b + 5) / (2 * b + 3)
q1 = lambda n2, m2, b: 3 / 16 * P(n2, m2, b - H) * (b + 1) / b
p1 = lambda n1, m1, b: 3 / 16 * P(n1, m1, b + 1) * (2 * b - 1) / (2 * b + 1)


def ar(lo, hi):
    n = int(round(hi - lo)) + 1
    return lo + np.arange(max(n, 0), dtype=float)


def F_CR(lo, hi, qf):
    c = ar(lo, hi)
    R = (hi - c + 1) / qf(c)
    return R.sum(), c, R


def F_RC(lo, hi, pf):
    i = ar(lo, hi)
    pv = pf(i)
    F = ((hi - i + 1) / pv).sum()
    R = np.cumsum(1 / pv)        # R at column i+1
    return F, i + 1, R


def F_RR(lo, hi, pf, qf):
    i = ar(lo, hi)
    F = 0.0
    a = i[i < hi]
    if a.size:
        F += ((hi - a) / pf(a)).sum()
    b = i[i > lo]
    if b.size:
        F += ((b - lo) / qf(b)).sum()
    return F


def regime_bounds(name, n1, m=None):
    if name == 'R1':
        m1 = m; mu = 0.0 if abs((m1 + 0.5) - round(m1 + 0.5)) < 1e-9 else 0.5
        F0, c, R0 = F_CR(m1, n1 - 1, lambda c: q0(n1 + K, m1 - K, c))
        F1, _, _ = F_CR(m1, n1 - 1, lambda c: q1(n1 + K, m1 - K, c))
        W = (np.maximum(Lam(c, n1, m1, n1 + K, mu), 0) * R0).sum()
        return F0, F1, W
    if name == 'R1c':
        F0, c, R0 = F_CR(H, n1 - 1, lambda c: q0(n1 + K, 0.0, c))
        F1, _, _ = F_CR(K, n1 - 1, lambda c: q1(n1 + K, 0.0, c))
        W = (np.maximum(Lam(c, n1, H, n1 + K, 0.0), 0) * R0).sum()
        return F0, F1, W
    if name in ('R2+', 'R2-'):
        m1 = m; m2 = m1 + H if name == 'R2+' else m1 - H
        F0, c, R0 = F_CR(m1, n1 - 1, lambda c: q0(n1 + K, m2, c))
        lo = m1 + 1 if name == 'R2+' else m1
        F1 = F_RR(lo, n1 - 1, lambda i: p1(n1, m1, i), lambda i: q1(n1 + K, m2, i))
        W = (np.maximum(Lam(c, n1, m1, n1 + K, m2), 0) * R0).sum()
        return F0, F1, W
    if name == 'R4':
        m2 = m; m1w = m2 - K
        F0 = F_RR(m2 - K, n1 - 1, lambda i: p0(n1, m1w, i), lambda i: q0(n1 + K, m2, i))
        F1 = F_RR(m2 + H, n1 - 1, lambda i: p1(n1, m1w, i), lambda i: q1(n1 + K, m2, i)) if m2 + H <= n1 - 1 else 0.0
        return F0, F1, 0.0
    if name == 'R4c':
        F0 = F_RR(0.0, n1 - 1, lambda i: p0(n1, 0.0, i), lambda i: q0(n1 + K, H, i))
        F1 = F_RR(1.0, n1 - 1, lambda i: p1(n1, 0.0, i), lambda i: q1(n1 + K, H, i))
        return F0, F1, 0.0
    if name in ('Tp', 'Tm'):
        n2 = n1 + H if name == 'Tp' else n1 - H
        hi0 = n1 - 2 if name == 'Tp' else n1 - 3
        F0, c, R0 = F_RC(0.0, hi0, lambda i: p0(n1, 0.0, i))
        F1 = F_RR(1.0, n1 - 1, lambda i: p1(n1, 0.0, i), lambda i: q1(n2, H, i))
        W = (np.maximum(Lam(c, n1, 0.0, n2, H), 0) * R0).sum()
        return F0, F1, W
    raise ValueError(name)


def sigma2(F0, F1, W, kap=4 / 3):
    k = kap + W
    if F1 == 0:
        return 1 / F0
    lo, hi = 0.0, 1 / k
    for _ in range(100):
        t = (lo + hi) / 2
        if (1 - t * k) / F0 > t / (1 + t) / F1:
            lo = t
        else:
            hi = t
    t = lo
    return min((1 - t * k) / F0, t / (1 + t) / F1)

"""Closed-form worst-case bounds per regime (the analytic part of the proof).

For each regime the bounds depend only on a few parameters (pair 1 and the regime constants):
  F0 >= 1/lam_min(D0^T D0),  F1 >= 1/lam_min(D1^T D1),  W >= sup_x x^T Lambda x / |D0 x|^2,
computed from 'count' forms of the chain lemmas with worst-case link lower bounds:
  CR (matched q, same-column ratios s<=1):   F <= sum_c tau_c / q_c^2,      R_c <= tau_c/q_c^2
  RC (matched p, same-row ratios r>=1):      F <= sum_i cnt_i / p_i^2,      R_c <= sum_{i<c} 1/p_i^2
  RR (tall, single crossing):                F <= sum_{i<hi} (hi-i)/p_i^2 + sum_{i>lo} (i-lo)/q_i^2
  W <= sum_c Lambdabar_c^+ R_c.
Then  sigma_min(B)^2 >= max_tau min( (1 - tau (4/3 + W))/F0 , tau/(1+tau)/F1 ).

Exact rational arithmetic is used when exact=True (Fractions), floats otherwise.
"""
from fractions import Fraction as Fr

H = Fr(1, 2)
K32 = Fr(3, 2)


def P(n, m, L):
    return (n * n - L * L) * (L * L - m * m)


def Lam(c, n1, m1, n2, m2):
    """Diagonal of N^T N - (4/3) D0^T D0 at zz0 column C_c (exact formula, valid at every column)."""
    L1 = P(n1, m1, c + 1) * c * (c + 2) / ((2 * c + 1) * (2 * c + 3)) - P(n1, m1, c) * (2 * c + 3) / (4 * (2 * c + 1))
    L2 = P(n2, m2, c + H) * (2 * c + 3) * (2 * c - 1) / (16 * c * (c + 1)) - P(n2, m2, c + K32) * c / (4 * (c + 1))
    return L1 + L2


# exact link formulas (squared)
def q0(n2, m2, b):  return Fr(3, 16) * P(n2, m2, b + K32) * b / (b + 1)
def p0(n1, m1, b):  return Fr(3, 16) * P(n1, m1, b + 1) * (2 * b + 5) / (2 * b + 3)
def q1(n2, m2, b):  return Fr(3, 16) * P(n2, m2, b - H) * (b + 1) / b
def p1(n1, m1, b):  return Fr(3, 16) * P(n1, m1, b + 1) * (2 * b - 1) / (2 * b + 1)


def rng(lo, hi):
    out = []
    b = lo
    while b <= hi:
        out.append(b); b += 1
    return out


def F_CR(lo, hi, qf):
    """count bound for a CR chain on [lo,hi] (matched q, s<=1): sum (hi-c+1)/q_c^2; returns (F, R dict)."""
    F = 0; R = {}
    for c in rng(lo, hi):
        v = (hi - c + 1) / qf(c)
        F += v; R[c] = v
    return F, R


def F_RC(lo, hi, pf):
    """RC chain: rows lo..hi, cols lo+1..hi+1, matched p (r>=1): F <= sum_i (hi-i+1)/p_i^2;
    row norms R_c (col c) <= sum_{i<c} 1/p_i^2."""
    F = 0; R = {}; acc = 0
    for i in rng(lo, hi):
        F += (hi - i + 1) / pf(i)
        acc += 1 / pf(i)
        R[i + 1] = acc
    return F, R


def F_RR(lo, hi, pf, qf):
    """tall chain rows lo..hi, cols lo+1..hi (single crossing)."""
    F = 0
    for i in rng(lo, hi):
        if i < hi:
            F += (hi - i) / pf(i)
        if i > lo:
            F += (i - lo) / qf(i)
    return F


def regime_bounds(name, prm):
    """prm: dict of exact parameters.  Returns (F0, F1, W)."""
    if name == 'R1':
        n1, m1 = prm['n1'], prm['m1']
        mu = Fr(0) if (m1 + H).denominator == 1 else H
        n2w, m2w = n1 + K32, m1 - K32          # worst for links
        F0, R0 = F_CR(m1, n1 - 1, lambda c: q0(n2w, m2w, c))
        F1, _ = F_CR(m1, n1 - 1, lambda c: q1(n2w, m2w, c))
        W = sum(max(Lam(c, n1, m1, n1 + K32, mu), 0) * R0[c] for c in R0)
        return F0, F1, W
    if name == 'R1c':
        n1 = prm['n1']; m1 = H
        F0, R0 = F_CR(H, n1 - 1, lambda c: q0(n1 + K32, 0, c))
        F1, _ = F_CR(K32, n1 - 1, lambda c: q1(n1 + K32, 0, c))
        W = sum(max(Lam(c, n1, m1, n1 + K32, 0), 0) * R0[c] for c in R0)
        return F0, F1, W
    if name in ('R2+', 'R2-'):
        n1, m1 = prm['n1'], prm['m1']
        m2 = m1 + H if name == 'R2+' else m1 - H
        n2w = n1 + K32
        F0, R0 = F_CR(m1, n1 - 1, lambda c: q0(n2w, m2, c))
        lo = m1 + 1 if name == 'R2+' else m1
        F1 = F_RR(lo, n1 - 1, lambda i: p1(n1, m1, i), lambda i: q1(n2w, m2, i))
        W = sum(max(Lam(c, n1, m1, n2w, m2), 0) * R0[c] for c in R0)
        return F0, F1, W
    if name == 'R4':
        n1, m2 = prm['n1'], prm['m2']
        m1w = m2 - K32; n2w = n1 + K32
        F0 = F_RR(m2 - K32, n1 - 1, lambda i: p0(n1, m1w, i), lambda i: q0(n2w, m2, i))
        if m2 + H <= n1 - 1:
            F1 = F_RR(m2 + H, n1 - 1, lambda i: p1(n1, m1w, i), lambda i: q1(n2w, m2, i))
        else:
            F1 = 0
        return F0, F1, 0
    if name == 'R4c':
        n1 = prm['n1']; n2w = n1 + K32
        F0 = F_RR(0, n1 - 1, lambda i: p0(n1, 0, i), lambda i: q0(n2w, H, i))
        F1 = F_RR(1, n1 - 1, lambda i: p1(n1, 0, i), lambda i: q1(n2w, H, i))
        return F0, F1, 0
    if name in ('Tp', 'Tm'):
        n1 = prm['n1']; n2 = n1 + H if name == 'Tp' else n1 - H
        hi0 = n1 - 2 if name == 'Tp' else n1 - 3
        F0, R0 = F_RC(0, hi0, lambda i: p0(n1, 0, i))
        F1 = F_RR(1, n1 - 1, lambda i: p1(n1, 0, i), lambda i: q1(n2, H, i))
        W = sum(max(Lam(c, n1, 0, n2, H), 0) * R0[c] for c in R0)
        return F0, F1, W
    raise ValueError(name)


def sigma2_bound(F0, F1, W, kap=Fr(4, 3), grid=400):
    """max over tau in (0, 1/(kap+W)) of min((1 - tau(kap+W))/F0, tau/(1+tau)/F1) (floats)."""
    F0, F1, W = float(F0), float(F1), float(W)
    k = float(kap) + W
    if F0 == 0 and F1 == 0:
        return float('inf')
    if F1 == 0:
        return 1 / F0
    if F0 == 0:
        return float('inf')
    # exact optimum: the min is maximised where the two are equal (first decreasing, second increasing)
    lo, hi = 0.0, 1 / k
    for _ in range(200):
        t = (lo + hi) / 2
        a = (1 - t * k) / F0; b = t / (1 + t) / F1
        if a > b:
            lo = t
        else:
            hi = t
    t = lo
    return min((1 - t * k) / F0, t / (1 + t) / F1)


def params_of(name, P5):
    """Regime parameters from the ordered doubled tuple."""
    s2, A, B, C, D = P5
    a, b, c, d = (Fr(x, 2) for x in (A, B, C, D))
    n1, m1, n2, m2 = a + b + 1, abs(a - b), c + d + 1, abs(c - d)
    if name in ('R1', 'R2+', 'R2-'):
        return {'n1': n1, 'm1': m1}
    if name == 'R4':
        return {'n1': n1, 'm2': m2}
    return {'n1': n1}

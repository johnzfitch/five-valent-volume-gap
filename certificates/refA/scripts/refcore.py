"""Referee A core (written from the formulas in the refereed texts; imports nothing from the authors' folders).

Spins are DOUBLED integers throughout: tj = (S, A, B, C, D) means j1 = S/2, j2 = A/2, ...
Pairing (23|45): b = |J2+J3|, c = |J4+J5| (doubled labels bb, cc).  Other pairings by permuting legs 2-5
(|spec T| is S4-invariant, Prop. 1(d) of [CL], proved in the referee report).

Closed form ([CL] section 1):
  X-link (b-1,c)-(b,c):  x^2 = Q1(b) P(b;s,c) / (16 (4b^2-1)),  Q1(b) = (n1^2-b^2)(b^2-m1^2),
                          P(b;s,c) = ((s+c+1)^2-b^2)(b^2-(s-c)^2),  n1 = j2+j3+1, m1 = |j2-j3|.
  D-link (b,c-1)-(b,c):  same with the pairs exchanged.
Stark link h(L)^2 = (n^2-L^2)(L^2-m^2)/(4L^2-1).
Everything returned as exact Fractions (real, undoubled values).
"""
from fractions import Fraction as Fr
import math

H = Fr(1, 2)


def pair_nm(a, b):
    """doubled spins a, b -> (n, m) undoubled Fractions, n = ja+jb+1, m = |ja-jb|"""
    return Fr(a + b + 2, 2), Fr(abs(a - b), 2)


def stark_h2(n, m, L):
    """squared Stark link at label L (undoubled Fractions)"""
    return (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)


def region(tj):
    """sites (b, c) as undoubled Fractions in R (pairing 23|45), plus the rectangle."""
    S, A, B, C, D = tj
    s = Fr(S, 2)
    n1, m1 = pair_nm(A, B)
    n2, m2 = pair_nm(C, D)
    rect = []
    b = m1
    while b <= n1 - 1:
        c = m2
        while c <= n2 - 1:
            rect.append((b, c))
            c += 1
        b += 1
    R = [(b, c) for (b, c) in rect if abs(b - c) <= s <= b + c and (b + c + s).denominator == 1]
    rect = [(b, c) for (b, c) in rect if (b + c + s).denominator == 1]   # same lattice as R
    return s, (n1, m1), (n2, m2), rect, R


def links_closed_form(tj):
    """dict {((b-1,c),(b,c)) or ((b,c-1),(b,c)): squared modulus} from the closed form, for sites in R."""
    s, (n1, m1), (n2, m2), rect, R = region(tj)
    Rs = set(R)
    out = {}
    for (b, c) in R:
        if (b - 1, c) in Rs:
            x2 = (n1 * n1 - b * b) * (b * b - m1 * m1) * ((s + c + 1) ** 2 - b * b) * (b * b - (s - c) ** 2) / (16 * (4 * b * b - 1))
            out[((b - 1, c), (b, c))] = x2
        if (b, c - 1) in Rs:
            d2 = (n2 * n2 - c * c) * (c * c - m2 * m2) * ((s + b + 1) ** 2 - c * c) * (c * c - (s - b) ** 2) / (16 * (4 * c * c - 1))
            out[((b, c - 1), (b, c))] = d2
    return out


def colour(s, b, c):
    """parity of u - s (u = b + c), an integer"""
    return int(b + c - s) % 2


def heron_sigma2(tj):
    """sigma^2(b,c) = phi(d) omega(u), phi(d)phi(d+1) = F(d+1/2), omega(u-1)omega(u) = G(u); phi = omega = 1 at the
    smallest d / u in R."""
    s, _, _, rect, R = region(tj)
    F = lambda t: (s + H) ** 2 - t * t
    G = lambda u: (u + s + 1) * (u - s)
    ds = sorted({b - c for b, c in R}); us = sorted({b + c for b, c in R})
    phi = {ds[0]: Fr(1)}
    for d in ds[1:]:
        assert d - 1 in phi
        phi[d] = F(d - H) / phi[d - 1]
    om = {us[0]: Fr(1)}
    for u in us[1:]:
        assert u - 1 in om
        om[u] = G(u) / om[u - 1]
    return {(b, c): phi[b - c] * om[b + c] for (b, c) in R}, phi, om


def wallis(K):
    """W_K = prod_{l=1}^{K} 4l^2/(4l^2-1) (K integer >= 0)"""
    w = Fr(1)
    for l in range(1, K + 1):
        w *= Fr(4 * l * l, 4 * l * l - 1)
    return w


def Phi_s(s):
    if s.denominator == 1:
        return 2 * s
    return 2 * s / wallis(int(s - H))


def pairings(tj):
    """the three pairings of legs 2-5, as re-ordered doubled tuples (S, a, b, c, d) with pairs (ab|cd)"""
    S, A, B, C, D = tj
    return [(S, A, B, C, D), (S, A, C, B, D), (S, A, D, B, C)]


def valid(tj):
    return all(t >= 1 for t in tj) and sum(tj) % 2 == 0

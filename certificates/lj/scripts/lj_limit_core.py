"""Large-j1 limit of T with three bounded legs.  Pairing (j2 j3 | j4 j5), j5 = s - M, c = s + gamma.
Sites (b, gamma): b in [|j2-j3|, j2+j3], gamma in [-M-j4, j4-M], |gamma| <= b  (valid once s >= s_min).
Exact links (nonnegative gauge of [DC] Lemma 2):
  X (b-1,g)-(b,g):  w = (1/4) h1(b) sqrt(b^2-g^2) sqrt((2s+1+g)^2 - b^2)
  D (b,g-1)-(b,g):  w = (1/4) sqrt(kD) sqrt(N(s)/(4(s+g)^2-1)),
        kD = (j4+1-M-g)(g+M+j4)(b-g+1)(b+g),
        N  = (2s+j4+1-M+g)(2s+g-M-j4)(2s+g-b)(2s+b+g+1)
Limits w/s -> (1/2) h1 sqrt(b^2-g^2)  and  (1/2) sqrt(kD).  Colour = (b+g) mod 2.
All spins given DOUBLED as integers; internally Fractions."""
from fractions import Fraction as Fr
import numpy as np


def h1sq(b, j2, j3):
    n1 = j2 + j3 + 1; m1 = abs(j2 - j3)
    return (n1 * n1 - b * b) * (b * b - m1 * m1) / (4 * b * b - 1)


def sites(t2, M2):
    j2, j3, j4 = (Fr(x, 2) for x in t2); M = Fr(M2, 2)
    out = []
    b = abs(j2 - j3)
    while b <= j2 + j3:
        g = -M - j4
        while g <= j4 - M:
            if abs(g) <= b:
                out.append((b, g))
            g += 1
        b += 1
    return out


def links(t2, M2):
    """list of (i_lo, i_up, kind, data) with exact squared limit weight and the data needed for error bounds."""
    j2, j3, j4 = (Fr(x, 2) for x in t2); M = Fr(M2, 2)
    S = sites(t2, M2); idx = {p: i for i, p in enumerate(S)}
    L = []
    for (b, g), i in idx.items():
        if (b - 1, g) in idx:
            c2 = h1sq(b, j2, j3) * (b * b - g * g)          # w_inf^2 = c2/4
            if c2 > 0:
                L.append((idx[(b - 1, g)], i, 'X', dict(c2=c2, b=b, g=g)))
        if (b, g - 1) in idx:
            kD = (j4 + 1 - M - g) * (g + M + j4) * (b - g + 1) * (b + g)
            if kD > 0:
                L.append((idx[(b, g - 1)], i, 'D', dict(c2=kD, b=b, g=g, al=[j4 + 1 - M + g, g - M - j4, g - b, b + g + 1])))
    return S, L


def w_exact(link, s, t2, M2):
    kind, d = link[2], link[3]
    b, g = d['b'], d['g']
    if kind == 'X':
        return 0.25 * float(d['c2']) ** .5 * max(0.0, (2 * s + 1 + float(g)) ** 2 - float(b) ** 2) ** .5
    N = np.prod([2 * s + float(a) for a in d['al']])
    return 0.25 * float(d['c2']) ** .5 * (N / (4 * (s + float(g)) ** 2 - 1)) ** .5


def polymul(p, q):
    r = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for k, c in enumerate(q):
            r[i + k] += a * c
    return r


def err_bound(link, s0):
    """Rigorous (exact-rational, then one sqrt) upper bound on |w(s) - s w_inf| for all real s >= s0,
    returned as (sqrt of c2/16) * factor, with factor an exact Fraction (the caller multiplies by sqrt(c2)/4)."""
    kind, d = link[2], link[3]
    b, g = d['b'], d['g']; s0 = Fr(s0)
    if kind == 'X':
        # |sqrt((2s+1+g)^2-b^2) - 2s| <= |4s(1+g) + (1+g)^2 - b^2| / (2s) <= 2|1+g| + |(1+g)^2-b^2|/(2 s0)
        return 2 * abs(1 + g) + abs((1 + g) ** 2 - b * b) / (2 * s0)
    # |sqrt(N/D) - 2s| <= |N - 4 s^2 D| / (2 s D),  D = 4(s+g)^2 - 1
    N = [Fr(1)]
    for a in d['al']:
        N = polymul(N, [Fr(a), Fr(2)])                     # coefficients low->high in s
    D = [4 * g * g - 1, 8 * g, Fr(4)]
    P = N[:] + [Fr(0)] * 0
    sD = polymul([Fr(0), Fr(0), Fr(4)], D)              # 4 s^2 D
    P = [N[k] - (sD[k] if k < len(sD) else 0) for k in range(max(len(N), len(sD)))]
    assert P[4] == 0
    num = sum(abs(P[k]) * s0 ** (k - 3) for k in range(4))   # |P(s)| <= s^3 * num  (s >= s0 >= 1)
    den = 8 * ((1 - abs(g) / s0) ** 2 - Fr(1, 4) / (s0 * s0))  # 2 s D >= s^3 * den
    assert den > 0
    return num / den

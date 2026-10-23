"""Core tools for the open-region study of T = J1.(J2xJ3 + J4xJ5) on Inv(j1..j5).

Independent implementation (imports nothing from other research folders).
Spins are passed DOUBLED (integers): tj = (2s, 2j2, 2j3, 2j4, 2j5); pairing (23|45).
Sites (b,c) are stored doubled: B = 2b, C = 2c.  u = b+c, d = b-c.

Closed forms (4-valent links, [SL] section 1 / [GD]):
  X-link (b-1,c)-(b,c):  x^2 = h1(b)^2 * F(d-1/2) * G(u) / 16
  D-link (b,c-1)-(b,c):  y^2 = h2(c)^2 * F(d+1/2) * G(u) / 16
  h_P(L)^2 = (n_P^2-L^2)(L^2-m_P^2)/(4L^2-1)   (hydrogen Stark link)
  F(t) = (s+1/2)^2 - t^2,  G(u) = (u+s+1)(u-s)   [upper end (b,c) of the link]
Normalisation: BR operator A = -8iT, q = 8|eig T|.
"""
import numpy as np
from fractions import Fraction
import itertools


def sites(tj):
    s, a, b_, c_, d_ = tj
    out = []
    for B in range(abs(a - b_), a + b_ + 1, 2):
        for C in range(abs(c_ - d_), c_ + d_ + 1, 2):
            if abs(B - C) <= s <= B + C and (B + C + s) % 2 == 0:
                out.append((B, C))
    return out


def rect(tj):
    s, a, b_, c_, d_ = tj
    return [(B, C) for B in range(abs(a - b_), a + b_ + 1, 2) for C in range(abs(c_ - d_), c_ + d_ + 1, 2)]


def h2(nP, mP, L):
    """Stark link squared, arguments undoubled (Fractions ok)."""
    return (nP * nP - L * L) * (L * L - mP * mP) / (4 * L * L - 1)


def F(s, t):
    return (s + Fraction(1, 2)) ** 2 - t * t


def G(s, u):
    return (u + s + 1) * (u - s)


def pars(tj):
    s = Fraction(tj[0], 2)
    j = [Fraction(t, 2) for t in tj[1:]]
    n1, m1 = j[0] + j[1] + 1, abs(j[0] - j[1])
    n2, m2 = j[2] + j[3] + 1, abs(j[2] - j[3])
    return s, n1, m1, n2, m2


def links(tj, kind='T'):
    """Exact squared links.  kind: 'T' (true operator), 'K' (Stark Kronecker, W0),
    'KF' (Stark x sqrt F, i.e. omega factored out), 'KG' (Stark x sqrt G, phi factored out).
    Returns (sites, dict {(i,k): w2})."""
    s, n1, m1, n2, m2 = pars(tj)
    S = sites(tj)
    idx = {p: i for i, p in enumerate(S)}
    out = {}
    for (B, C), i in idx.items():
        b, c = Fraction(B, 2), Fraction(C, 2)
        u, d = b + c, b - c
        k = idx.get((B - 2, C))  # X link, upper end (b,c)
        if k is not None:
            w = h2(n1, m1, b)
            if kind in ('T', 'KF'):
                w *= F(s, d - Fraction(1, 2))
            if kind in ('T', 'KG'):
                w *= G(s, u)
            if kind == 'T':
                w /= 16
            out[(k, i)] = w
        k = idx.get((B, C - 2))  # D link, upper end (b,c)
        if k is not None:
            w = h2(n2, m2, c)
            if kind in ('T', 'KF'):
                w *= F(s, d + Fraction(1, 2))
            if kind in ('T', 'KG'):
                w *= G(s, u)
            if kind == 'T':
                w /= 16
            out[(k, i)] = w
    return S, out


def matrix(tj, kind='T'):
    S, L = links(tj, kind)
    W = np.zeros((len(S), len(S)))
    for (i, k), w in L.items():
        W[i, k] = W[k, i] = float(w) ** 0.5
    return S, W


def colour(p, s2):
    """parity of u = (B+C)/2 - well defined since B+C+s2 even;  use floor((B+C)/2) parity."""
    return ((p[0] + p[1]) // 2) % 2


def bip(S, W, s2):
    ev = [i for i, p in enumerate(S) if colour(p, s2) == 0]
    od = [i for i, p in enumerate(S) if colour(p, s2) == 1]
    return W[np.ix_(ev, od)], ev, od


def nz_svals(B, tol=1e-10):
    if B.size == 0:
        return np.array([]), 0
    sv = np.linalg.svd(B, compute_uv=False)
    thr = tol * max(sv.max() if sv.size else 0, 1) * (sum(B.shape))
    nz = sv[sv > thr]
    return nz, (min(B.shape) - nz.size)


def lam(tj, kind='T'):
    S, W = matrix(tj, kind)
    if len(S) < 2:
        return None
    B, ev, od = bip(S, W, tj[0])
    nz, _ = nz_svals(B)
    return float(nz.min()) if nz.size else None


def closed_classes(tj):
    """For pairing (23|45): which colour classes are closed (all rectangle-neighbours of class sites lie in R)."""
    S = set(sites(tj))
    Rr = set(rect(tj))
    res = [True, True]
    for p in S:
        for q in ((p[0] + 2, p[1]), (p[0] - 2, p[1]), (p[0], p[1] + 2), (p[0], p[1] - 2)):
            if q in Rr and q not in S:
                res[colour(p, tj[0])] = False
    return res


def pairings(tj):
    s, a, b, c, d = tj
    return [(s, a, b, c, d), (s, a, c, b, d), (s, a, d, b, c)]


def is_open(tj):
    """Half-integer s >= 3/2 and no pairing has a closed colour class (the 'open regions')."""
    if tj[0] % 2 == 0 or tj[0] < 3:
        return False
    if len(sites(tj)) < 2:
        return False
    for P in pairings(tj):
        if any(closed_classes(P)):
            return False
    return True


def tuples_sorted(M, smin=1):
    """2s in [smin..M], multisets 2j2<=2j3<=2j4<=2j5<=M, integer total."""
    for s2 in range(smin, M + 1):
        for a in range(1, M + 1):
            for b in range(a, M + 1):
                for c in range(b, M + 1):
                    for d in range(c, M + 1):
                        if (s2 + a + b + c + d) % 2 == 0:
                            yield (s2, a, b, c, d)

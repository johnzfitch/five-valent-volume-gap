"""The normal-form operator A = -4i[alpha J1.(J2xJ3) + beta J1.(J4xJ5)] on the (b,c) grid.

Basis |b,c>, b = |J2+J3|, c = |J4+J5|, (s,b,c) a triangle, s = j1. In a flux-free
gauge (proved in the paper) the eigenvalue moduli of iA equal the spectrum of the
nonnegative matrix with links

  X-link (b-1,c)-(b,c):  |alpha| * h1(b) * sqrt(F(d-1/2) G(u))
  D-link (b,c-1)-(b,c):  |beta|  * h2(c) * sqrt(F(d+1/2) G(u))

with (b,c) the upper end, d = b-c, u = b+c, h_P(L)^2 = (n_P^2-L^2)(L^2-m_P^2)/(4L^2-1),
F(t) = (s+1/2)^2 - t^2, G(u) = (u+s+1)(u-s). For alpha = beta = 2 this is 8T.
All inputs doubled integers. Exact squared links are returned as Fractions.
"""
from fractions import Fraction as Fr
import numpy as np


def pair_data(ta, tb):
    """(n, m) of a pair with doubled spins: n = ja+jb+1, m = |ja-jb| (as Fractions)."""
    return Fr(ta + tb, 2) + 1, Fr(abs(ta - tb), 2)


def h2(n, m, L):
    return (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)


def sites(sp):
    """Sites (tb, tc) doubled, for doubled spins sp = (s,j2,j3,j4,j5)."""
    ts, t2, t3, t4, t5 = sp
    out = []
    for tb in range(abs(t2 - t3), t2 + t3 + 1, 2):
        for tc in range(abs(t4 - t5), t4 + t5 + 1, 2):
            if abs(tb - tc) <= ts <= tb + tc and (tb + tc + ts) % 2 == 0:
                out.append((tb, tc))
    return out


def links(sp, alpha=2, beta=2):
    """Exact squared link weights {((tb,tc),(tb',tc')): Fraction} of iA."""
    ts, t2, t3, t4, t5 = sp
    s = Fr(ts, 2)
    n1, m1 = pair_data(t2, t3)
    n2, m2 = pair_data(t4, t5)
    S = set(sites(sp))
    out = {}
    for (tb, tc) in S:
        b, c = Fr(tb, 2), Fr(tc, 2)
        d, u = b - c, b + c
        G = (u + s + 1) * (u - s)
        if (tb - 2, tc) in S:     # X-link, upper end (b,c)
            F = (s + Fr(1, 2)) ** 2 - (d - Fr(1, 2)) ** 2
            out[((tb - 2, tc), (tb, tc))] = alpha * alpha * h2(n1, m1, b) * F * G
        if (tb, tc - 2) in S:     # D-link
            F = (s + Fr(1, 2)) ** 2 - (d + Fr(1, 2)) ** 2
            out[((tb, tc - 2), (tb, tc))] = beta * beta * h2(n2, m2, c) * F * G
    return S, out


def bipartite(sp, alpha=2, beta=2):
    """Float bipartite block (rows: colour 0, columns: colour 1) of the nonnegative grid."""
    S, L = links(sp, alpha, beta)
    ts = sp[0]
    col = {x: ((x[0] + x[1] - ts) // 2) % 2 for x in S}
    R = sorted(x for x in S if col[x] == 0)
    C = sorted(x for x in S if col[x] == 1)
    ri = {x: i for i, x in enumerate(R)}
    ci = {x: i for i, x in enumerate(C)}
    B = np.zeros((len(R), len(C)))
    for (x, y), w2 in L.items():
        w = float(w2) ** 0.5
        if col[x] == 0:
            B[ri[x], ci[y]] = w
        else:
            B[ri[y], ci[x]] = w
    return B


def full_matrix(sp, alpha=2, beta=2):
    S, L = links(sp, alpha, beta)
    idx = {x: i for i, x in enumerate(sorted(S))}
    W = np.zeros((len(S), len(S)))
    for (x, y), w2 in L.items():
        W[idx[x], idx[y]] = W[idx[y], idx[x]] = float(w2) ** 0.5
    return W


def qmin(sp, alpha=2, beta=2, rel=1e-12):
    """Smallest nonzero eigenvalue modulus of iA (float64), and kernel dimension."""
    B = bipartite(sp, alpha, beta)
    if B.size == 0:
        return None, B.shape[0] + B.shape[1]
    sv = np.linalg.svd(B, compute_uv=False)
    tol = rel * max(1.0, sv.max()) * max(B.shape)
    nz = sv[sv > tol]
    ker = B.shape[0] + B.shape[1] - 2 * len(nz)
    return (nz.min() if len(nz) else None), ker

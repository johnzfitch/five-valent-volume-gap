"""Layer (antidiagonal u = b+c) structure of T.
Gamma[u]: block of T from layer u-1 (columns) to layer u (rows), in the nonnegative gauge.
Identity to test:  Gamma[u+1]^T Gamma[u+1] - (G(u+1)/G(u)) Gamma[u] Gamma[u]^T = diag(Lambda^u)
with Lambda^u_x = G(u+1)/16 [F(d+1/2)(h1(b+1)^2-h2(c)^2) + F(d-1/2)(h2(c+1)^2-h1(b)^2)]."""
import numpy as np
from fractions import Fraction as Fr
from t1core import grid, pair_params, h2stark


def layer_data(S, a2, a3, a4, a5, exact=False):
    st, links = grid(S, a2, a3, a4, a5, exact=exact)
    lay = {}
    for i, (b, c) in enumerate(st):
        lay.setdefault((b + c) // 2 if (b + c) % 2 == 0 else Fr(b + c, 2), []).append(i)
    # use doubled u for keys to be safe
    layD = {}
    for i, (b, c) in enumerate(st):
        layD.setdefault(b + c, []).append(i)
    for k in layD:
        layD[k].sort(key=lambda i: st[i][0])
    pos = {}
    for k, L in layD.items():
        for j, i in enumerate(L):
            pos[i] = (k, j)
    Gam = {}
    for i, k_, w2 in links:
        # link between st[i] (lower u) and st[k_] (upper u) -- grid stores (lower, upper)
        lo, hi = i, k_
        U = st[hi][0] + st[hi][1]
        if U not in Gam:
            Gam[U] = np.zeros((len(layD[U]), len(layD.get(U - 2, []))), dtype=object if exact else float)
        Gam[U][pos[hi][1], pos[lo][1]] = w2 if exact else np.sqrt(w2)
    return st, layD, Gam


def Lambda_closed(S, a2, a3, a4, a5, b2, c2):
    """closed-form Lambda^u at doubled site (b2, c2)."""
    n1, m1 = pair_params(a2, a3); n2, m2 = pair_params(a4, a5)
    s = Fr(S, 2); b = Fr(b2, 2); c = Fr(c2, 2); d = b - c; u = b + c
    F = lambda t: (s + Fr(1, 2)) ** 2 - t * t
    G = lambda x: (x + s + 1) * (x - s)
    def hh(n, m, L):  # Stark h^2 at label L, zero outside [m+1, n-1]
        Lx = 2 * L
        if L <= Fr(m, 2) or L >= Fr(n, 2):
            return Fr(0)
        return h2stark(n, m, int(Lx))
    return G(u + 1) / 16 * (F(d + Fr(1, 2)) * (hh(n1, m1, b + 1) - hh(n2, m2, c))
                            + F(d - Fr(1, 2)) * (hh(n2, m2, c + 1) - hh(n1, m1, b)))

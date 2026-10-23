"""W0 = P_R K P_R (compressed Kronecker sum of Stark matrices) and the Heron weights.
Doubled-spin inputs. Pairings of legs 2..5 are given as ((a,b),(c,d)) of doubled spins."""
import numpy as np
from fractions import Fraction as Fr
from t1core import sites, pair_params, h2stark, colour


def W0_block(S, p1, p2, col=0):
    """Bipartite block of W0 (links = Stark links h1(b) or h2(c)) from class col to the other,
    for pairs p1=(a2,a3), p2=(a4,a5). Returns (B0, cols, rows, sigma2 dict)."""
    a2, a3 = p1; a4, a5 = p2
    st = sites(S, a2, a3, a4, a5)
    idx = {x: i for i, x in enumerate(st)}
    n1, m1 = pair_params(a2, a3)
    n2, m2 = pair_params(a4, a5)
    n = len(st)
    W = np.zeros((n, n))
    for (b, c), i in idx.items():
        if (b - 2, c) in idx:
            W[idx[(b - 2, c)], i] = W[i, idx[(b - 2, c)]] = float(h2stark(n1, m1, b)) ** 0.5
        if (b, c - 2) in idx:
            W[idx[(b, c - 2)], i] = W[i, idx[(b, c - 2)]] = float(h2stark(n2, m2, c)) ** 0.5
    cols = [i for i, x in enumerate(st) if colour(S, x) == col]
    rows = [i for i, x in enumerate(st) if colour(S, x) != col]
    return W[np.ix_(rows, cols)], [st[i] for i in cols], [st[i] for i in rows]


def lam(B, tol=1e-9):
    if B.size == 0:
        return None
    sv = np.linalg.svd(B, compute_uv=False)
    nz = sv[sv > tol * max(1, sv.max())]
    return nz.min() if len(nz) else None


def pairings(a2, a3, a4, a5):
    return [((a2, a3), (a4, a5)), ((a2, a4), (a3, a5)), ((a2, a5), (a3, a4))]


def cut_colours(S, p1, p2):
    """Set of colours of cut sites (sites of R with a rectangle neighbour outside R)."""
    a2, a3 = p1; a4, a5 = p2
    st = set(sites(S, a2, a3, a4, a5))
    n1, m1 = pair_params(a2, a3)
    n2, m2 = pair_params(a4, a5)
    cols = set()
    for (b, c) in st:
        for (bb, cc) in [(b - 2, c), (b + 2, c), (b, c - 2), (b, c + 2)]:
            if m1 <= bb <= n1 - 2 and m2 <= cc <= n2 - 2 and (bb, cc) not in st:
                cols.add(colour(S, (b, c)))
    return cols


def covered(S, a2, a3, a4, a5):
    """Thm 6.5 (closed + odd sum or spin-1/2 pair) or ladder (some leg 1/2) or j1<=2."""
    if S <= 4:
        return 'slice'
    if 1 in (a2, a3, a4, a5):
        return 'ladder'
    for p1, p2 in pairings(a2, a3, a4, a5):
        cc = cut_colours(S, p1, p2)
        closed = [k for k in (0, 1) if k not in cc]
        if not closed:
            continue
        N1 = min(p1) + 1; N2 = min(p2) + 1
        if (N1 + N2) % 2 == 1:
            return 'closed-odd'
    for p1, p2 in pairings(a2, a3, a4, a5):
        cc = cut_colours(S, p1, p2)
        if len(cc) < 2:
            return 'W2'
    return 'open'

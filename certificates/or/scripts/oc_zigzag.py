"""Zigzag decomposition of the bipartite block of T for half-integer s.

Rows   = sites on diagonals d = -s, -s+2, ..., s-1   (class containing the wall d=-s)
Columns= sites on diagonals d = -s+1, ..., s          (class containing the wall d=+s)
Zigzag k = diagonals {d_k, d_k+1}, d_k = -s+2k, k = 0..s-1/2.
Claim: B[rows of zigzag k, cols of zigzag l] = 0 unless l in {k, k-1}  (block lower bidiagonal),
       and B_kk is a path (alternating c-hop / b-hop along u).
Functions return the blocks so that other scripts can analyse them.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_fast import region

np.seterr(divide='ignore', invalid='ignore')


def zig_blocks(tj, kind='T'):
    """Return (B, row_k, col_k, rows, cols) with B rows = row-class sites, cols = column-class sites,
    row_k / col_k = zigzag index of each row / column, rows/cols = list of (2b,2c).  Requires 2s odd."""
    s2, a, b_, c_, d_ = tj
    assert s2 % 2 == 1
    s = s2 / 2
    n1, m1 = (a + b_) / 2 + 1, abs(a - b_) / 2
    n2, m2 = (c_ + d_) / 2 + 1, abs(c_ - d_) / 2
    Bd, Cd = region(tj)
    D2 = Bd - Cd                    # 2d
    k2 = D2 + s2                    # 2(d+s) = 0,1,...,2s
    isrow = ((k2 // 2) % 2 == 0)       # d+s even
    rows = [(int(x), int(y)) for x, y, r in zip(Bd, Cd, isrow) if r]
    cols = [(int(x), int(y)) for x, y, r in zip(Bd, Cd, isrow) if not r]
    ri = {p: i for i, p in enumerate(rows)}; ci = {p: i for i, p in enumerate(cols)}
    row_k = np.array([((x - y) + s2) // 4 for x, y in rows])
    col_k = np.array([((x - y) + s2 - 2) // 4 for x, y in cols])
    M = np.zeros((len(rows), len(cols)))

    def w(up, isX):
        b, c = up[0] / 2, up[1] / 2
        u, d = b + c, b - c
        if isX:
            h2 = (n1 ** 2 - b ** 2) * (b ** 2 - m1 ** 2) / (4 * b ** 2 - 1); db = d - 0.5
        else:
            h2 = (n2 ** 2 - c ** 2) * (c ** 2 - m2 ** 2) / (4 * c ** 2 - 1); db = d + 0.5
        F = (s + 0.5) ** 2 - db ** 2; G = (u + s + 1) * (u - s)
        val = {'T': h2 * F * G / 16, 'KG': h2 * G, 'K': h2, 'KF': h2 * F}[kind]
        return np.sqrt(max(val, 0.0))

    for p, i in ri.items():
        for (q, isX, up) in (((p[0] + 2, p[1]), True, (p[0] + 2, p[1])), ((p[0] - 2, p[1]), True, p),
                             ((p[0], p[1] + 2), False, (p[0], p[1] + 2)), ((p[0], p[1] - 2), False, p)):
            j = ci.get(q)
            if j is not None:
                M[i, j] = w(up, isX)
    return M, row_k, col_k, rows, cols


def check_structure(tj):
    M, rk, ck, rows, cols = zig_blocks(tj)
    bad = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] != 0 and ck[j] not in (rk[i], rk[i] - 1):
                bad += 1
    # each zigzag block is a path: every row has <=2 entries in its own zigzag, every col <=2
    path_ok = True
    for k in set(rk.tolist()):
        R = np.where(rk == k)[0]; C = np.where(ck == k)[0]
        Bk = M[np.ix_(R, C)]
        if Bk.size and ((Bk != 0).sum(1).max() > 2 or (Bk != 0).sum(0).max() > 2):
            path_ok = False
    return bad, path_ok


if __name__ == '__main__':
    import itertools
    from oc_fast import is_open
    M = int(sys.argv[1])
    n = 0; nb = 0; npath = 0
    for tj in itertools.product(range(1, M + 1), repeat=5):
        if tj[0] % 2 == 0 or sum(tj) % 2 or len(region(tj)[0]) < 2:
            continue
        bad, pok = check_structure(tj)
        n += 1; nb += (bad > 0); npath += (not pok)
    print(f"half-integer s, all tuples 2j<={M}: {n} tuples; block-bidiagonal violations: {nb}; non-path zigzag blocks: {npath}")

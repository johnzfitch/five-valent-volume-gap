"""d-direction (zigzag) left inverse of T's bipartite block, half-integer s.
Diagonals k = 0..2s.  Reading A ('up'): rows = even k, forward from k=0; needs G^-_{odd k} > 0.
Reading B ('down'): rows = odd k, forward from k=2s; needs G^+_{even k} > 0.
Returns 8/||L|| (a lower bound for q when the reading is admissible) or None."""
import numpy as np
from tladder import tblocks

def leftinv_bound(S, legs, direction):
    sites, W, V, N = tblocks(S, legs)
    s = S / 2
    K = S + 1
    R = [W[np.ix_(V[k + 1], V[k])] for k in range(S)]   # raising V_k -> V_{k+1}
    if direction == 'up':
        rows = list(range(0, K, 2)); cols = list(range(1, K, 2))
        # row 2i: y = R_{2i-1} x_{2i-1} + R_{2i}^T x_{2i+1}; solve x_{2i+1}
        solve = {2*i+1: (R[2*i].T, (R[2*i-1] if 2*i-1 >= 0 else None), 2*i-1) for i in range(len(rows))}
        order = cols
        rowof = {c: c - 1 for c in cols}
    else:
        rows = list(range(S, -1, -2)); cols = list(range(S - 1, -1, -2))
        # row r (odd): y = R_{r-1} x_{r-1} + R_r^T x_{r+1}; solve x_{r-1} given x_{r+1}
        solve = {r-1: (R[r-1], (R[r].T if r < S else None), r+1) for r in rows}
        order = cols
        rowof = {c: c + 1 for c in cols}
    # check injectivity of the solving blocks
    for c in order:
        Aslv = solve[c][0]
        if Aslv.shape[0] < Aslv.shape[1] or np.linalg.svd(Aslv, compute_uv=False).min() < 1e-9 * max(1, np.abs(Aslv).max()):
            return None
    # build L: x_c = sum_r L[c][r] y_r
    rdim = {r: len(V[r]) for r in rows}
    L = {}
    for c in order:
        Aslv, Aprev, cprev = solve[c]
        P = np.linalg.pinv(Aslv)
        r = rowof[c]
        Lc = {r: P.copy()}
        if Aprev is not None and cprev in L:
            for rr, M in L[cprev].items():
                Lc[rr] = Lc.get(rr, 0) - P @ Aprev @ M
        L[c] = Lc
    # assemble dense
    ri = {}; off = 0
    for r in rows: ri[r] = off; off += rdim[r]
    ci = {}; off2 = 0
    for c in order: ci[c] = off2; off2 += len(V[c])
    Ld = np.zeros((off2, off))
    for c in order:
        for r, M in L[c].items():
            Ld[ci[c]:ci[c]+len(V[c]), ri[r]:ri[r]+rdim[r]] = M
    return 8 / np.linalg.norm(Ld, 2)

"""General zigzag (d-direction) left inverse of T's bipartite block (T-picture), any s.
A reading = list of steps (row r, solved column c, other column c' or None) solving each column of the
chosen class exactly once; rows not used are dropped (dropping rows only lowers ||Bx||).
Admissible iff every solving block is injective.  Returns 8/||L|| (lower bound for q on that class)
and per-step data."""
import numpy as np
from t1core import Tmatrix

def tdiag(S, legs):
    sites, W = Tmatrix(S, *legs)
    by = {}
    for i, (b, c) in enumerate(sites):
        by.setdefault((b - c + S) // 2, []).append(i)
    V = [sorted(by.get(k, []), key=lambda i: sites[i][0] + sites[i][1]) for k in range(S + 1)]
    return sites, W, V

def readings(S):
    """Readings with columns = odd k (rows = even k).  For half-integer s (S odd) also columns = even k."""
    out = {}
    K = S + 1
    # columns odd k, rows even k
    up = [(r, r + 1, (r - 1) if r - 1 >= 1 else None) for r in range(0, S, 2) if r + 1 <= S]
    out['odd-up'] = up
    if S % 2 == 0:
        # rows even incl. top row S: read down from the top
        down = [(r, r - 1, (r + 1) if r + 1 <= S - 1 else None) for r in range(S, 0, -2)]
        out['odd-down'] = down
        for ks in range(2, S, 2):     # drop row ks, lower part up, upper part down
            lo = [(r, r + 1, (r - 1) if r - 1 >= 1 else None) for r in range(0, ks, 2)]
            hi = [(r, r - 1, (r + 1) if r + 1 <= S - 1 else None) for r in range(S, ks, -2)]
            out[f'odd-split{ks}'] = lo + hi
    else:
        # columns even k, rows odd k: read down from the top row S (odd)
        down = [(r, r - 1, (r + 1) if r + 1 <= S - 1 else None) for r in range(S, 0, -2)]
        out['even-down'] = down
    return out

def block(W, V, r, c):
    return W[np.ix_(V[r], V[c])]

def leftinv(S, legs, steps, data=None):
    sites, W, V = data if data is not None else tdiag(S, legs)
    cols = [c for (_, c, _) in steps]
    rows = [r for (r, _, _) in steps]
    L = {}
    for (r, c, cp) in steps:
        A = block(W, V, r, c)
        if A.shape[0] < A.shape[1]:
            return None
        sv = np.linalg.svd(A, compute_uv=False)
        if sv.size and sv.min() < 1e-10 * max(1.0, sv.max()):
            return None
        P = np.linalg.pinv(A)
        Lc = {r: P}
        if cp is not None:
            Bo = block(W, V, r, cp)
            for rr, M in L[cp].items():
                Lc[rr] = Lc.get(rr, 0) - P @ Bo @ M
        L[c] = Lc
    ri, off = {}, 0
    for r in rows: ri[r] = off; off += len(V[r])
    ci, off2 = {}, 0
    for c in cols: ci[c] = off2; off2 += len(V[c])
    Ld = np.zeros((off2, off))
    for c in cols:
        for r, M in L[c].items():
            Ld[ci[c]:ci[c] + len(V[c]), ri[r]:ri[r] + len(V[r])] = M
    return 8 / np.linalg.norm(Ld, 2)

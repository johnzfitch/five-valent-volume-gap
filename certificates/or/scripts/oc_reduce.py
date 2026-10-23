"""Square reduction of the zigzag-triangular block (exploration + checks).

For a pairing with 2s odd: B (rows: d+s even, cols: d+s odd), zigzag k = diagonals {d_k, d_k+1}.
Imbalance imb_k = #rows_k - #cols_k in {-1,0,1}.  Delete one end site of every imbalanced zigzag
(end chosen by `end`), giving B' square, block lower bidiagonal, with square bidiagonal diagonal blocks
(invertible since all links are nonzero).  If dim ker T = sum_k |imb_k| then rank B' = rank B and, by
singular-value interlacing,  lam(T) = smallest nonzero sv of B  >=  sigma_min(B').
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_zigzag import zig_blocks


def zig_paths(tj, kind='T'):
    B, rk, ck, rows, cols = zig_blocks(tj, kind)
    K = tj[0] // 2 + 1
    Z = []
    for k in range(K):
        R = [i for i in range(len(rows)) if rk[i] == k]
        C = [j for j in range(len(cols)) if ck[j] == k]
        # order by u
        R.sort(key=lambda i: rows[i][0] + rows[i][1]); C.sort(key=lambda j: cols[j][0] + cols[j][1])
        Z.append((R, C))
    return B, rk, ck, rows, cols, Z


def reduce_square(tj, end='low', kind='T'):
    """Return (Bp, keepR, keepC, imb, info).  end: which end to delete for imbalanced zigzags ('low'/'high')."""
    B, rk, ck, rows, cols, Z = zig_paths(tj, kind)
    keepR, keepC, imb = [], [], []
    for R, C in Z:
        im = len(R) - len(C); imb.append(im)
        R2, C2 = list(R), list(C)
        if im == 1:
            R2 = R2[1:] if end == 'low' else R2[:-1]
        elif im == -1:
            C2 = C2[1:] if end == 'low' else C2[:-1]
        elif abs(im) > 1:
            raise ValueError('zigzag imbalance > 1 ' + str(tj))
        keepR += R2; keepC += C2
    Bp = B[np.ix_(keepR, keepC)]
    return B, Bp, keepR, keepC, imb


def ker_dim(B):
    if B.size == 0:
        return sum(B.shape)
    sv = np.linalg.svd(B, compute_uv=False)
    rank = int((sv > 1e-9 * sv.max() * sum(B.shape)).sum())
    return sum(B.shape) - 2 * rank


def monotone_ok(imb):
    """No wide block (-1) before a tall block (+1) in k order."""
    seen_wide = False
    for x in imb:
        if x == -1:
            seen_wide = True
        if x == 1 and seen_wide:
            return False
    return True


def pairing_matrix(tj, end='low', kind='T'):
    """Square reduced B' plus the M-matrix A = (pairing part) - (rest), where the pairing pairs each kept row
    with the kept column of its own zigzag that makes the diagonal block bidiagonal-triangular.
    Returns (B, Bp, A, perm) with perm[i] = column index (in Bp) paired with row i."""
    B, rk, ck, rows, cols, Z = zig_paths(tj, kind)
    keepR, keepC, perm, imb = [], [], [], []
    for R, C in Z:
        im = len(R) - len(C); imb.append(im)
        R2, C2 = list(R), list(C)
        if im == 1:
            R2 = R2[1:] if end == 'low' else R2[:-1]
        elif im == -1:
            C2 = C2[1:] if end == 'low' else C2[:-1]
        base = len(keepC)
        # pair i-th row with i-th column (both sorted by u) -- along a balanced path this is a link
        for i in range(len(R2)):
            perm.append(base + i)
        keepR += R2; keepC += C2
    Bp = B[np.ix_(keepR, keepC)]
    P = np.zeros_like(Bp)
    for i, j in enumerate(perm):
        P[i, j] = Bp[i, j]
    A = P - (Bp - P)
    return B, Bp, A, perm, imb


def reduce_choice(tj, choice, kind='T'):
    """Square reduction with a per-zigzag end choice: choice[k] in ('low','high') (ignored for balanced zigzags)."""
    B, rk, ck, rows, cols, Z = zig_paths(tj, kind)
    keepR, keepC, imb = [], [], []
    for k, (R, C) in enumerate(Z):
        im = len(R) - len(C); imb.append(im)
        R2, C2 = list(R), list(C)
        e = choice[k]
        if im == 1:
            R2 = R2[1:] if e == 'low' else R2[:-1]
        elif im == -1:
            C2 = C2[1:] if e == 'low' else C2[:-1]
        keepR += R2; keepC += C2
    return B, B[np.ix_(keepR, keepC)], imb


def best_reduction(tj, kind='T', max_enum=10):
    """Max over monotone pairings and per-zigzag end choices of sigma_min(B').  Returns (sigma, pairing, choice, imb)
    or None if no pairing is monotone.  Enumerates all choices if #imbalanced <= max_enum, else greedy."""
    import itertools
    from oc_fast import pairings
    best = None
    for P in pairings(tj):
        B, rk, ck, rows, cols, Z = zig_paths(P, kind)
        imb = [len(R) - len(C) for R, C in Z]
        if not monotone_ok(imb):
            continue
        idx = [k for k, x in enumerate(imb) if x != 0]
        K = len(Z)
        if len(idx) <= max_enum:
            choices = []
            for bits in itertools.product(('low', 'high'), repeat=len(idx)):
                ch = ['low'] * K
                for k, bb in zip(idx, bits):
                    ch[k] = bb
                choices.append(ch)
        else:
            choices = [['low'] * K, ['high'] * K]
        for ch in choices:
            _, Bp, _ = reduce_choice(P, ch, kind)
            if Bp.size == 0:
                continue
            sv = np.linalg.svd(Bp, compute_uv=False).min()
            if best is None or sv > best[0]:
                best = (sv, P, tuple(ch), tuple(imb))
    return best


def maxvol_reduction(tj, kind='T', sweeps=3):
    """Square reduction deleting ONE ARBITRARY row from each tall zigzag and one arbitrary column from each
    wide zigzag (any such choice leaves square invertible zigzag blocks), chosen to (locally) maximise
    |det Y_S| (Y = orthonormal basis of ker B^T restricted to deleted rows) and |det X_S| for columns.
    Returns (Bp, rowsdeleted, colsdeleted, imb) for the given pairing tj (must be monotone)."""
    B, rk, ck, rows, cols, Z = zig_paths(tj, kind)
    imb = [len(R) - len(C) for R, C in Z]
    U, sv, Vt = np.linalg.svd(B)
    thr = 1e-9 * sv.max() * sum(B.shape)
    r = int((sv > thr).sum())
    Y = U[:, r:]                      # left kernel (rows)
    X = Vt[r:, :].T                   # right kernel (cols)

    def pick(Kmat, groups):
        sel = [g[0] for g in groups]
        if Kmat.shape[1] != len(groups) or len(groups) == 0:
            return sel
        for _ in range(sweeps):
            for gi, g in enumerate(groups):
                bestv, bestx = -1, sel[gi]
                for x in g:
                    trial = list(sel); trial[gi] = x
                    v = abs(np.linalg.det(Kmat[trial, :]))
                    if v > bestv:
                        bestv, bestx = v, x
                sel[gi] = bestx
        return sel
    tall = [R for (R, C), m in zip(Z, imb) if m == 1]
    wide = [C for (R, C), m in zip(Z, imb) if m == -1]
    dr = pick(Y, tall); dc = pick(X, wide)
    keepR = [i for i in range(B.shape[0]) if i not in set(dr)]
    keepC = [j for j in range(B.shape[1]) if j not in set(dc)]
    return B, B[np.ix_(keepR, keepC)], dr, dc, imb

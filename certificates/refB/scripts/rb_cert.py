"""Exact certificates of 'every nonzero q = 8|eig T| >= q*' for single tuples, from the Racah entries of rb_core.

Method (no floating point anywhere in the decision):
  * exact entries of T: sign * sqrt(rational) (rb_core.R_entries scaled by C0^2/4);
  * bipartite block B (rows: colour class 1, columns: colour class 0);
  * spanning-forest gauge: B = D_r Bt D_c with D_r^2, D_c^2 rational diagonal and Bt rational; every off-tree entry is checked
    to be rational (perfect-square test), otherwise the tuple is reported as 'irrational' (never happened);
  * B^T B - lam I is congruent (by D_c) to S = Bt^T D_r^2 Bt - lam D_c^{-2}, a rational symmetric banded matrix
    (columns ordered by u = b + c);
  * inertia of S by exact LDL^T (python-flint fmpq), all pivots nonzero required;
  * dim ker B = dim ker Bt by exact rank (fmpq_mat);
  * certified  <=>  #negative pivots == dim ker B, i.e. no nonzero sigma(B)^2 lies below lam = (q*/8)^2.
Since the nonzero eigenvalues of T are exactly +-(nonzero singular values of B), this certifies the claim for T.
"""
import sys, os
from fractions import Fraction as Fr
from flint import fmpq, fmpq_mat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C


def fq(x):
    return fmpq(x.numerator, x.denominator)


def certify(tj, qstar):
    """returns (status, info). qstar a Fraction."""
    S, ent, col = C.colour_classes(tj)
    if len(S) < 2 or not ent:
        return 'trivial', {}
    sc = C.scale_sq(tj[0])
    lam = (Fr(qstar) / 8) ** 2
    rows = [i for i in range(len(S)) if col[i] == 1]
    cols = [i for i in range(len(S)) if col[i] == 0]
    # order columns by u, then b ; rows likewise
    key = lambda i: (S[i][0] + S[i][1], S[i][0])
    rows.sort(key=key); cols.sort(key=key)
    pr = {v: k for k, v in enumerate(rows)}; pc = {v: k for k, v in enumerate(cols)}
    edges = []                                   # (r, c, sign, square)
    for (i, j), (sg, sq) in ent.items():
        r, c = (i, j) if col[i] == 1 else (j, i)
        edges.append((pr[r], pc[c], sg, sq * sc))
    nr, nc = len(rows), len(cols)
    # spanning forest gauge: value d^2 (Fraction) and sign for each vertex; vertices: ('r',k) / ('c',k)
    adj = {}
    for e, (r, c, sg, sq) in enumerate(edges):
        adj.setdefault(('r', r), []).append((('c', c), e)); adj.setdefault(('c', c), []).append((('r', r), e))
    d2 = {}; dsg = {}; tree = set()
    for v0 in [('r', k) for k in range(nr)] + [('c', k) for k in range(nc)]:
        if v0 in d2:
            continue
        d2[v0] = Fr(1); dsg[v0] = 1; st = [v0]
        while st:
            v = st.pop()
            for w, e in adj.get(v, []):
                if w in d2:
                    continue
                r, c, sg, sq = edges[e]
                # B_e = d_v d_w * 1  =>  d_w = B_e / d_v
                d2[w] = sq / d2[v]; dsg[w] = sg * dsg[v]
                tree.add(e); st.append(w)
    Bt = {}
    for e, (r, c, sg, sq) in enumerate(edges):
        if e in tree:
            Bt[(r, c)] = Fr(1)
            continue
        ratio = sq / (d2[('r', r)] * d2[('c', c)])
        rt = C.is_square_fr(ratio)
        if rt is None:
            return 'irrational', {}
        Bt[(r, c)] = sg * dsg[('r', r)] * dsg[('c', c)] * rt
    # S = Bt^T Dr^2 Bt - lam Dc^{-2}
    colsof = {}
    for (r, c), v in Bt.items():
        colsof.setdefault(r, []).append((c, v))
    Sm = {}
    for r, lst in colsof.items():
        w = d2[('r', r)]
        for (c1, v1) in lst:
            for (c2, v2) in lst:
                if c1 <= c2:
                    Sm[(c1, c2)] = Sm.get((c1, c2), Fr(0)) + w * v1 * v2
    for c in range(nc):
        Sm[(c, c)] = Sm.get((c, c), Fr(0)) - lam / d2[('c', c)]
    bw = max((c2 - c1 for (c1, c2) in Sm), default=0)
    # banded LDL^T in fmpq
    A = [[fmpq(0)] * (bw + 1) for _ in range(nc)]      # A[i][k] = S[i, i+k]
    for (c1, c2), v in Sm.items():
        A[c1][c2 - c1] = fq(v)
    neg = 0
    for k in range(nc):
        p = A[k][0]
        if p == 0:
            return 'zeropivot', {'k': k}
        if p < 0:
            neg += 1
        row = A[k]
        for a in range(1, bw + 1):
            if k + a >= nc or row[a] == 0:
                continue
            f = row[a] / p
            Ai = A[k + a]
            for b in range(a, bw + 1):
                if k + b >= nc or row[b] == 0:
                    continue
                Ai[b - a] -= f * row[b]
    # exact kernel dimension of Bt (nr x nc)
    M = fmpq_mat(nr, nc)
    for (r, c), v in Bt.items():
        M[r, c] = fq(v)
    rank = M.rank()
    kdim = nc - rank
    info = {'neg': neg, 'ker': kdim, 'nc': nc, 'nr': nr, 'bw': bw}
    return ('certified' if neg == kdim else 'FAIL'), info

"""Exact certificate core for T = J1.(J2xJ3 + J4xJ5) on Inv(j1..j5), pairing (23|45), doubled spins tj.
Nonzero spectrum of T = +- singular values of the bipartite block B (rows: colour E, cols: colour O) of the
nonnegative gauge-fixed matrix W ([CL] Lemma 2; squared links exact rationals, closing_core.link2).
Rationalisation: every cycle of W has a rational product (plaquette identity), so there is a diagonal gauge
Theta (Theta^2 rational) with Bt := Theta_E B Theta_O rational. Then
   B^T B - eps  =  Theta_O^{-1} ( Bt^T Theta_E^{-2} Bt - eps Theta_O^2 ) Theta_O^{-1},
so inertia(B^T B - eps) = inertia(N_eps), N_eps := Bt^T Theta_E^{-2} Bt - eps Theta_O^2 (rational, symmetric)."""
from fractions import Fraction as Fr
from math import isqrt
from collections import deque
import numpy as np
from closing_core import sites, link2

def fsqrt(q):
    """exact sqrt of a rational square, else None"""
    q = Fr(q)
    if q < 0: return None
    a, b = isqrt(q.numerator), isqrt(q.denominator)
    if a * a == q.numerator and b * b == q.denominator: return Fr(a, b)
    return None

def build(tj):
    s, j2, j3, j4, j5 = tj
    S = sites(tj)
    if len(S) < 2: return None
    idx = {p: i for i, p in enumerate(S)}
    n23, m23 = j2 + j3 + 2, abs(j2 - j3)
    n45, m45 = j4 + j5 + 2, abs(j4 - j5)
    edges = {}
    for (b, c), i in idx.items():
        k = idx.get((b + 2, c))
        if k is not None:
            v = link2(n23, m23, s, c, b + 2)
            if v: edges[(i, k)] = v
        k = idx.get((b, c + 2))
        if k is not None:
            v = link2(n45, m45, s, b, c + 2)
            if v: edges[(i, k)] = v
    col = [((b + c) // 2) % 2 for (b, c) in S]
    return S, col, edges

def rationalise(n, edges):
    """theta2[i] (Fraction) and rational link values r_e = x_e theta_i theta_j (exact)."""
    adj = [[] for _ in range(n)]
    for (i, k), v in edges.items():
        adj[i].append((k, v)); adj[k].append((i, v))
    th2 = [None] * n
    for root in range(n):
        if th2[root] is not None: continue
        th2[root] = Fr(1); dq = deque([root])
        while dq:
            i = dq.popleft()
            for k, v in adj[i]:
                if th2[k] is None:
                    th2[k] = 1 / (v * th2[i]); dq.append(k)
    r = {}
    for (i, k), v in edges.items():
        val = fsqrt(v * th2[i] * th2[k])
        if val is None: raise ValueError("non-rational cycle product")
        r[(i, k)] = val
    return th2, r

def matrices(tj):
    """returns dict with rational Bt (rows E, cols O), theta2 lists, float B for checks; O = smaller colour."""
    out = build(tj)
    if out is None: return None
    S, col, edges = out
    n = len(S)
    th2, r = rationalise(n, edges)
    c0 = [i for i in range(n) if col[i] == 0]; c1 = [i for i in range(n) if col[i] == 1]
    O, E = (c0, c1) if len(c0) <= len(c1) else (c1, c0)
    pO = {i: a for a, i in enumerate(O)}; pE = {i: a for a, i in enumerate(E)}
    Bt = {}   # (rowE, colO) -> Fraction
    Bf = np.zeros((len(E), len(O)))
    for (i, k), v in r.items():
        if i in pE: e, o = pE[i], pO[k]
        else: e, o = pE[k], pO[i]
        Bt[(e, o)] = v
        Bf[e, o] = float(edges[(i, k)]) ** 0.5
    return dict(S=S, E=E, O=O, Bt=Bt, thE=[th2[i] for i in E], thO=[th2[i] for i in O], Bf=Bf, nE=len(E), nO=len(O))

# ---------------- exact inertia ----------------
import flint
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

def Cmat(D):
    """C = Bt^T Theta_E^{-2} Bt as dict-of-dicts over O indices (Fractions)."""
    rows = {}
    for (e, o), v in D['Bt'].items(): rows.setdefault(e, []).append((o, v))
    C = {}
    for e, lst in rows.items():
        w = 1 / D['thE'][e]
        for o, v in lst:
            for o2, v2 in lst:
                if o2 < o: continue
                C[(o, o2)] = C.get((o, o2), 0) + w * v * v2
    return C

def inertia_ldl(nO, C, thO, eps):
    """Exact inertia (n_-, n_0, n_+) of N = C - eps*diag(thO) via banded LDL^T with flint fmpq.
    Returns None if a zero pivot is met (caller falls back)."""
    if nO == 0: return (0, 0, 0)
    # ordering
    I = [i for (i, j) in C] + [j for (i, j) in C]; J = [j for (i, j) in C] + [i for (i, j) in C]
    G = csr_matrix((np.ones(len(I)), (I, J)), shape=(nO, nO))
    perm = list(reverse_cuthill_mckee(G, symmetric_mode=True)); pos = {p: a for a, p in enumerate(perm)}
    A = [dict() for _ in range(nO)]   # A[i][j] for j >= i in permuted indices
    for (i, j), v in C.items():
        a, b = sorted((pos[i], pos[j]))
        A[a][b] = A[a].get(b, flint.fmpq(0)) + flint.fmpq(v.numerator, v.denominator)
    epsq = flint.fmpq(eps.numerator, eps.denominator)
    for i in range(nO):
        t = thO[perm[i]]
        A[i][i] = A[i].get(i, flint.fmpq(0)) - epsq * flint.fmpq(t.numerator, t.denominator)
    neg = pos_ = 0
    for k in range(nO):
        piv = A[k].get(k, flint.fmpq(0))
        if piv == 0: return None
        if piv < 0: neg += 1
        else: pos_ += 1
        row = sorted((j, v) for j, v in A[k].items() if j > k and v != 0)
        for a, (j, vj) in enumerate(row):
            f = vj / piv
            Aj = A[j]
            for (l, vl) in row[a:]:
                Aj[l] = Aj.get(l, flint.fmpq(0)) - f * vl
        A[k] = None
    return (neg, 0, pos_)

def rank_exact(D):
    """exact rank of Bt over Q (flint)."""
    M = flint.fmpq_mat(D['nE'], D['nO'])
    for (e, o), v in D['Bt'].items(): M[e, o] = flint.fmpq(v.numerator, v.denominator)
    return M.rank()

"""Core builder for the fixed-j1 study.  All half-integers are passed DOUBLED (integers).

Pair P = (j_a, j_b):  n = j_a + j_b + 1,  m = |j_a - j_b|,  labels L in [m, n-1].
Region R (pairing pair1 | pair2): sites (b, c), b in [m1, n1-1], c in [m2, n2-1], |b-c| <= s <= b+c.
u = b + c, d = b - c.  Colour = parity of u  (equivalently of d + s).
Nonnegative gauge ([DC] Lemma 2):
  X-link (b-1,c)-(b,c): w^2 = h1(b)^2 F(d-1/2) G(u) / 16
  D-link (b,c-1)-(b,c): w^2 = h2(c)^2 F(d+1/2) G(u) / 16      ((b,c) the upper end)
  h_P(L)^2 = (n^2-L^2)(L^2-m^2)/(4L^2-1),  F(t) = (s+1/2)^2 - t^2,  G(u) = (u+s+1)(u-s).
q = 8 * (smallest nonzero singular value of the bipartite block).
"""
import numpy as np
import scipy.sparse as sp


def pair_nm(ja2, jb2):
    """doubled spins -> doubled (n, m)"""
    return ja2 + jb2 + 2, abs(ja2 - jb2)


def region(s2, n1, m1, n2, m2):
    """doubled inputs; returns arrays of doubled b, c of the region (sorted by u then d)."""
    B = np.arange(m1, n1 - 1, 2)
    C = np.arange(m2, n2 - 1, 2)
    if B.size == 0 or C.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    BB, CC = np.meshgrid(B, C, indexing='ij')
    BB = BB.ravel(); CC = CC.ravel()
    ok = (np.abs(BB - CC) <= s2) & (BB + CC >= s2) & ((BB + CC + s2) % 2 == 0)
    BB, CC = BB[ok], CC[ok]
    o = np.lexsort((BB - CC, BB + CC))
    return BB[o], CC[o]


def h2(nP, mP, L):
    """doubled n, m, L -> h^2 (float); vectorised"""
    n = nP / 2.0; m = mP / 2.0; l = np.asarray(L) / 2.0
    return (n * n - l * l) * (l * l - m * m) / (4 * l * l - 1)


def links(s2, n1, m1, n2, m2):
    Bd, Cd = region(s2, n1, m1, n2, m2)
    nS = Bd.size
    idx = {(int(b), int(c)): i for i, (b, c) in enumerate(zip(Bd, Cd))}
    s = s2 / 2.0
    lo, up, w2 = [], [], []
    for i, (b, c) in enumerate(zip(Bd, Cd)):
        b = int(b); c = int(c)
        u = (b + c) / 2.0; d = (b - c) / 2.0
        G = (u + s + 1) * (u - s)
        j = idx.get((b - 2, c))
        if j is not None:
            lo.append(j); up.append(i); w2.append(h2(n1, m1, b) * ((s + .5) ** 2 - (d - .5) ** 2) * G / 16)
        j = idx.get((b, c - 2))
        if j is not None:
            lo.append(j); up.append(i); w2.append(h2(n2, m2, c) * ((s + .5) ** 2 - (d + .5) ** 2) * G / 16)
    return Bd, Cd, np.array(lo, int), np.array(up, int), np.array(w2, float)


def bipartite(s2, n1, m1, n2, m2, colclass=None):
    """Block B: rows = class C', columns = class C.  Class of a site = (d + s) mod 2 = ((b-c+s2)/2) mod 2.
    colclass: parity (0/1) of (d+s) for the column class; default = the class with fewer sites."""
    Bd, Cd, lo, up, w2 = links(s2, n1, m1, n2, m2)
    nS = Bd.size
    cls = (((Bd - Cd + s2) // 2) % 2).astype(int)
    if colclass is None:
        colclass = 0 if (cls == 0).sum() <= (cls == 1).sum() else 1
    cols = np.where(cls == colclass)[0]; rows = np.where(cls != colclass)[0]
    pos = np.empty(nS, int); pos[cols] = np.arange(cols.size); pos[rows] = np.arange(rows.size)
    w = np.sqrt(np.maximum(w2, 0))
    r = np.where(cls[lo] != colclass, pos[lo], pos[up])
    cc = np.where(cls[lo] == colclass, pos[lo], pos[up])
    M = sp.csr_matrix((w, (r, cc)), shape=(rows.size, cols.size))
    return M, (Bd, Cd, rows, cols)


def svals(s2, n1, m1, n2, m2, colclass=None):
    M, info = bipartite(s2, n1, m1, n2, m2, colclass)
    if min(M.shape) == 0:
        return np.zeros(0), M.shape
    sv = np.linalg.svd(M.toarray(), compute_uv=False)
    return np.sort(sv), M.shape


def qmin_tuple(s2, n1, m1, n2, m2):
    """(q, dim ker T, dims) with q = 8 * smallest nonzero singular value"""
    sv, sh = svals(s2, n1, m1, n2, m2)
    if sv.size == 0:
        return None, sh[0] + sh[1], sh
    thr = 1e-10 * max(1.0, sv.max()) * (sh[0] + sh[1])
    nz = sv[sv > thr]
    ker = sh[0] + sh[1] - 2 * nz.size
    return (8 * nz.min() if nz.size else None), ker, sh


def pairings(t):
    """t = doubled (j2,j3,j4,j5); yields ordered pairings as (n1,m1,n2,m2) doubled, with a label"""
    a, b, c, d = t
    for lab, (p, q) in (('23|45', ((a, b), (c, d))), ('24|35', ((a, c), (b, d))), ('25|34', ((a, d), (b, c)))):
        n1, m1 = pair_nm(*p); n2, m2 = pair_nm(*q)
        yield lab, (n1, m1, n2, m2)
        yield lab + 'r', (n2, m2, n1, m1)

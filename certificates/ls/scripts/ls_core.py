"""Core builders for the large-spin study (float64, vectorised).  Spins are DOUBLED integers.

Tuple tj = (2j1, 2j2, 2j3, 2j4, 2j5); pairing (23|45).  Sites (b, c) of the region
  b in [|j2-j3|, j2+j3], c in [|j4-j5|, j4+j5], |b-c| <= s <= b+c, b+c+s integer  (s = j1).
Nonnegative gauge ([DC] Lemma 2): X-link (b-1,c)-(b,c) with weight^2 = h1(b)^2 F(d-1/2) G(u) / 16,
D-link (b,c-1)-(b,c) with weight^2 = h2(c)^2 F(d+1/2) G(u) / 16, (b,c) the upper end, where
h_P(L)^2 = (n_P^2-L^2)(L^2-m_P^2)/(4L^2-1), F(t) = (s+1/2)^2 - t^2, G(u) = (u+s+1)(u-s).
q = 8 * (smallest nonzero |eigenvalue| of T) = 8 * (smallest nonzero singular value of the bipartite block).
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def sites(tj):
    s2, a, b_, c_, d_ = tj
    B = np.arange(abs(a - b_), a + b_ + 1, 2, dtype=np.int64)
    C = np.arange(abs(c_ - d_), c_ + d_ + 1, 2, dtype=np.int64)
    BB, CC = np.meshgrid(B, C, indexing='ij')
    BB = BB.ravel(); CC = CC.ravel()
    ok = (np.abs(BB - CC) <= s2) & (s2 <= BB + CC) & ((BB + CC + s2) % 2 == 0)
    return BB[ok], CC[ok]


def links(tj):
    """Return (Bd, Cd, lo, up, w) : site arrays (doubled) and link list with weights (float64)."""
    s2, a, b_, c_, d_ = tj
    s = s2 / 2
    n1, m1 = (a + b_) / 2 + 1, abs(a - b_) / 2
    n2, m2 = (c_ + d_) / 2 + 1, abs(c_ - d_) / 2
    Bd, Cd = sites(tj)
    nS = len(Bd)
    if nS == 0:
        return Bd, Cd, np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    # index lookup via a dense grid
    b0, c0 = Bd.min(), Cd.min()
    nb, nc = (Bd.max() - b0) // 2 + 1, (Cd.max() - c0) // 2 + 1
    grid = -np.ones((nb + 1, nc + 1), dtype=np.int64)
    ib, ic = (Bd - b0) // 2, (Cd - c0) // 2
    grid[ib, ic] = np.arange(nS)
    b = Bd / 2.0; c = Cd / 2.0; u = b + c; d = b - c
    G = (u + s + 1) * (u - s)
    lo_all, up_all, w_all = [], [], []
    # X links: lower (b-1, c)
    jb = ib - 1
    okx = jb >= 0
    jx = np.where(okx, grid[np.maximum(jb, 0), ic], -1)
    okx &= jx >= 0
    hX2 = (n1 ** 2 - b ** 2) * (b ** 2 - m1 ** 2) / (4 * b ** 2 - 1)
    wx2 = hX2 * ((s + .5) ** 2 - (d - .5) ** 2) * G / 16
    lo_all.append(jx[okx]); up_all.append(np.arange(nS)[okx]); w_all.append(np.sqrt(np.maximum(wx2[okx], 0)))
    jc = ic - 1
    okd = jc >= 0
    jd = np.where(okd, grid[ib, np.maximum(jc, 0)], -1)
    okd &= jd >= 0
    hD2 = (n2 ** 2 - c ** 2) * (c ** 2 - m2 ** 2) / (4 * c ** 2 - 1)
    wd2 = hD2 * ((s + .5) ** 2 - (d + .5) ** 2) * G / 16
    lo_all.append(jd[okd]); up_all.append(np.arange(nS)[okd]); w_all.append(np.sqrt(np.maximum(wd2[okd], 0)))
    return Bd, Cd, np.concatenate(lo_all), np.concatenate(up_all), np.concatenate(w_all)


def bipartite(tj):
    """Sparse bipartite block Bmat (even-colour rows x odd-colour columns), colour = parity of u."""
    Bd, Cd, lo, up, w = links(tj)
    nS = len(Bd)
    col = ((Bd + Cd) // 2) % 2
    ev = np.where(col == 0)[0]; od = np.where(col == 1)[0]
    pos = np.empty(nS, dtype=np.int64); pos[ev] = np.arange(len(ev)); pos[od] = np.arange(len(od))
    r = np.where(col[lo] == 0, pos[lo], pos[up])
    cc = np.where(col[lo] == 0, pos[up], pos[lo])
    M = sp.csr_matrix((w, (r, cc)), shape=(len(ev), len(od)))
    return M, (Bd, Cd, ev, od)


def full_T(tj):
    Bd, Cd, lo, up, w = links(tj)
    n = len(Bd)
    return sp.csr_matrix((np.concatenate([w, w]), (np.concatenate([lo, up]), np.concatenate([up, lo]))), shape=(n, n))


def qmin(tj, dense_max=2500, want=1):
    """8 * smallest nonzero singular values of the bipartite block (first `want` of them), kernel dim of T."""
    M, _ = bipartite(tj)
    nr, nc = M.shape
    if nr == 0 or nc == 0:
        return None, nr + nc
    if max(nr, nc) <= dense_max:
        sv = np.linalg.svd(M.toarray(), compute_uv=False)
        thr = 1e-10 * max(sv.max(), 1.0) * (nr + nc)
        nz = np.sort(sv[sv > thr])
        ker = (nr + nc) - 2 * nz.size
        return (8 * nz[:want]) if nz.size else None, ker
    # sparse: smallest eigenvalues of the Gram matrix on the smaller side, shift-invert just below 0
    A = (M.T @ M) if nc <= nr else (M @ M.T)
    A = A.tocsc()
    smax = spl.eigsh(A, k=1, which='LA', return_eigenvectors=False)[0]
    k = 12
    while True:
        vals = spl.eigsh(A, k=min(k, A.shape[0] - 2), sigma=-1e-6 * smax, which='LM',
                         return_eigenvectors=False, tol=1e-13)
        vals = np.sort(np.abs(vals))
        thr = 1e-11 * smax
        nz = vals[vals > thr]
        kz = int((vals <= thr).sum())
        if nz.size >= want or k >= A.shape[0] - 2:
            ker_small = kz
            ker = (nr + nc) - 2 * (min(nr, nc) - ker_small)
            return 8 * np.sqrt(nz[:want]), ker
        k *= 3


def triangle_ok(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def dim_inv(tj):
    return len(sites(tj)[0])

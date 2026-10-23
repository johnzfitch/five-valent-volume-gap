"""Sparse W0 = P_R K P_R builder (pairing (a2 a3 | a4 a5), doubled spins) and smallest nonzero |eig|.
Float64.  Sites: b in [m1, n1-1], c in [m2, n2-1] (doubled), |b-c| <= S <= b+c.
Links: X (b-2,c)-(b,c) weight h1(b); D (b,c-2)-(b,c) weight h2(c); h^2 = (n^2-L^2)(L^2-m^2)/(4L^2-1)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def h2(n2x, m2x, L2x):
    n = n2x / 2.0; m = m2x / 2.0; L = L2x / 2.0
    return (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)


def build(S, a2, a3, a4, a5):
    n1 = a2 + a3 + 2; m1 = abs(a2 - a3)
    n2 = a4 + a5 + 2; m2 = abs(a4 - a5)
    sites = []
    for b in range(m1, n1 - 1, 2):
        lo = max(m2, abs(b - S), S - b)
        # parity: c ≡ m2 (mod 2)
        if (lo - m2) % 2:
            lo += 1
        hi = min(n2 - 2, b + S)
        for c in range(lo, hi + 1, 2):
            if abs(b - c) <= S <= b + c:
                sites.append((b, c))
    idx = {x: i for i, x in enumerate(sites)}
    I = []; J = []; V = []
    for (b, c), i in idx.items():
        k = idx.get((b - 2, c))
        if k is not None:
            w = np.sqrt(h2(n1, m1, b)); I += [i, k]; J += [k, i]; V += [w, w]
        k = idx.get((b, c - 2))
        if k is not None:
            w = np.sqrt(h2(n2, m2, c)); I += [i, k]; J += [k, i]; V += [w, w]
    N = len(sites)
    W = sp.csr_matrix((V, (I, J)), shape=(N, N))
    return sites, W


def colour(S, x):
    b, c = x
    return ((b + c - S) // 2) % 2


def lam_dense(S, legs, col=0):
    sites, W = build(S, *legs)
    cols = [i for i, x in enumerate(sites) if colour(S, x) == col]
    rows = [i for i, x in enumerate(sites) if colour(S, x) != col]
    B = W[rows][:, cols].toarray()
    sv = np.linalg.svd(B, compute_uv=False)
    nz = sv[sv > 1e-9 * max(1, sv.max())]
    return nz.min() if len(nz) else None


def lam_sparse(S, legs, k=6, sigma=1e-3, dense_max=3000):
    """smallest nonzero |eig W0|: dense SVD of the bipartite block when it is at most dense_max
    on a side, otherwise shift-invert Lanczos (both classes at once)."""
    sites, W = build(S, *legs)
    N = W.shape[0]
    cols = [i for i, x in enumerate(sites) if colour(S, x) == 0]
    rows = [i for i, x in enumerate(sites) if colour(S, x) == 1]
    if max(len(cols), len(rows)) <= dense_max:
        if not cols or not rows:
            return None, N
        B = W[rows][:, cols].toarray()
        sv = np.linalg.svd(B, compute_uv=False)
        nz = sv[sv > 1e-9 * max(1, sv.max())]
        return (nz.min() if len(nz) else None), N
    if N < 200:
        ev = np.abs(np.linalg.eigvalsh(W.toarray()))
        nz = ev[ev > 1e-9 * max(1, ev.max())]
        return nz.min() if len(nz) else None, N
    while True:
        try:
            vals = spl.eigsh(W.tocsc(), k=min(k, N - 2), sigma=sigma, which='LM', return_eigenvectors=False)
        except spl.ArpackNoConvergence:
            # fall back to a dense SVD of the bipartite block (exact path, slower)
            B = W[rows][:, cols].toarray()
            sv = np.linalg.svd(B, compute_uv=False)
            nz = sv[sv > 1e-9 * max(1, sv.max())]
            return (nz.min() if len(nz) else None), N
        a = np.abs(vals)
        nz = a[a > 1e-7]
        nzer = int((a <= 1e-7).sum())
        # need at least 2 nonzero values beyond the kernel to be sure the smallest nonzero is found
        if len(nz) >= 2 or k >= N - 2:
            return (nz.min() if len(nz) else None), N
        k = 2 * k + nzer


def pairings(a2, a3, a4, a5):
    return [(a2, a3, a4, a5), (a2, a4, a3, a5), (a2, a5, a3, a4)]

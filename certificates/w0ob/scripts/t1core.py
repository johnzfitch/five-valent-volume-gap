"""Core builders for T = J1.(J2xJ3 + J4xJ5) on Inv(j1..j5).

All spins are passed DOUBLED (integers): S = 2*j1, a2..a5 = 2*j2..2*j5.
q = 8 |eigenvalue of T|  (Brunnemann-Rideout normalisation of A = -8iT).

grid(...)   : closed-form recoupling grid (pairing 23|45), nonnegative gauge,
              links 1/4 h1(b) sqrt(F(d-1/2) G(u)) and 1/4 h2(c) sqrt(F(d+1/2) G(u)).
brute(...)  : full tensor-product construction (no recoupling theory), for validation.
"""
import numpy as np
from fractions import Fraction as Fr


def spin_mats(tj):
    """Hermitian spin matrices for spin tj/2, basis m = j, j-1, ..., -j."""
    j = tj / 2.0
    ms = j - np.arange(tj + 1)
    Jz = np.diag(ms)
    Jp = np.zeros((tj + 1, tj + 1))
    for k in range(1, tj + 1):
        m = ms[k]
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - m * (m + 1))
    Jx = (Jp + Jp.T) / 2
    Jy = (Jp - Jp.T) / (2j)
    return Jx, Jy, Jz


def brute(S, a2, a3, a4, a5):
    """Spectrum of T on the invariant subspace by full tensor product.
    Returns sorted eigenvalues of T (real), or empty array."""
    import scipy.sparse as sp
    spins = [S, a2, a3, a4, a5]
    dims = [t + 1 for t in spins]
    N = int(np.prod(dims))
    ops = []
    for i, t in enumerate(spins):
        mats = spin_mats(t)
        full = []
        for M in mats:
            op = sp.csr_matrix(np.array([[1.0 + 0j]]))
            for k in range(5):
                op = sp.kron(op, sp.csr_matrix(M.astype(complex)) if k == i else sp.identity(dims[k], format='csr'), format='csr')
            full.append(op)
        ops.append(full)
    Jtot = [sum(ops[i][a] for i in range(5)) for a in range(3)]
    # invariant vectors: total Jz = 0 and annihilated by J+ = Jx + i Jy
    jz = np.real(Jtot[2].diagonal())
    Z = np.where(np.abs(jz) < 1e-9)[0]
    if len(Z) == 0:
        return np.array([])
    Jp = (Jtot[0] + 1j * Jtot[1])[:, Z].toarray()
    u, sv, vh = np.linalg.svd(Jp, full_matrices=True)
    rank = int(np.sum(sv > 1e-8))
    kern = vh[rank:].conj().T            # in the Z-coordinates
    if kern.shape[1] == 0:
        return np.array([])
    inv = np.zeros((N, kern.shape[1]), dtype=complex)
    inv[Z, :] = kern

    def triple(a, b, c, X):
        Ja, Jb, Jc = ops[a], ops[b], ops[c]
        out = np.zeros(X.shape, dtype=complex)
        for (i, k, l, sgn) in [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]:
            out += sgn * (Ja[i] @ (Jb[k] @ (Jc[l] @ X)))
        return out
    Tr = inv.conj().T @ (triple(0, 1, 2, inv) + triple(0, 3, 4, inv))
    Tr = (Tr + Tr.conj().T) / 2
    return np.sort(np.linalg.eigvalsh(Tr))


def h2stark(n2x, m2x, L2x):
    """Squared Stark link h^2 at label L for H(n,m); doubled inputs (n2x=2n etc.). Exact Fraction."""
    n = Fr(n2x, 2); m = Fr(m2x, 2); L = Fr(L2x, 2)
    return (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)


def pair_params(ax, bx):
    """doubled (n, m) for a pair of doubled spins: n = ja+jb+1, m = |ja-jb|."""
    return ax + bx + 2, abs(ax - bx)


def sites(S, a2, a3, a4, a5):
    """Sites (b2x, c2x) of the region R, pairing (23|45), doubled."""
    n1, m1 = pair_params(a2, a3)
    n2, m2 = pair_params(a4, a5)
    out = []
    for b in range(m1, n1 - 1, 2):
        for c in range(m2, n2 - 1, 2):
            if abs(b - c) <= S <= b + c and (b + c + S) % 2 == 0:
                out.append((b, c))
    return out


def grid(S, a2, a3, a4, a5, exact=False):
    """Return (sites, links) where links = list of (i, k, w2) with w2 = squared link of T
    (as Fraction if exact else float)."""
    st = sites(S, a2, a3, a4, a5)
    idx = {x: i for i, x in enumerate(st)}
    n1, m1 = pair_params(a2, a3)
    n2, m2 = pair_params(a4, a5)
    s = Fr(S, 2)
    F = lambda t: (s + Fr(1, 2)) ** 2 - t * t
    G = lambda u: (u + s + 1) * (u - s)
    links = []
    for (b, c), i in idx.items():
        bb, cc = Fr(b, 2), Fr(c, 2)
        d, u = bb - cc, bb + cc
        if (b - 2, c) in idx:
            w2 = h2stark(n1, m1, b) * F(d - Fr(1, 2)) * G(u) / 16
            links.append((idx[(b - 2, c)], i, w2 if exact else float(w2)))
        if (b, c - 2) in idx:
            w2 = h2stark(n2, m2, c) * F(d + Fr(1, 2)) * G(u) / 16
            links.append((idx[(b, c - 2)], i, w2 if exact else float(w2)))
    return st, links


def Tmatrix(S, a2, a3, a4, a5):
    st, links = grid(S, a2, a3, a4, a5)
    n = len(st)
    W = np.zeros((n, n))
    for i, k, w2 in links:
        W[i, k] = W[k, i] = np.sqrt(w2)
    return st, W


def colour(S, site):
    """parity of u - s (doubled: (b+c-S)/2 mod 2)."""
    b, c = site
    return ((b + c - S) // 2) % 2


def block(S, a2, a3, a4, a5, col=0):
    """Bipartite block B: l2(class col) -> l2(other class) of T. Returns (B, cols, rows)."""
    st, W = Tmatrix(S, a2, a3, a4, a5)
    cols = [i for i, x in enumerate(st) if colour(S, x) == col]
    rows = [i for i, x in enumerate(st) if colour(S, x) != col]
    return W[np.ix_(rows, cols)], [st[i] for i in cols], [st[i] for i in rows]


def qmin(S, a2, a3, a4, a5, tol=1e-9):
    """Smallest nonzero q = 8|eig T| via SVD of the bipartite block; returns (qmin, dimker, dim)."""
    st, W = Tmatrix(S, a2, a3, a4, a5)
    n = len(st)
    if n == 0:
        return None, 0, 0
    cols = [i for i, x in enumerate(st) if colour(S, x) == 0]
    rows = [i for i, x in enumerate(st) if colour(S, x) == 1]
    B = W[np.ix_(rows, cols)]
    if B.size == 0:
        return None, n, n
    sv = np.linalg.svd(B, compute_uv=False)
    scale = max(1.0, sv.max())
    nz = sv[sv > tol * scale]
    dimker = n - 2 * len(nz)
    if len(nz) == 0:
        return None, dimker, n
    return 8 * nz.min(), dimker, n


def valid(S, a2, a3, a4, a5):
    return (S + a2 + a3 + a4 + a5) % 2 == 0

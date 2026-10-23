"""Core tools for the orbit-2 operator T = J1.(J2xJ3 + J4xJ5) on Inv(j1..j5).

Independent re-implementation (does not import from other folders).
Normalisation: BR operator A = -8i T for (alpha,beta)=(2,2); q = 8|eig T|.
Spins are passed DOUBLED (integers) unless stated.

Grid form (closed): basis |b,c>, b = |J2+J3|, c = |J4+J5|, triangle (j1,b,c).
X-link (b-1,c)-(b,c):  x^2 = (n23^2-b^2)(b^2-m23^2)((s+c+1)^2-b^2)(b^2-(s-c)^2) / (16(4b^2-1))
D-link (b,c-1)-(b,c):  d^2 = (n45^2-c^2)(c^2-m45^2)((s+b+1)^2-c^2)(c^2-(s-b)^2) / (16(4c^2-1))
with n23=j2+j3+1, m23=|j2-j3| etc. (undoubled).  These are the 4-valent closed forms of [VG]/[SL].
"""
import numpy as np
from fractions import Fraction


# ---------------- brute force on the tensor product ----------------
def spin_mats(tj):
    j = tj / 2.0
    d = tj + 1
    m = j - np.arange(d)
    Jz = np.diag(m)
    Jp = np.zeros((d, d))
    for k in range(1, d):
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    Jx = (Jp + Jp.T) / 2
    Jy = (Jp - Jp.T) / (2j)
    return [Jx.astype(complex), Jy, Jz.astype(complex)]


def leg_ops(tjs):
    """List over legs of [Jx,Jy,Jz] on the full tensor product."""
    dims = [t + 1 for t in tjs]
    out = []
    for i, t in enumerate(tjs):
        S = spin_mats(t)
        L = []
        for a in range(3):
            M = np.array([[1.0 + 0j]])
            for k, dk in enumerate(dims):
                M = np.kron(M, S[a] if k == i else np.eye(dk))
            L.append(M)
        out.append(L)
    return out


def inv_basis(ops):
    tot = [sum(o[a] for o in ops) for a in range(3)]
    C = sum(t @ t for t in tot)
    w, v = np.linalg.eigh(C)
    return v[:, np.abs(w) < 1e-8]


def triple(ops, i, j, k):
    A, B, C = ops[i], ops[j], ops[k]
    eps = [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]
    return sum(s * A[a] @ B[b] @ C[c] for a, b, c, s in eps)


# ---------------- closed-form grid ----------------
def sites(tj):
    s, j2, j3, j4, j5 = tj
    out = []
    for b in range(abs(j2 - j3), j2 + j3 + 1, 2):
        for c in range(abs(j4 - j5), j4 + j5 + 1, 2):
            if abs(b - c) <= s <= b + c and (b + c + s) % 2 == 0:
                out.append((b, c))
    return out


def link2(n, m, S, C, k):
    """Squared 4-valent link at doubled upper label k for pair (n,m) [doubled n=2(j+j'+1), m=2|j-j'|]
    against (S,C) doubled: ((S+C+2)^2-k^2)(k^2-(S-C)^2).  Returns Fraction (exact)."""
    k2 = k * k
    A2, B2 = S + C + 2, abs(S - C)
    num = (n * n - k2) * (k2 - m * m) * (A2 * A2 - k2) * (k2 - B2 * B2)
    if num <= 0:
        return Fraction(0)
    return Fraction(num, 256 * 16 * (k2 - 1))


def grid(tj, exact=False, alpha=1, beta=1):
    """Return sites and the symmetric nonnegative matrix W of T (gauge-fixed) -- |eig W| = |eig T|.
    If exact: dict of squared weights (Fractions)."""
    s, j2, j3, j4, j5 = tj
    S = sites(tj)
    idx = {p: i for i, p in enumerate(S)}
    n23, m23 = j2 + j3 + 2, abs(j2 - j3)
    n45, m45 = j4 + j5 + 2, abs(j4 - j5)
    W = np.zeros((len(S), len(S)))
    sq = {}
    for (b, c), i in idx.items():
        k = idx.get((b + 2, c))
        if k is not None and alpha:
            v = alpha * alpha * link2(n23, m23, s, c, b + 2)
            sq[(i, k)] = v
            W[i, k] = W[k, i] = np.sqrt(float(v))
        k = idx.get((b, c + 2))
        if k is not None and beta:
            v = beta * beta * link2(n45, m45, s, b, c + 2)
            sq[(i, k)] = v
            W[i, k] = W[k, i] = np.sqrt(float(v))
    if exact:
        return S, sq
    return S, W


def bip(S, W):
    ev = [i for i, (b, c) in enumerate(S) if ((b + c) // 2) % 2 == 0]
    od = [i for i, (b, c) in enumerate(S) if ((b + c) // 2) % 2 == 1]
    return W[np.ix_(ev, od)], ev, od


def lam_min(tj, tol=1e-10):
    """Smallest nonzero |eig T| (T = J1.(J2xJ3+J4xJ5)), via singular values of the bipartite block."""
    S, W = grid(tj)
    if len(S) < 2:
        return None
    B, ev, od = bip(S, W)
    if B.size == 0:
        return None
    sv = np.linalg.svd(B, compute_uv=False)
    smax = sv.max() if sv.size else 0
    nz = sv[sv > tol * max(smax, 1) * len(S)]
    return float(nz.min()) if nz.size else None


def valid(tj):
    return sum(tj) % 2 == 0 and all(t >= 1 for t in tj)

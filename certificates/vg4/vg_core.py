"""Core routines for the 4-valent volume-gap study.

Operator: Q = J1 . (J2 x J3) on Inv(j1 x j2 x j3 x j4), Hermitian generators,
[J_a, J_b] = i eps_abc J_c.  In the recoupling basis |k> (legs 1,2 coupled to k),
Q is tridiagonal with zero diagonal and |Q_{k-1,k}| = a_k with

  a_k^2 = [(j1+j2+1)^2-k^2][k^2-(j1-j2)^2][(j3+j4+1)^2-k^2][k^2-(j3-j4)^2] / (16(4k^2-1))

(Levy-Leblond & Levy-Nahas, J. Math. Phys. 6 (1965) 1372; Brunnemann-Thiemann
gr-qc/0405060 eq. (67), whose q_hat_123 = [(J1+J2)^2,(J2+J3)^2] has entries 4 a_k;
Bianchi-Haggard 1208.2228 App. A).  k runs from k_min = max(|j1-j2|,|j3-j4|) to
k_max = min(j1+j2, j3+j4); d = k_max - k_min + 1; the off-diagonals are
c_s = a_{k_min+s}, s = 1..d-1.

Spins are handled as doubled integers J = 2j throughout.
"""
from fractions import Fraction
import math
import numpy as np


# ---------------------------------------------------------------- exact data
def krange2(J1, J2, J3, J4):
    """Doubled k_min, k_max (2k) for pairing (12)(34)."""
    kmin2 = max(abs(J1 - J2), abs(J3 - J4))
    kmax2 = min(J1 + J2, J3 + J4)
    return kmin2, kmax2


def admissible(J1, J2, J3, J4):
    if min(J1, J2, J3, J4) < 0:
        return False
    if (J1 + J2 + J3 + J4) % 2:
        return False
    kmin2, kmax2 = krange2(J1, J2, J3, J4)
    return kmax2 >= kmin2


def dim(J1, J2, J3, J4):
    kmin2, kmax2 = krange2(J1, J2, J3, J4)
    return (kmax2 - kmin2) // 2 + 1 if kmax2 >= kmin2 else 0


def a2_exact(J1, J2, J3, J4, K):
    """a_k^2 as a Fraction, k = K/2 (doubled input)."""
    j1, j2, j3, j4, k = (Fraction(x, 2) for x in (J1, J2, J3, J4, K))
    num = (((j1 + j2 + 1) ** 2 - k * k) * (k * k - (j1 - j2) ** 2)
           * ((j3 + j4 + 1) ** 2 - k * k) * (k * k - (j3 - j4) ** 2))
    return num / (16 * (4 * k * k - 1))


def offdiag2_exact(J1, J2, J3, J4):
    """List of c_s^2 (Fractions), s = 1..d-1."""
    kmin2, kmax2 = krange2(J1, J2, J3, J4)
    return [a2_exact(J1, J2, J3, J4, K) for K in range(kmin2 + 2, kmax2 + 1, 2)]


def offdiag(J1, J2, J3, J4):
    return np.sqrt(np.array([float(x) for x in offdiag2_exact(J1, J2, J3, J4)], dtype=np.float64))


def q_matrix(J1, J2, J3, J4):
    """Dense Hermitian Q (phase convention: +i a_k above the diagonal)."""
    c = offdiag(J1, J2, J3, J4)
    d = len(c) + 1
    M = np.zeros((d, d), dtype=np.complex128)
    for s, v in enumerate(c):
        M[s, s + 1] = 1j * v
        M[s + 1, s] = -1j * v
    return M


# ------------------------------------------------- brute-force construction
def spin_mats(J):
    """(Jx, Jy, Jz) for spin j = J/2, basis m = j, j-1, ..., -j."""
    j = J / 2.0
    ms = np.array([j - i for i in range(J + 1)])
    Jz = np.diag(ms).astype(np.complex128)
    Jp = np.zeros((J + 1, J + 1), dtype=np.complex128)
    for i in range(1, J + 1):  # J+ |m> = sqrt(j(j+1)-m(m+1)) |m+1>
        m = ms[i]
        Jp[i - 1, i] = math.sqrt(j * (j + 1) - m * (m + 1))
    Jm = Jp.conj().T
    return (Jp + Jm) / 2, (Jp - Jm) / (2j), Jz


def bruteforce_Q_spectrum(Js):
    """Spectrum of J1.(J2xJ3) on the SU(2)-invariant subspace of the full tensor product."""
    dims = [J + 1 for J in Js]
    ops = []
    for leg, J in enumerate(Js):
        mats = spin_mats(J)
        emb = []
        for M in mats:
            T = np.array([[1.0 + 0j]])
            for l2, dd in enumerate(dims):
                T = np.kron(T, M if l2 == leg else np.eye(dd))
            emb.append(T)
        ops.append(emb)
    tot = [sum(ops[l][a] for l in range(4)) for a in range(3)]
    C = sum(t @ t for t in tot)
    w, V = np.linalg.eigh(C)
    P = V[:, np.abs(w) < 1e-8]
    eps = np.zeros((3, 3, 3))
    for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[a, b, c] = 1
        eps[a, c, b] = -1
    Q = np.zeros_like(C)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if eps[a, b, c]:
                    Q += eps[a, b, c] * ops[0][a] @ ops[1][b] @ ops[2][c]
    Qr = P.conj().T @ Q @ P
    return np.sort(np.linalg.eigvalsh((Qr + Qr.conj().T) / 2)), P.shape[1]


# ------------------------------------------------------- enumeration
def sorted_tuples(N, Nmin=1):
    """All doubled sorted tuples 1<=J1<=J2<=J3<=J4<=N, J4 in [Nmin, N], admissible, d>=1."""
    out = []
    for J4 in range(Nmin, N + 1):
        for J3 in range(1, J4 + 1):
            for J2 in range(1, J3 + 1):
                for J1 in range(1, J2 + 1):
                    if (J1 + J2 + J3 + J4) % 2 == 0 and J4 <= J1 + J2 + J3:
                        out.append((J1, J2, J3, J4))
    return out


def sorted_tuples_np(N, Nmin=1):
    """Vectorised version of sorted_tuples; returns int32 array (n,4)."""
    rows = []
    for J4 in range(Nmin, N + 1):
        J3, J2, J1 = np.meshgrid(np.arange(1, J4 + 1), np.arange(1, J4 + 1), np.arange(1, J4 + 1), indexing="ij")
        m = (J2 <= J3) & (J1 <= J2) & ((J1 + J2 + J3 + J4) % 2 == 0) & (J4 <= J1 + J2 + J3)
        r = np.stack([J1[m], J2[m], J3[m], np.full(m.sum(), J4)], axis=1).astype(np.int32)
        rows.append(r)
    return np.concatenate(rows, axis=0)


def offdiag2_batch(T):
    """Float64 c_s^2 for a batch of tuples of equal d.  T: (n,4) doubled ints.
    Returns (n, d-1) array (d = common dimension)."""
    J1, J2, J3, J4 = (T[:, i].astype(np.float64) / 2 for i in range(4))
    kmin = np.maximum(np.abs(J1 - J2), np.abs(J3 - J4))
    kmax = np.minimum(J1 + J2, J3 + J4)
    d = np.rint(kmax - kmin).astype(np.int64) + 1
    assert np.all(d == d[0])
    s = np.arange(1, d[0])[None, :]
    k = kmin[:, None] + s
    num = (((J1 + J2 + 1)[:, None] ** 2 - k * k) * (k * k - ((J1 - J2) ** 2)[:, None])
           * ((J3 + J4 + 1)[:, None] ** 2 - k * k) * (k * k - ((J3 - J4) ** 2)[:, None]))
    return num / (16.0 * (4.0 * k * k - 1.0))


# ------------------------------------------- batched Sturm bisection
def smallest_positive_eig(b2, iters=110, xp=np, device=None):
    """Smallest positive eigenvalue of zero-diagonal symmetric tridiagonal matrices.

    b2: (n, d-1) squared off-diagonals (all > 0), xp = numpy or torch.
    Sturm count N(x) = #{eigenvalues < x} via LDL^T pivots q_1 = -x,
    q_i = -x - b2_{i-1}/q_{i-1}; the smallest positive eigenvalue is the
    (ceil(d/2)+1)-th smallest.  Bisection on [0, 2 max b] in float64.
    """
    n, m = b2.shape
    d = m + 1
    n0 = (d + 1) // 2
    if xp is np:
        B2 = b2
        hi = 2.0 * np.sqrt(B2.max(axis=1))
        lo = np.zeros(n)
        tiny = np.finfo(np.float64).tiny * 1e10
        for _ in range(iters):
            x = 0.5 * (lo + hi)
            q = -x.copy()
            cnt = (q < 0).astype(np.int64)
            for i in range(m):
                q = np.where(np.abs(q) < tiny, -tiny, q)
                q = -x - B2[:, i] / q
                cnt += (q < 0)
            big = cnt > n0
            hi = np.where(big, x, hi)
            lo = np.where(big, lo, x)
        return 0.5 * (lo + hi)
    else:
        import torch
        B2 = b2 if isinstance(b2, torch.Tensor) else torch.as_tensor(b2, dtype=torch.float64, device=device)
        hi = 2.0 * torch.sqrt(B2.max(dim=1).values)
        lo = torch.zeros(n, dtype=torch.float64, device=B2.device)
        tiny = torch.finfo(torch.float64).tiny * 1e10
        cols = [B2[:, i].contiguous() for i in range(m)]
        for _ in range(iters):
            x = 0.5 * (lo + hi)
            q = -x
            cnt = (q < 0).to(torch.int32)
            for i in range(m):
                q = torch.where(q.abs() < tiny, torch.full_like(q, -tiny), q)
                q = -x - cols[i] / q
                cnt += (q < 0).to(torch.int32)
            big = cnt > n0
            hi = torch.where(big, x, hi)
            lo = torch.where(big, lo, x)
        return 0.5 * (lo + hi)


def lapack_smallest_positive(c):
    """Reference: scipy eigvalsh_tridiagonal (LAPACK stebz/stemr) on one chain."""
    from scipy.linalg import eigvalsh_tridiagonal
    d = len(c) + 1
    w = eigvalsh_tridiagonal(np.zeros(d), np.asarray(c, dtype=np.float64))
    n0 = (d + 1) // 2
    return np.sort(w)[n0]


# ----------------------------------------------- theorem quantities
def harmonic_sum(c):
    """S = sum_s 1/c_s."""
    return float(np.sum(1.0 / np.asarray(c, dtype=np.float64)))

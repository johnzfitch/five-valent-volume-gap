"""SU(2) building blocks, written from scratch for the paper.

Two independent constructions of the vertex operators:

  * brute force from the generators: Hermitian spin matrices on each leg, the
    tensor product, and the gauge-invariant subspace Inv(j_1,...,j_N) realised as
    the highest-weight vectors of weight j_N inside j_1 x ... x j_{N-1};
  * Racah's formula for 6j symbols in exact rational arithmetic, used for the
    4-valent recoupling matrix and its closed form.

Spins are passed doubled (integers), so 2j = 1 means spin 1/2.
Convention: [J_a, J_b] = i eps_abc J_c,  qhat_IJK = [(J_I+J_J)^2, (J_J+J_K)^2]
= -4i J_I.(J_J x J_K)  (Brunnemann-Rideout II, eq. (2.24)-(2.26)).
"""
from fractions import Fraction
from math import factorial, sqrt
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla


# ---------------------------------------------------------------- generators
def spin_matrices(tj):
    """Hermitian (Jx, Jy, Jz) for doubled spin tj, basis m = j, j-1, ..., -j."""
    j = tj / 2.0
    ms = np.array([j - k for k in range(tj + 1)])
    jp = np.zeros((tj + 1, tj + 1))
    for k in range(1, tj + 1):          # J+ |m> = sqrt(j(j+1)-m(m+1)) |m+1>
        m = ms[k]
        jp[k - 1, k] = sqrt(j * (j + 1) - m * (m + 1))
    jm = jp.T
    jx = (jp + jm) / 2.0
    jy = (jp - jm) / 2.0j
    jz = np.diag(ms)
    return jx.astype(complex), jy, jz.astype(complex), ms


class Leg:
    def __init__(self, spins2):
        self.spins2 = list(spins2)
        self.mats = [spin_matrices(t) for t in self.spins2]
        self.dims = [t + 1 for t in self.spins2]

    def op(self, leg, comp):
        """Component comp (0,1,2) of J_leg on the tensor product, sparse."""
        out = None
        for i, d in enumerate(self.dims):
            m = sp.csr_matrix(self.mats[i][comp]) if i == leg else sp.identity(d, format='csr', dtype=complex)
            out = m if out is None else sp.kron(out, m, format='csr')
        return out

    def mtot(self):
        tot = np.zeros(1)
        for (_, _, _, ms) in self.mats:
            tot = (tot[:, None] + ms[None, :]).ravel()
        return tot


def invariant_space(spins2):
    """Orthonormal basis of Inv(j_1..j_N), realised in j_1 x ... x j_{N-1} as the
    vectors of weight M = j_N annihilated by the total raising operator.
    Returns (legs, basis matrix V on the full product of the first N-1 legs)."""
    legs = Leg(spins2[:-1])
    target = spins2[-1] / 2.0
    mt = legs.mtot()
    idx = np.where(np.abs(mt - target) < 1e-9)[0]
    up = np.where(np.abs(mt - target - 1) < 1e-9)[0]
    n = len(legs.dims)
    jp = sum(legs.op(i, 0) + 1j * legs.op(i, 1) for i in range(n))
    if len(idx) == 0:
        return legs, np.zeros((int(np.prod(legs.dims)), 0))
    block = jp[up][:, idx].toarray() if len(up) else np.zeros((0, len(idx)))
    if block.shape[0] == 0:
        ker = np.eye(len(idx))
    else:
        ker = sla.null_space(block, rcond=1e-10)
    V = np.zeros((int(np.prod(legs.dims)), ker.shape[1]), dtype=complex)
    V[idx, :] = ker
    return legs, V


def triple(legs, V, I, J, K):
    """Matrix of J_I.(J_J x J_K) on Inv, legs indexed from 0 among the first N-1
    legs; the last leg N-1 is eliminated by J_last = -(J_0+...+J_{N-2}) acting on
    the highest-weight vectors (invariance)."""
    n = len(legs.dims)

    def comp(leg, c):
        if leg == n:  # the eliminated leg
            return -sum(legs.op(i, c) for i in range(n))
        return legs.op(leg, c)

    eps = [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]
    M = None
    for a, b, c, s in eps:
        t = comp(I, a) @ (comp(J, b) @ (comp(K, c) @ V))
        M = s * t if M is None else M + s * t
    return V.conj().T @ M


def qhat(legs, V, I, J, K):
    """BR's qhat_IJK = -4i J_I.(J_J x J_K) on Inv (anti-Hermitian)."""
    return -4j * triple(legs, V, I, J, K)


def sigma_operator(spins2, sigma, triples):
    """A = sum_T sigma_T qhat_T on Inv(spins2). triples: list of (I,J,K), 0-based,
    each < N (legs up to N-1 inclusive; the last leg is eliminated)."""
    legs, V = invariant_space(spins2)
    A = np.zeros((V.shape[1], V.shape[1]), dtype=complex)
    for s, (I, J, K) in zip(sigma, triples):
        if s != 0:
            A = A + s * qhat(legs, V, I, J, K)
    return A


def nonzero_moduli(A, rel=1e-10):
    """Moduli of the nonzero eigenvalues of an anti-Hermitian A (via iA Hermitian)."""
    if A.shape[0] == 0:
        return np.array([])
    ev = np.linalg.eigvalsh(1j * A)
    sc = max(1.0, np.max(np.abs(ev)))
    return np.sort(np.abs(ev[np.abs(ev) > rel * sc]))


def triples_sorted(N):
    """BR order of the sigma triples I<J<K<N (1-based labels in output)."""
    return [t for t in itertools.combinations(range(1, N), 3)]


# ---------------------------------------------------------------- Racah 6j
def _tri(a, b, c):
    """Doubled-spin triangle with integer sum."""
    return (a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b


def _delta2(a, b, c):
    """Delta(abc)^2 as a Fraction, doubled arguments."""
    return Fraction(factorial((a + b - c) // 2) * factorial((a - b + c) // 2) * factorial((-a + b + c) // 2),
                    factorial((a + b + c) // 2 + 1))


def sixj(a, b, c, d, e, f):
    """Racah's formula. Returns (sign, value^2) with value^2 a Fraction; doubled
    arguments {a b c; d e f}. sign is 0 for a vanishing symbol."""
    if not (_tri(a, b, c) and _tri(a, e, f) and _tri(d, b, f) and _tri(d, e, c)):
        return 0, Fraction(0)
    pre = _delta2(a, b, c) * _delta2(a, e, f) * _delta2(d, b, f) * _delta2(d, e, c)
    t1 = (a + b + c) // 2; t2 = (a + e + f) // 2; t3 = (d + b + f) // 2; t4 = (d + e + c) // 2
    p1 = (a + b + d + e) // 2; p2 = (a + c + d + f) // 2; p3 = (b + c + e + f) // 2
    s = 0
    for z in range(max(t1, t2, t3, t4), min(p1, p2, p3) + 1):
        s += (-1) ** z * Fraction(factorial(z + 1),
                                   factorial(z - t1) * factorial(z - t2) * factorial(z - t3) * factorial(z - t4)
                                   * factorial(p1 - z) * factorial(p2 - z) * factorial(p3 - z))
    if s == 0:
        return 0, Fraction(0)
    return (1 if s > 0 else -1), s * s * pre


def sixj_float(*args):
    sg, v2 = sixj(*args)
    return sg * sqrt(float(v2))


def recoupling_4valent(j):
    """Orthogonal matrix <(12)k,(34)|(23)l,(14)> for doubled spins j=(j1..j4) in the
    standard convention U_kl = (-1)^{j1+j2+j3+j4} sqrt((2k+1)(2l+1)) {j1 j2 k; j3 j4 l}.
    Returns (ks, ls, U)."""
    j1, j2, j3, j4 = j
    ks = [k for k in range(max(abs(j1 - j2), abs(j3 - j4)), min(j1 + j2, j3 + j4) + 1, 2)]
    ls = [l for l in range(max(abs(j2 - j3), abs(j1 - j4)), min(j2 + j3, j1 + j4) + 1, 2)]
    ph = (-1) ** ((j1 + j2 + j3 + j4) // 2)
    U = np.array([[ph * sqrt((k + 1) * (l + 1)) * sixj_float(j1, j2, k, j3, j4, l) for l in ls] for k in ks])
    return ks, ls, U


def Q4_racah(j):
    """Q = J1.(J2 x J3) = (i/4)[J12^2, J23^2] on Inv(j1..j4) in the basis |k = j12>."""
    ks, ls, U = recoupling_4valent(j)
    K2 = np.diag([(k / 2) * (k / 2 + 1) for k in ks])
    L2 = U @ np.diag([(l / 2) * (l / 2 + 1) for l in ls]) @ U.T
    return 0.25j * (K2 @ L2 - L2 @ K2), ks


def a2_closed(j, k2):
    """Squared modulus of <k-1|Q|k> (doubled spins, doubled k) from the closed form
    a_k^2 = [(A1^2-k^2)(k^2-B1^2)(A2^2-k^2)(k^2-B2^2)] / (16 (4k^2-1)),
    A1 = j1+j2+1, B1 = |j1-j2|, A2 = j3+j4+1, B2 = |j3-j4|. Exact Fraction."""
    j1, j2, j3, j4 = [Fraction(x, 2) for x in j]
    k = Fraction(k2, 2)
    A1, B1, A2, B2 = j1 + j2 + 1, abs(j1 - j2), j3 + j4 + 1, abs(j3 - j4)
    return (A1 ** 2 - k ** 2) * (k ** 2 - B1 ** 2) * (A2 ** 2 - k ** 2) * (k ** 2 - B2 ** 2) / (16 * (4 * k ** 2 - 1))


def chain_4valent(j):
    """Closed-form zero-diagonal Jacobi matrix with |Q| links (float), doubled spins."""
    j1, j2, j3, j4 = j
    kmin = max(abs(j1 - j2), abs(j3 - j4)); kmax = min(j1 + j2, j3 + j4)
    ks = list(range(kmin, kmax + 1, 2))
    n = len(ks)
    H = np.zeros((n, n))
    for i in range(1, n):
        a = sqrt(float(a2_closed(j, ks[i])))
        H[i - 1, i] = H[i, i - 1] = a
    return H


def stark(n, m):
    """Hydrogen Stark matrix H(n,m): zero-diagonal Jacobi on L = m..n-1,
    links h_L^2 = (n^2-L^2)(L^2-m^2)/(4L^2-1), L = m+1..n-1 (real n-m integer)."""
    N = int(round(n - m))
    H = np.zeros((N, N))
    for s in range(1, N):
        L = m + s
        H[s - 1, s] = H[s, s - 1] = sqrt((n * n - L * L) * (L * L - m * m) / (4 * L * L - 1))
    return H

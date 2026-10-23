"""Normal-form grid operator for the 5-valent AL vertex.

After relabelling (orbits.py, chain_forms.py) every non-coplanar 5-valent vertex with nonzero operator has
  A = -4i [ alpha * J1.(J2 x J3) + beta * J1.(J4 x J5) ],  (alpha, beta) in {(2,0), (2,2), (4,2)}
(leg 1 = the distinguished edge).  Basis |b, c>: b = spin of J2+J3, c = spin of J4+J5 (= spin of J1+J2+J3).
X = J1.(J2xJ3) is tridiagonal in b at fixed c (4-valent operator on Inv(j2,j3,j1,c)), D = J1.(J4xJ5)
tridiagonal in c at fixed b (4-valent on Inv(j4,j5,j1,b)).  With the gauge of grid_signs.py, |eig A| are the
|eig| of the real symmetric nonnegative matrix W with W[(b,c),(b+1,c)] = 4|alpha| a(j2,j3 | j1,c ; b+1),
W[(b,c),(b,c+1)] = 4|beta| a(j4,j5 | j1,b ; c+1), a = 4-valent closed form.
Spins are doubled integers. Sites: b in [|j2-j3|, j2+j3], c in [|j4-j5|, j4+j5], triangle (b, c, j1).
"""
import numpy as np
from fractions import Fraction


def a2(A1, B1, A2, B2, k):
    """4-valent squared modulus with doubled inputs: A1=2(j+j'+1) etc, k doubled. Returns float."""
    k2 = k * k
    num = (A1 * A1 - k2) * (k2 - B1 * B1) * (A2 * A2 - k2) * (k2 - B2 * B2)
    return num / (16.0 * 16.0 * (k2 - 1)) / 16.0 if num > 0 else 0.0


def a2_exact(A1, B1, A2, B2, k):
    k2 = k * k
    num = (A1 * A1 - k2) * (k2 - B1 * B1) * (A2 * A2 - k2) * (k2 - B2 * B2)
    if num <= 0:
        return Fraction(0)
    # doubled variables: (A^2-k^2) etc carry factor 4 each -> 4^4 ; (4k^2-1) -> (k2-1) with k doubled; /16
    return Fraction(num, 256 * 16 * (k2 - 1))


def sites(J):
    j1, j2, j3, j4, j5 = J
    out = []
    for b in range(abs(j2 - j3), j2 + j3 + 1, 2):
        for c in range(abs(j4 - j5), j4 + j5 + 1, 2):
            if abs(b - c) <= j1 <= b + c and (b + c + j1) % 2 == 0:
                out.append((b, c))
    return out


def W_matrix(J, alpha, beta, exact=False):
    """Nonnegative symmetric matrix whose |eigenvalues| equal |eig A| (A in BR normalisation).
    exact=True returns (sites, dict of squared weights as Fractions)."""
    j1, j2, j3, j4, j5 = J
    S = sites(J)
    idx = {s: i for i, s in enumerate(S)}
    n = len(S)
    W = np.zeros((n, n))
    sq = {}
    for (b, c), i in idx.items():
        k = idx.get((b + 2, c))
        if k is not None and alpha:
            # X-link: 4-valent on Inv(j2,j3,j1,c), pairing (23)(1c), k = b+1 (doubled b+2)
            A1, B1, A2, B2 = j2 + j3 + 2, abs(j2 - j3), j1 + c + 2, abs(j1 - c)
            v = (a2_exact if exact else a2)(A1, B1, A2, B2, b + 2)
            sq[(i, k)] = 16 * alpha * alpha * v
            W[i, k] = W[k, i] = 4 * abs(alpha) * np.sqrt(float(v))
        k = idx.get((b, c + 2))
        if k is not None and beta:
            A1, B1, A2, B2 = j4 + j5 + 2, abs(j4 - j5), j1 + b + 2, abs(j1 - b)
            v = (a2_exact if exact else a2)(A1, B1, A2, B2, c + 2)
            sq[(i, k)] = 16 * beta * beta * v
            W[i, k] = W[k, i] = 4 * abs(beta) * np.sqrt(float(v))
    if exact:
        return S, sq
    return S, W


def spectrum(J, alpha, beta):
    S, W = W_matrix(J, alpha, beta)
    if not S:
        return np.zeros(0)
    return np.sort(np.abs(np.linalg.eigvalsh(W)))

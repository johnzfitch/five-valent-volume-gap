"""Check of the chain reduction (Proposition 1 of THEOREM.md).

For a 5-valent vertex with spins (1/2, j2, j3, j4, j5) and sigma = (alpha, beta, beta, 0)
(BR ordering sigma(123), sigma(124), sigma(134), sigma(234)), the BR operator
  A = alpha*qhat_123 + beta*(qhat_124 + qhat_134)
is claimed to be unitarily equivalent (after multiplying by i) to a zero-diagonal Jacobi matrix on
sites (b, c), c = b -+ 1/2, with
  D-link (b,b-1/2)--(b,b+1/2):   |4 beta|  * sqrt(F(j4,j5; b+1/2))
  X-link (c-1/2,c)--(c+1/2,c):   |4 alpha| * sqrt(F(j2,j3; c+1/2))
  F(p,q;k) = ((p+q+1)^2 - k^2)(k^2 - (p-q)^2)/16 .
This script compares the two spectra for all tuples with 2j <= JMAX and several (alpha,beta),
in float64, and prints the maximum discrepancy.  Usage: python chain_check.py [JMAX]
"""
import sys
import itertools
import numpy as np
from fractions import Fraction
from hv_core import Vertex, sigma_vec_BR, spectrum_abs


def F2(P, Qd, K):
    """F(p,q;k) with doubled args, exact Fraction."""
    p, q, k = Fraction(P, 2), Fraction(Qd, 2), Fraction(K, 2)
    return ((p + q + 1) ** 2 - k * k) * (k * k - (p - q) ** 2) / 16


def chain_sites(Jd):
    """Sites (b,c) doubled, ordered along the path. Jd = (1, J2, J3, J4, J5)."""
    _, J2, J3, J4, J5 = Jd
    sites = []
    for B in range(abs(J2 - J3), J2 + J3 + 1, 2):
        for C in (B - 1, B + 1):
            if C < 0:
                continue
            if abs(J4 - J5) <= C <= J4 + J5 and (C + J4 + J5) % 2 == 0:
                sites.append((B, C))
    sites.sort(key=lambda s: (s[0] + s[1], s[0]))
    return sites


def chain_couplings_sq(Jd, alpha, beta):
    """Exact squared couplings c_s^2 (Fractions) along the path, and the site list."""
    _, J2, J3, J4, J5 = Jd
    S = chain_sites(Jd)
    c2 = []
    for (B1, C1), (B2, C2) in zip(S[:-1], S[1:]):
        if B1 == B2:            # D-link, k = b + 1/2 = max(c)
            K = max(C1, C2)
            c2.append(Fraction(16 * beta * beta) * F2(J4, J5, K))
        elif C1 == C2:          # X-link, k = c + 1/2 = max(b)
            K = max(B1, B2)
            c2.append(Fraction(16 * alpha * alpha) * F2(J2, J3, K))
        else:
            raise RuntimeError("path broken", (B1, C1), (B2, C2))
    return S, c2


def chain_spectrum(c2):
    d = len(c2) + 1
    M = np.zeros((d, d))
    for s, v in enumerate(c2):
        M[s, s + 1] = M[s + 1, s] = np.sqrt(float(v))
    return np.sort(np.abs(np.linalg.eigvalsh(M)))


def main(JMAX=8):
    worst = 0.0
    count = 0
    for J2, J3, J4, J5 in itertools.product(range(1, JMAX + 1), repeat=4):
        Jd = (1, J2, J3, J4, J5)
        if sum(Jd) % 2:
            continue
        v = Vertex(Jd)
        if v.dim == 0:
            continue
        for (al, be) in [(4, 2), (4, -2), (2, 2), (2, -4), (2, 4), (4, 4), (3, 1)]:
            A = v.Qsigma(sigma_vec_BR((al, be, be, 0), 5))
            e1 = spectrum_abs(A)
            S, c2 = chain_couplings_sq(Jd, al, be)
            assert len(S) == v.dim, (Jd, len(S), v.dim)
            assert all(x > 0 for x in c2), (Jd, c2)
            e2 = chain_spectrum(c2) if len(S) > 1 else np.zeros(1)
            err = np.abs(np.sort(e1) - np.sort(e2)).max() / max(1.0, e1.max())
            worst = max(worst, err)
            count += 1
    print(f"chain reduction check: {count} (tuple, sigma) cases with 2j<= {JMAX}, "
          f"max relative spectral discrepancy {worst:.2e}; all couplings > 0, path connected")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)

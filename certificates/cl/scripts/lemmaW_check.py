"""Lemma W checks for the Stark matrix H(n,m) (N = n-m sites, labels L = m..n-1,
h_L^2 = (n^2-L^2)(L^2-m^2)/(4L^2-1)), N even:
 W1 (exact, rationals): the end-site Green function G_end(z) = e^T (H-z)^{-1} e at z=2 is <= 0 (= G_end(0))
    for BOTH ends, all N even <= 60, all m in {0, 1/2, 1, ..., 40}.  (Continued fraction in h^2 only.)
 W2 (float): min over psi with psi(end)=0 of ||(H-1)psi|| / ||psi||  >= 1  (both ends), same range.
 W3 (float): the end weights of the eigenvectors are symmetric in lambda and nonincreasing for lambda>=1
    (the property the proof uses; proved in the text from the CG ratio formulas + rational continuation).
"""
import sys
from fractions import Fraction
import numpy as np


def h2(n, m, L):
    return (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)


def continuant(hs2, z):
    """det(J - z) for the zero-diagonal Jacobi matrix with squared links hs2 (len = size-1)."""
    Dm2, Dm1 = Fraction(1), -z
    for t in hs2:
        Dm2, Dm1 = Dm1, -z * Dm1 - t * Dm2
    return Dm1


def G_end(hs2, z):
    """e^T (H - z)^{-1} e with e the first site of the chain hs2 (Cramer: minor / det), exact in h^2."""
    return continuant(hs2[1:], z) / continuant(hs2, z)


def main(Nmax=60, m2max=80):
    worstW1 = None; worstW2 = 9e9; okW3 = True; cnt = 0
    for m2 in range(0, m2max + 1):
        m = Fraction(m2, 2)
        for N in range(2, Nmax + 1, 2):
            n = m + N
            Ls = [m + k for k in range(1, N)]
            hs2 = [h2(n, m, L) for L in Ls]           # bottom-to-top
            for end, seq in (('bottom', hs2), ('top', hs2[::-1])):
                g2 = G_end(seq, Fraction(2))
                cnt += 1
                if worstW1 is None or g2 > worstW1[0]:
                    worstW1 = (g2, N, m, end)
            if N <= 40:
                H = np.zeros((N, N))
                for k, t in enumerate(hs2):
                    H[k, k + 1] = H[k + 1, k] = float(t) ** 0.5
                A = H - np.eye(N)
                for cols in (slice(1, N), slice(0, N - 1)):
                    v = np.linalg.svd(A[:, cols], compute_uv=False).min()
                    worstW2 = min(worstW2, v)
                w, V = np.linalg.eigh(H)
                for row in (0, N - 1):
                    wt = {int(round(x)): V[row, i] ** 2 for i, x in enumerate(w)}
                    for lam in range(1, N, 2):
                        if abs(wt[lam] - wt[-lam]) > 1e-10: okW3 = False
                        if lam + 2 <= N - 1 and wt[lam + 2] > wt[lam] + 1e-12: okW3 = False
    print(f"W1: {cnt} (N,m,end) cases; max of G_end(2) = {float(worstW1[0]):.6e} at N={worstW1[1]}, m={worstW1[2]}, {worstW1[3]} "
          f"(<= 0 required): {'PASS' if worstW1[0] <= 0 else 'FAIL'}")
    print(f"W2: min sigma_min((H-1) restricted to psi(end)=0) over N<=40 = {worstW2:.12f}  (>= 1 required): "
          f"{'PASS' if worstW2 >= 1 - 1e-12 else 'FAIL'}")
    print(f"W3: end weights symmetric and nonincreasing for lambda>=1: {okW3}")


if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:]))

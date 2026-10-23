"""Check of Theorem 2 (THEOREM.md): the two-parameter family of 5-valent spins (doubled)
   (1, M, M+2D, M+1, M+2D),  M >= 1, 2D >= 1   [j1 = 1/2, j2 = M/2, j3 = M/2 + D, j4 = (M+1)/2, j5 = M/2 + D]
with sigma = (4, 2, 2, 0).  Claimed: the chain has 2(M+1) sites, starts and ends with a D-link,
  1/E_{M+1,1} = P := 2^(2-M) (M+1) sqrt(D (M+D+1))   (exact),   and   3P/8 <= q_min <= P.
For each (M, 2D) in the grid the script checks (i) path structure and link types, (ii) the exact identity
for P^2 with Fractions, (iii) q_min/P in [3/8, 1] with 40-digit arithmetic, and (iv) for M <= 6, 2D <= 6 the
full recoupling-basis spectrum against the chain (float64).
Usage: python general_family_check.py [Mmax] [twoDmax]
"""
import sys
from fractions import Fraction
import numpy as np
import mpmath as mp
from chain_check import chain_couplings_sq, chain_spectrum, chain_sites
from hv_core import Vertex, sigma_vec_BR, spectrum_abs

mp.mp.dps = 40


def main(Mmax=30, D2max=30):
    lo, hi = mp.inf, mp.mpf(0)
    n = 0
    worst_full = 0.0
    for M in range(1, Mmax + 1):
        for D2 in range(1, D2max + 1):
            Jd = (1, M, M + D2, M + 1, M + D2)
            S, c2 = chain_couplings_sq(Jd, 4, 2)
            assert len(S) == 2 * (M + 1)
            # link types: D-links are those with equal b
            types = ["D" if a[0] == b[0] else "X" for a, b in zip(S[:-1], S[1:])]
            assert types[0] == "D" and types[-1] == "D" and all(t != u for t, u in zip(types, types[1:]))
            odd, even = c2[0::2], c2[1::2]
            E2 = Fraction(1)
            for x in even:
                E2 *= x
            for x in odd:
                E2 /= x
            D = Fraction(D2, 2)
            P2 = Fraction(4) ** (2 - M) * (M + 1) ** 2 * D * (M + D + 1) if M <= 2 else \
                Fraction((M + 1) ** 2 * D * (M + D + 1), 4 ** (M - 2))
            assert 1 / E2 == P2, (M, D2)
            B = mp.zeros(M + 1, M + 1)
            for i in range(M + 1):
                B[i, i] = mp.sqrt(mp.mpf(odd[i].numerator) / odd[i].denominator)
                if i < M:
                    B[i + 1, i] = mp.sqrt(mp.mpf(even[i].numerator) / even[i].denominator)
            q = min(mp.svd_r(B, compute_uv=False))
            P = mp.sqrt(mp.mpf(P2.numerator) / P2.denominator)
            r = q / P
            lo, hi = min(lo, r), max(hi, r)
            n += 1
            if M <= 6 and D2 <= 6:
                e1 = spectrum_abs(Vertex(Jd).Qsigma(sigma_vec_BR((4, 2, 2, 0), 5)))
                e2 = chain_spectrum(c2)
                worst_full = max(worst_full, np.abs(e1 - e2).max() / e1.max())
    print(f"{n} (M, 2D) pairs, M<= {Mmax}, 2D<= {D2max}: path/link structure OK, exact identity 1/E^2 = P^2 OK")
    print(f"   q_min/P in [{mp.nstr(lo, 8)}, {mp.nstr(hi, 8)}]  (Theorem 2: [3/8, 1])")
    print(f"   full recoupling operator vs chain (M,2D<=6): max relative discrepancy {worst_full:.1e}")
    assert lo >= mp.mpf(3) / 8 and hi <= 1


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a)

"""Checks for Theorem 1 (THEOREM.md): the 5-valent family j = (1/2, j-1/2, j, j, j), sigma = (4,2,2,0).

1. Realizability: explicit integer tangent vectors, exact determinants, eps and sigma.
2. End-to-end: full tensor-product AL operator sum_{all triples} eps(IJK) qhat_IJK on Inv, for 2j = 2,3,4,
   versus the closed-form Jacobi chain.
3. Recoupling-basis operator (hv_core, sigma from eps) versus chain for 2j = 2..40 (float64).
4. Exact identity  E_{N,1}^{-2} = P^2 = 16 N^2 (2N+1) / 4^N  (N = 2j) for N = 2..400 (Fractions).
5. q_min to 60 digits for N = 2..100: bounds 3/8 <= q_min/P <= 1 and the limit kappa.
Usage: python family_check.py
"""
import itertools
from fractions import Fraction
import numpy as np
import mpmath as mp
from hv_core import Vertex, brute_Qsigma_spectrum, sigma_from_eps, spectrum_abs
from chain_check import chain_couplings_sq, chain_spectrum

mp.mp.dps = 60
VECS = [(1, 3, 1), (0, -2, 0), (-2, 0, -3), (-3, 1, 0), (-3, 1, 2)]   # tangents of e1..e5


def idet(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def eps_exact():
    sg = lambda x: (x > 0) - (x < 0)
    return {t: sg(idet(*[VECS[i] for i in t])) for t in itertools.combinations(range(5), 3)}


def main():
    eps = eps_exact()
    dets = {t: idet(*[VECS[i] for i in t]) for t in itertools.combinations(range(5), 3)}
    print("1. tangents:", VECS)
    print("   determinants (all nonzero => non-coplanar):", {''.join(str(i + 1) for i in t): d for t, d in dets.items()})
    sig = sigma_from_eps(eps, 5)
    print("   sigma(123,124,134,234) =", tuple(sig[t] for t in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]))
    assert tuple(sig[t] for t in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]) == (4, 2, 2, 0)

    print("2. full tensor product, all 10 eps-weighted triples, vs chain:")
    for N in (2, 3, 4):
        Jd = (1, N - 1, N, N, N)
        e_bf, dimb = brute_Qsigma_spectrum(Jd, eps)
        _, c2 = chain_couplings_sq(Jd, 4, 2)
        e_ch = chain_spectrum(c2)
        err = np.abs(np.sort(np.abs(e_bf)) - e_ch).max()
        print(f"   2j={N}: dim Inv={dimb}, chain sites={len(c2) + 1}, max|diff|={err:.2e}, "
              f"min|q| brute={np.abs(e_bf).min():.12f} chain={e_ch.min():.12f}")
        assert err < 1e-9

    print("3. recoupling basis (sigma from eps) vs chain, 2j = 2..40:")
    worst = 0
    for N in range(2, 41):
        Jd = (1, N - 1, N, N, N)
        v = Vertex(Jd)
        e1 = spectrum_abs(v.Qsigma(sig))
        _, c2 = chain_couplings_sq(Jd, 4, 2)
        e2 = chain_spectrum(c2)
        worst = max(worst, np.abs(e1 - e2).max() / e1.max())
    print(f"   max relative discrepancy {worst:.2e}")

    print("4. exact product identity:")
    ok = True
    for N in range(2, 401):
        _, c2 = chain_couplings_sq((1, N - 1, N, N, N), 4, 2)
        odd, even = c2[0::2], c2[1::2]
        assert len(odd) == N and len(even) == N - 1
        E2 = Fraction(1)
        for x in even:
            E2 *= x
        for x in odd:
            E2 /= x
        ok &= (1 / E2 == Fraction(16 * N * N * (2 * N + 1), 4 ** N))
    print("   E_{N,1}^{-2} == 16 N^2 (2N+1)/4^N for all N=2..400:", ok)
    assert ok

    print("5. q_min (60 digits) vs P = 4N sqrt(2N+1) 2^-N:")
    kappa = 1 / mp.sqrt(mp.mpf(16) / 3 * mp.log(mp.mpf(4) / 3))
    print("   kappa = (16/3 ln(4/3))^(-1/2) =", mp.nstr(kappa, 15))
    lo, hi = mp.inf, 0
    for N in list(range(2, 41)) + [50, 60, 80, 100]:
        _, c2 = chain_couplings_sq((1, N - 1, N, N, N), 4, 2)
        odd, even = c2[0::2], c2[1::2]
        B = mp.zeros(N, N)
        for i in range(N):
            B[i, i] = mp.sqrt(mp.mpf(odd[i].numerator) / odd[i].denominator)
            if i + 1 < N:
                B[i + 1, i] = mp.sqrt(mp.mpf(even[i].numerator) / even[i].denominator)
        q = min(mp.svd_r(B, compute_uv=False))
        P = 4 * N * mp.sqrt(2 * N + 1) / mp.mpf(2) ** N
        r = q / P
        lo, hi = min(lo, r), max(hi, r)
        if N in (2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 80, 100):
            print(f"   2j={N:3d}  q_min={mp.nstr(q, 15):>22}  lambda_V=sqrt(q)={mp.nstr(mp.sqrt(q), 10):>16}"
                  f"  q/P={mp.nstr(r, 12)}")
    print(f"   min q/P = {mp.nstr(lo, 10)}, max q/P = {mp.nstr(hi, 10)} (Theorem: [0.375, 1])")
    assert lo >= mp.mpf(3) / 8 and hi <= 1


if __name__ == "__main__":
    main()

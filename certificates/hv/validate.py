"""Validation of hv_core (recoupling-basis AL operator) against
 (a) a brute-force tensor-product construction on the SU(2)-invariant subspace (random sigma, valence 5, 6);
 (b) the eps -> sigma gauge reduction (BR 0706.0382 eq. (4.12)) using eps from random tangent vectors,
     brute force over all N-leg triples;
 (c) the published value lambda_V = 0.000597922 of BR 0706.0382 sec. 6.5.3 (7-vertex, spins
     2j = (2,2,2,2,2,3,5), sigma = (2,2,2,-4,0,0,0,0,-2,-2,2,2,4,-2,4,4,0,0,0,-2));
 (d) identities: J1.J2 diagonal, Casimir sum rule.
Usage: python validate.py
"""
import numpy as np
from hv_core import (Vertex, triples, brute_Qsigma_spectrum, sigma_from_eps, eps_from_vectors,
                     sigma_vec_BR, spectrum_abs)


def main():
    rng = np.random.default_rng(1)
    worst = 0.0
    for Jd in [(1, 1, 2, 2, 2), (2, 2, 2, 2, 2), (1, 2, 1, 2, 2), (1, 3, 2, 2, 2), (2, 2, 3, 3, 4),
               (1, 1, 1, 1, 1, 1), (2, 1, 1, 2, 1, 1), (1, 1, 1, 1, 2, 2)]:
        v = Vertex(Jd)
        N = len(Jd)
        n = N - 1
        d12 = np.diag([((c[1] * (c[1] + 2)) - Jd[0] * (Jd[0] + 2) - Jd[1] * (Jd[1] + 2)) / 8 for c in v.basis])
        e_d = np.abs(v.dot(0, 1) - d12).max()
        S = sum(2 * v.dot(I, J) for I in range(n) for J in range(I + 1, n)) + \
            sum(Jd[I] * (Jd[I] + 2) / 4 for I in range(n)) * np.eye(v.dim)
        e_c = np.abs(S - Jd[-1] * (Jd[-1] + 2) / 4 * np.eye(v.dim)).max()
        for trial in range(3):
            sig = {t: int(rng.integers(-2, 3)) * 2 for t in triples(N - 1)}
            e1 = spectrum_abs(v.Qsigma(sig))
            e2, dimb = brute_Qsigma_spectrum(Jd, sig)
            assert dimb == v.dim
            worst = max(worst, np.abs(np.sort(np.abs(e2)) - e1).max() / max(1, e1.max()))
            V = rng.normal(size=(N, 3))
            eps = eps_from_vectors(V)
            e1 = spectrum_abs(v.Qsigma(sigma_from_eps(eps, N)))
            e2, _ = brute_Qsigma_spectrum(Jd, eps)
            worst = max(worst, np.abs(np.sort(np.abs(e2)) - e1).max() / max(1, e1.max()))
        print(f"spins(2j)={Jd}: dim {v.dim}; J1.J2 diag err {e_d:.1e}; Casimir err {e_c:.1e}")
    print(f"(a,b) brute force vs recoupling, random sigma and eps->sigma: max relative error {worst:.2e}")
    sv = (2, 2, 2, -4, 0, 0, 0, 0, -2, -2, 2, 2, 4, -2, 4, 4, 0, 0, 0, -2)
    v = Vertex((2, 2, 2, 2, 2, 3, 5))
    s = spectrum_abs(v.Qsigma(sigma_vec_BR(sv, 7)))
    q = s[s > 1e-9].min()
    print(f"(c) BR 7-vertex: dim {v.dim}, min nonzero q = {q:.10e}, lambda_V = sqrt(q) = {np.sqrt(q):.9f} "
          f"(BR printed 0.000597922)")


if __name__ == "__main__":
    main()

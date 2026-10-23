"""Checks for Theorem 3 (THEOREM.md): N-valent vertices (N = 6, 7, 8) whose AL operator on Inv equals the
5-valent family operator with leg 5 replaced by a bundle of edges sharing the tangent direction of e5
(merging lemma, Lemma 4).

Construction: e1..e5 = the integer tangents of family_check.VECS; e6 = e7 = ... = e5 (distinct edges with a
common tangent at the vertex).  Then eps(I,J,a) = eps(I,J,5) for every bundle edge a and eps = 0 on every
triple containing two bundle edges.  (A non-coplanar version, e_{m+1} = M e_m + u with eps(I,m,m+1)
independent of I, does not exist for these tangents; extend_noncoplanar() reports the search.)
Then the spectrum of the N-valent operator (sigma from eps, gauge-reduced, hv_core) is compared with the
union over the bundle spin k of the 5-valent spectra; and the smallest nonzero |q| is printed for spins
(1/2, j-1/2, j, j, j-1/2, 1/2, ...) to show it is <= the 5-valent family value.
Usage: python nvalent_check.py
"""
import itertools
import numpy as np
from hv_core import Vertex, sigma_from_eps, spectrum_abs
from family_check import VECS, idet
from chain_check import chain_couplings_sq, chain_spectrum

sg = lambda x: (x > 0) - (x < 0)


def eps_of(V):
    return {t: sg(idet(*[V[i] for i in t])) for t in itertools.combinations(range(len(V)), 3)}


def extend(V):
    """Append e_new = M*e_last + u satisfying (a)-(c)."""
    m = len(V) - 1
    em = V[m]
    for M in range(1, 40):
        for u in itertools.product(range(-3, 4), repeat=3):
            e = tuple(M * em[i] + u[i] for i in range(3))
            W = V + [e]
            E = eps_of(W)
            if 0 in E.values():
                continue
            if any(E[(I, J, m + 1)] != E[(I, J, m)] for I, J in itertools.combinations(range(m), 2)):
                continue
            s = {E[(I, m, m + 1)] for I in range(m)}
            if len(s) != 1:
                continue
            return W
    raise RuntimeError("no extension found")


def main():
    V = [tuple(x) for x in VECS]
    try:
        extend(V)
        print("non-coplanar bundle extension exists")
    except RuntimeError:
        print("non-coplanar bundle extension of e5 (M<40, |u_i|<=3): none found; using tangential bundle")
    Vs = {5: list(V)}
    for N in (6, 7, 8):
        V = V + [V[4]]
        Vs[N] = list(V)
        E = eps_of(V)
        zeros = [t for t, e in E.items() if e == 0]
        print(f"N={N}: tangents {V}; coplanar triples (all contain two bundle edges): {len(zeros)}, "
              f"all of that form: {all(sum(1 for i in t if i >= 4) >= 2 for t in zeros)}")
    # spectral check N=6: union over k of 5-valent blocks
    print("spectral check (merging lemma), N=6, spins (1/2, j-1/2, j, j, j5', j6'):")
    E6 = eps_of(Vs[6])
    E5 = eps_of(Vs[5])
    sig6 = sigma_from_eps(E6, 6)
    sig5 = sigma_from_eps(E5, 5)
    worst = 0
    for N2 in (2, 3, 4, 5, 6):
        for (A5, A6) in [(N2 - 1, 1), (N2, 2), (N2 - 2, 2)]:
            if A5 < 1:
                continue
            Jd6 = (1, N2 - 1, N2, N2, A5, A6)
            if sum(Jd6) % 2:
                continue
            v6 = Vertex(Jd6)
            if v6.dim == 0:
                continue
            s6 = spectrum_abs(v6.Qsigma(sig6))
            blocks = []
            for K in range(abs(A5 - A6), A5 + A6 + 1, 2):
                v5 = Vertex((1, N2 - 1, N2, N2, K))
                if v5.dim:
                    blocks.append(spectrum_abs(v5.Qsigma(sig5)))
            s5 = np.sort(np.concatenate(blocks))
            err = np.abs(s6 - s5).max() / s6.max()
            worst = max(worst, err)
            q6 = s6[s6 > 1e-9 * s6.max()].min()
            _, c2 = chain_couplings_sq((1, N2 - 1, N2, N2, N2), 4, 2)
            qf = chain_spectrum(c2).min()
            print(f"   spins(2j)={Jd6}: dim {v6.dim}, block-union discrepancy {err:.1e}, "
                  f"min nonzero |q| = {q6:.6e}, 5-valent family value {qf:.6e}")
    print(f"   worst relative discrepancy {worst:.2e}")
    # N = 7, 8: one spin tuple each, union over bundle spins
    for N in (7, 8):
        E = eps_of(Vs[N])
        sig = sigma_from_eps(E, N)
        N2 = 4
        Jd = (1, N2 - 1, N2, N2) + ((N2 - 1, 1, 2) if N == 7 else (N2 - 1, 1, 1, 1))
        assert len(Jd) == N and sum(Jd) % 2 == 0
        v = Vertex(Jd)
        s = spectrum_abs(v.Qsigma(sig))
        # bundle legs 5..N couple to K; union of 5-valent blocks with multiplicity
        from hv_core import chains
        bundle = Jd[4:]
        mult = {}
        for ch in chains(list(bundle), set(range(0, 2 * sum(bundle) + 1))):
            mult[ch[-1]] = mult.get(ch[-1], 0) + 1
        blocks = []
        for K, mlt in mult.items():
            v5 = Vertex((1, N2 - 1, N2, N2, K))
            if v5.dim:
                blocks += [spectrum_abs(v5.Qsigma(sig5))] * mlt
        s5 = np.sort(np.concatenate(blocks))
        err = np.abs(s - s5).max() / s.max() if len(s) == len(s5) else float("inf")
        print(f"N={N}: spins(2j)={Jd}, dim {v.dim}, block-union discrepancy {err:.1e}")


if __name__ == "__main__":
    main()

"""Exact (sympy) checks, small and explicitly scoped.

(A) Step 2: for the 4-valent sector Inv(1/2, jJ, jK, S) with dim 2, the single
    off-diagonal entry c of q = [(J_1+J_J)^2,(J_J+J_K)^2] satisfies c^2 = X with
    X = ((jK+S+1)^2-k^2)(k^2-(jK-S)^2), k = jJ+1/2, and X >= k(2k+1) >= 3.
    Exact over all doubled spins jJ,jK,S <= NA.
(B) Theorem 1 kernel claim, exactly: for every spin-1/2 5-tuple with 2j_max <= NB
    and dim >= 2, the stacked matrix [q_T ; T contains leg 1] has full column
    rank d (so the leg-1 triples alone have trivial common kernel), computed in
    exact arithmetic.
"""
import sys, itertools
import sympy as sp
from ref_core import U6, q_triples, dimInv, _tri


def partA(NA):
    n = 0; nmin = None
    for tJ in range(1, NA + 1):
        for tK in range(1, NA + 1):
            for tS in range(0, NA + 1):
                if (1 + tJ + tK + tS) % 2:
                    continue
                ks = [k for k in (tJ - 1, tJ + 1) if k >= 0 and _tri(1, tJ, k) and _tri(k, tK, tS)]
                if len(ks) != 2:
                    continue
                k1, k2 = ks
                xs = [x for x in range(abs(tJ - tK), tJ + tK + 1, 2) if _tri(1, x, tS)]
                B12 = sum(sp.Rational(x * (x + 2), 4) * U6(1, tJ, k1, tK, tS, x, True) * U6(1, tJ, k2, tK, tS, x, True) for x in xs)
                c = (sp.Rational(k1 * (k1 + 2), 4) - sp.Rational(k2 * (k2 + 2), 4)) * B12
                k = sp.Rational(tJ + 1, 2); jK = sp.Rational(tK, 2); S = sp.Rational(tS, 2)
                X = ((jK + S + 1) ** 2 - k ** 2) * (k ** 2 - (jK - S) ** 2)
                diff = sp.nsimplify(sp.expand(c ** 2 - X))
                assert diff == 0, (tJ, tK, tS, c ** 2, X)
                assert X >= k * (2 * k + 1) >= 3
                nmin = X if nmin is None else min(nmin, X)
                n += 1
    return n, nmin


def partB(NB):
    n = 0
    for rest in itertools.combinations_with_replacement(range(1, NB + 1), 4):
        js5 = (1,) + rest
        if sum(js5) % 2:
            continue
        d = dimInv(js5)
        if d < 2:
            continue
        qs, _ = q_triples(js5, exact=True)
        Mx = sp.Matrix.vstack(*[sp.simplify(qs[T]) for T in qs if 0 in T])
        r = Mx.rank(simplify=True)
        assert r == d, (js5, r, d)
        n += 1
    return n


if __name__ == "__main__":
    NA = int(sys.argv[1]); NB = int(sys.argv[2])
    nA, xm = partA(NA)
    print(f"(A) exact: {nA} two-dimensional sectors (1/2,jJ,jK,S) with 2jJ,2jK,2S <= {NA}: c^2 == X in every case; min X = {xm} (=3 gives 4a = sqrt3)")
    nB = partB(NB)
    print(f"(B) exact: {nB} spin-1/2 5-tuples with dim>=2 and 2j_max <= {NB}: stacked leg-1 triples have full column rank (trivial common kernel)")

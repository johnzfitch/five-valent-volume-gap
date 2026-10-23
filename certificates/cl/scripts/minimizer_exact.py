"""Exact value of q_min at the conjectured minimiser (j1; j2,j3; j4,j5) = (1/2; 1/2,1; 1,1) [doubled (1,1,2,2,2)]
and at the next candidates, from the closed-form grid (exact squared weights) with sympy; plus a 60-digit
check of q_min for all tuples with 2j_max <= 4 using mpmath (independent of float LAPACK).
Also the exact Heron-Stark data (T0 window) for the minimiser.
"""
import sys, os, itertools
import sympy as sp
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import grid

mp.mp.dps = 60


def exact_W(tj):
    S, sq = grid(tj, exact=True)
    n = len(S)
    W = sp.zeros(n, n)
    for (i, k), v in sq.items():
        W[i, k] = W[k, i] = sp.sqrt(sp.Rational(v.numerator, v.denominator))
    return S, W


def qmin_exact(tj):
    S, W = exact_W(tj)
    x = sp.symbols('x')
    p = sp.factor((W * W).charpoly(x).as_expr())
    roots = [r for r in sp.solve(p, x) if sp.simplify(r) != 0]
    vals = sorted(roots, key=lambda r: float(r))
    lam2 = vals[0]
    return S, p, sp.nsimplify(sp.simplify(64 * lam2)), sp.sqrt(64 * lam2)


S, p, q2, q = qmin_exact((1, 1, 2, 2, 2))
print("minimiser (1/2;1/2,1;1,1): sites", S)
print("  charpoly of W^2:", p)
print("  q_min^2 = 64 lambda_min^2 =", sp.simplify(q2), " -> q_min =", sp.N(q, 30))
target = 2 * sp.sqrt(18 - 2 * sp.sqrt(41))
print("  equals 2 sqrt(18 - 2 sqrt 41):", sp.simplify(q - target) == 0, sp.N(target, 30))

# 60-digit check over all tuples with 2j_max <= 4 (multisets in legs 2..5)
best = (mp.mpf(10) ** 9, None)
for s2 in range(1, 5):
    for t in itertools.combinations_with_replacement(range(1, 5), 4):
        tj = (s2,) + t
        if sum(tj) % 2: continue
        S, sq = grid(tj, exact=True)
        if len(S) < 2: continue
        n = len(S)
        A = mp.matrix(n, n)
        for (i, k), v in sq.items():
            A[i, k] = A[k, i] = mp.sqrt(mp.mpf(v.numerator) / v.denominator)
        ev = mp.eigsy(A)[0]
        nz = [abs(e) for e in ev if abs(e) > mp.mpf(10) ** -40]
        if nz and min(nz) * 8 < best[0]:
            best = (min(nz) * 8, tj)
print("60-digit minimum of q over all tuples with 2j_max<=4:", mp.nstr(best[0], 40), "at", best[1])

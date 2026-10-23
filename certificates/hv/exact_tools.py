"""Exact rational form of the AL volume operator and certified smallest nonzero eigenvalue.

In the chain basis every reduced matrix element of J_I is s*sqrt(r) with r rational (hv_core.reduced_J
returns the signed square).  A diagonal rescaling |c> -> sqrt(w_c)|c> with rational w_c (built by BFS and
checked on every edge) makes all of them rational; the matrices of J_I.J_J and hence
A = sum sigma qhat become rational matrices similar to the float ones.  Their characteristic polynomial
p(x) = det(x - A) is computed exactly (python-flint).  Since A is real antisymmetric,
p(x) = x^k0 * P(x^2) with P having only negative real roots -q^2 (q = |eigenvalue of iA| > 0).
k0 = dim ker A exactly; the smallest q is certified by rigorous real-root isolation of P(-t) (arb balls).
"""
from fractions import Fraction
from math import isqrt
import flint
from hv_core import Vertex, reduced_J


def _is_sq(fr):
    if fr < 0:
        return False
    n, d = fr.numerator, fr.denominator
    a, b = isqrt(n), isqrt(d)
    return a * a == n and b * b == d


def _sqrt_q(fr):
    return Fraction(isqrt(fr.numerator), isqrt(fr.denominator))


def rational_A(Jd, sigma):
    """Return (fmpq_mat A_rat, dim) with A_rat similar to sum sigma qhat (BR normalisation)."""
    v = Vertex(Jd)
    n = len(v.legs)
    legs = sorted({I for t in sigma for I in t})
    # exact reduced elements for needed legs
    red = {}
    for I in legs:
        v.red(I)
        red[I] = v._red[I][1]            # dict (i_basis, k_ext) -> signed square
    d = v.dim
    L1 = v.L + 1
    # columns: ext chains c''.  Entry (a, k) of leg I has value s*sqrt(r).
    cols = {}
    for I in legs:
        for (i, k), val in red[I].items():
            cols.setdefault(k, []).append((I, i, val))
    # row weights w_a (rational): need |v1||v2| w_b / w_a to be a rational square for any two entries
    # (a, k), (b, k) in the same column (any legs).  BFS over rows.
    adj = {}
    for k, lst in cols.items():
        I0, a0, v0 = lst[0]
        for (I, b, vb) in lst[1:]:
            adj.setdefault(a0, []).append((b, abs(v0), abs(vb)))
            adj.setdefault(b, []).append((a0, abs(vb), abs(v0)))
    w = {}
    for st in range(d):
        if st in w:
            continue
        w[st] = Fraction(1)
        stack = [st]
        while stack:
            a = stack.pop()
            for b, va, vb in adj.get(a, []):
                if b not in w:
                    # choose w_b so that va*vb*w_b/w_a = 1 * square -> w_b = w_a/(va*vb)
                    w[b] = w[a] / (va * vb)
                    stack.append(b)

    def term(va, vb, a, b):
        x = abs(va) * abs(vb) * w[b] / w[a]
        if not _is_sq(x):
            raise AssertionError("rationalisation failed")
        q = _sqrt_q(x)
        return q if (va > 0) == (vb > 0) else -q

    def dot(I, J):
        M = [[Fraction(0)] * d for _ in range(d)]
        for k, lst in cols.items():
            rowsI = [(a, va) for (K, a, va) in lst if K == I]
            rowsJ = [(b, vb) for (K, b, vb) in lst if K == J]
            for a, va in rowsI:
                for b, vb in rowsJ:
                    M[a][b] += term(va, vb, a, b) / L1
        return flint.fmpq_mat(d, d, [flint.fmpq(e.numerator, e.denominator) for row in M for e in row])

    dots = {}

    def D(I, J):
        key = (I, J)
        if key not in dots:
            dots[key] = dot(I, J)
        return dots[key]

    A = flint.fmpq_mat(d, d)
    for (I, J, K), s in sigma.items():
        if not s:
            continue
        X, Y = D(I, J), D(J, K)
        A = A + (X * Y - Y * X) * (4 * s)
    return A, d


def certify(Jd, sigma, prec_digits=30):
    """Exact kernel dimension and rigorous enclosure of the smallest nonzero q (= |eig of iA|)."""
    A, d = rational_A(Jd, sigma)
    p = A.charpoly()                       # fmpq_poly, monic, degree d
    c = [p.coeffs()[i] for i in range(d + 1)]
    k0 = next(i for i, x in enumerate(c) if x != 0)
    # P(x^2) = p(x)/x^k0 ; coefficients of x^(k0+2m)
    assert all(c[k0 + 2 * m + 1] == 0 for m in range((d - k0) // 2)), "not even in x"
    Pc = [c[k0 + 2 * m] for m in range((d - k0) // 2 + 1)]   # P(y), y = x^2 = -q^2
    # roots t = q^2 > 0 of P(-t)
    Q = [Pc[m] * (-1) ** m for m in range(len(Pc))]
    den = 1
    for x in Q:
        den = den * x.q // flint.fmpz.gcd(den, x.q) if hasattr(flint.fmpz, 'gcd') else den * int(x.q)
    Zp = flint.fmpz_poly([int(x * den) for x in Q])
    sq = Zp.factor_squarefree()
    best = None
    for fac, mult in sq[1]:
        for r in fac.complex_roots():
            rr = r[0] if isinstance(r, tuple) else r
            if best is None or rr.real.mid() < best.real.mid():
                best = rr
    return {"dim": d, "kernel_dim": k0, "q2_min_ball": best, "charpoly_degree": p.degree()}

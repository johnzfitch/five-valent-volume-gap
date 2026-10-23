"""Independent referee implementation of the 5-valent RS volume operator.

Written from scratch for the referee; imports nothing from the author's code.
Spins are stored doubled (integers).  Basis of Inv(j1..j5): tree
((((l1 l2)a l3)b l4)c) with c = j5 (leg 5 is the root leg).  Pair operators
(J_I+J_J)^2 are obtained by changing leg order with adjacent-swap overlaps
built from Racah 6j symbols (Edmonds conventions).  Both a float64 mode and an
exact sympy mode are provided.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, combinations
import math
import numpy as np

# ---------------------------------------------------------------- 6j symbols
@lru_cache(maxsize=None)
def fact(n):
    return math.factorial(n)


def _tri(a, b, c):  # doubled spins
    return (a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b


def _delta2(a, b, c):  # Delta^2 as Fraction, doubled spins
    return Fraction(fact((a + b - c) // 2) * fact((a - b + c) // 2) * fact((-a + b + c) // 2),
                    fact((a + b + c) // 2 + 1))


@lru_cache(maxsize=1<<20)
def sixj_parts(a, b, c, d, e, f):
    """{a b c; d e f} = sign * sqrt(P) * S with P, S Fractions; returns (P, S) or None."""
    if not (_tri(a, b, c) and _tri(a, e, f) and _tri(d, b, f) and _tri(d, e, c)):
        return None
    P = _delta2(a, b, c) * _delta2(a, e, f) * _delta2(d, b, f) * _delta2(d, e, c)
    t1 = (a + b + c) // 2; t2 = (a + e + f) // 2; t3 = (d + b + f) // 2; t4 = (d + e + c) // 2
    s1 = (a + b + d + e) // 2; s2 = (a + c + d + f) // 2; s3 = (b + c + e + f) // 2
    S = 0
    for t in range(max(t1, t2, t3, t4), min(s1, s2, s3) + 1):
        S += Fraction((-1) ** t * fact(t + 1),
                      fact(t - t1) * fact(t - t2) * fact(t - t3) * fact(t - t4)
                      * fact(s1 - t) * fact(s2 - t) * fact(s3 - t))
    return P, S


@lru_cache(maxsize=1<<20)
def sixj(a, b, c, d, e, f):
    r = sixj_parts(a, b, c, d, e, f)
    if r is None:
        return 0.0
    P, S = r
    return math.sqrt(float(P)) * float(S)


def sixj_exact(a, b, c, d, e, f):
    import sympy as sp
    r = sixj_parts(a, b, c, d, e, f)
    if r is None:
        return sp.Integer(0)
    P, S = r
    return sp.sqrt(sp.Rational(P.numerator, P.denominator)) * sp.Rational(S.numerator, S.denominator)


def phase(n2):  # (-1)^(n2/2) for even n2
    assert n2 % 2 == 0
    return -1 if (n2 // 2) % 2 else 1


@lru_cache(maxsize=1<<20)
def U6(j1, j2, j12, j3, J, j23, exact=False):
    """<j1,(j2 j3)j23;J | (j1 j2)j12, j3;J>  (Edmonds 6.4.3), doubled spins."""
    s = phase(j1 + j2 + j3 + J)
    if exact:
        import sympy as sp
        return s * sp.sqrt((j12 + 1) * (j23 + 1)) * sixj_exact(j1, j2, j12, j3, J, j23)
    return s * math.sqrt((j12 + 1) * (j23 + 1)) * sixj(j1, j2, j12, j3, J, j23)


# ---------------------------------------------------------------- basis
def basis(js):
    """js: doubled spins of legs in tree order (l1,l2,l3,l4,l5). Returns list of (a,b)."""
    l1, l2, l3, l4, l5 = js
    out = []
    for a in range(abs(l1 - l2), l1 + l2 + 1, 2):
        for b in range(abs(a - l3), a + l3 + 1, 2):
            if _tri(b, l4, l5):
                out.append((a, b))
    return out


def swap_overlap(js, pos, exact=False):
    """Matrix O with O[i,k] = <old_i | new_k>, where new ordering swaps positions pos,pos+1
    (0-based, pos in 0,1,2) of the old ordering js.  Returns (O, new_js)."""
    new = list(js); new[pos], new[pos + 1] = new[pos + 1], new[pos]
    B0 = basis(js); B1 = basis(new)
    idx1 = {v: k for k, v in enumerate(B1)}
    if exact:
        import sympy as sp
        O = sp.zeros(len(B0), len(B1))
    else:
        O = np.zeros((len(B0), len(B1)))
    l1, l2, l3, l4, l5 = js
    byb = {}; bya = {}
    for k, (a2, b2) in enumerate(B1):
        byb.setdefault(b2, []).append(k); bya.setdefault(a2, []).append(k)
    for i, (a, b) in enumerate(B0):
        if pos == 0:
            k = idx1[(a, b)]
            O[i, k] = phase(l1 + l2 - a)
        elif pos == 1:
            # ((l1 l2)a l3)b -> ((l1 l3)a' l2)b
            for k in byb.get(b, ()):
                a2, b2 = B1[k]
                val = 0
                for x in range(abs(l2 - l3), l2 + l3 + 1, 2):
                    if not (_tri(l1, x, b)):
                        continue
                    val += U6(l1, l2, a, l3, b, x, exact) * phase(l2 + l3 - x) * U6(l1, l3, a2, l2, b, x, exact)
                O[i, k] = val
        else:
            # ((a l3)b l4)c -> ((a l4)b' l3)c, c = l5
            for k in bya.get(a, ()):
                a2, b2 = B1[k]
                val = 0
                for x in range(abs(l3 - l4), l3 + l4 + 1, 2):
                    if not _tri(a, x, l5):
                        continue
                    val += U6(a, l3, b, l4, l5, x, exact) * phase(l3 + l4 - x) * U6(a, l4, b2, l3, l5, x, exact)
                O[i, k] = val
    return O, tuple(new)


def pair_ops(js5, exact=False):
    """Return dict {(I,J): matrix of (J_I+J_J)^2} on Inv in the reference basis, I<J in 0..4.
    js5 doubled spins of legs 0..4; leg 4 is the root leg."""
    ref = tuple(range(4))
    # BFS over orderings of legs 0..3
    V = {ref: None}
    frontier = [ref]
    while frontier:
        nf = []
        for perm in frontier:
            for pos in range(3):
                p2 = list(perm); p2[pos], p2[pos + 1] = p2[pos + 1], p2[pos]; p2 = tuple(p2)
                if p2 in V:
                    continue
                js = tuple(js5[p] for p in perm) + (js5[4],)
                O, _ = swap_overlap(js, pos, exact)
                if V[perm] is None:
                    V[p2] = O
                else:
                    V[p2] = V[perm] * O if exact else V[perm] @ O
                nf.append(p2)
        frontier = nf
    n = len(basis(js5))
    ops = {}
    for perm in permutations(range(4)):
        js = tuple(js5[p] for p in perm) + (js5[4],)
        B = basis(js)
        if exact:
            import sympy as sp
            Da = sp.diag(*[sp.Rational(a * (a + 2), 4) for a, b in B]) if B else sp.zeros(0, 0)
            Db = sp.diag(*[sp.Rational(b * (b + 2), 4) for a, b in B]) if B else sp.zeros(0, 0)
            Vp = sp.eye(n) if V[perm] is None else V[perm]
            opa = Vp * Da * Vp.T; opb = Vp * Db * Vp.T
        else:
            Da = np.diag([a * (a + 2) / 4 for a, b in B]); Db = np.diag([b * (b + 2) / 4 for a, b in B])
            Vp = np.eye(n) if V[perm] is None else V[perm]
            opa = Vp @ Da @ Vp.T; opb = Vp @ Db @ Vp.T
        p = tuple(sorted(perm[:2]))
        ops.setdefault(p, opa)
        q = (perm[3], 4)
        ops.setdefault(q, opb)
    return ops, V


def q_triples(js5, exact=False):
    ops, _ = pair_ops(js5, exact)
    def P(i, j):
        return ops[(min(i, j), max(i, j))]
    qs = {}
    for (I, J, K) in combinations(range(5), 3):
        A = P(I, J); B = P(J, K)
        qs[(I, J, K)] = A * B - B * A if exact else A @ B - B @ A
    return qs, ops


def absq(q):
    """|q| for real antisymmetric q (anti-Hermitian): diagonalise Hermitian i q."""
    H = 1j * q
    w, U = np.linalg.eigh(H)
    return (U * np.abs(w)) @ U.conj().T, w, U


def dimInv(js5):
    return len(basis(js5))

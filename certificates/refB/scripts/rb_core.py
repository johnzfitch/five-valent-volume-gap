"""Referee B core: an independent exact construction of T = J1.(J2xJ3 + J4xJ5) on Inv(j1..j5).

Written from Edmonds' formulas only; imports nothing from the authors' folders.

Basis |(j2 j3)b, (j4 j5)c ; L = j1> coupled with j1 to total 0  (pairing (23|45)).
  J1.V on the coupled state            : (-1)^{2s} {0 s s; 1 s s} <s||J||s> <R'||V||R>          (Edmonds 7.1.6)
  V acting on the b factor of (b c)s   : (-1)^{b'+c+s+1} (2s+1) {b' s c; s b 1} <b'||V||b>      (7.1.7)
  V acting on the c factor             : (-1)^{b+c+s+1}  (2s+1) {c' s b; s c 1} <c'||V||c>      (7.1.8)
  J2xJ3 = (1/2i)[J2.J3, J2 - J3]  and  <b'||J2+J3||b> = 0 for b' != b, so
  <b'||J2xJ3||b> = (1/2i)(e_b' - e_b) * 2 <b'||J2||b>,  e_b = (b(b+1) - j2(j2+1) - j3(j3+1))/2,
  <b'||J2||b> = (-1)^{j2+j3+b+1} sqrt((2b+1)(2b'+1)) {j2 b' j3; b j2 1} sqrt(j2(j2+1)(2j2+1)).
So T = (C0 / 2i) * R with R real antisymmetric and every entry of R equal to  sign * sqrt(rational).
All spins are DOUBLED integers.  Exact 6j symbols: Racah sum in integers, returned as (sign, square as Fraction).
"""
from fractions import Fraction as Fr
from functools import lru_cache
from math import factorial, isqrt
import numpy as np


def _tri(a, b, c):
    """doubled spins: triangle and integrality"""
    return (a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b


@lru_cache(maxsize=None)
def _fact(n):
    return factorial(n)


def _delta2(a, b, c):
    """Delta(abc)^2 for doubled spins, as Fraction"""
    return Fr(_fact((a + b - c) // 2) * _fact((a - b + c) // 2) * _fact((-a + b + c) // 2), _fact((a + b + c) // 2 + 1))


@lru_cache(maxsize=None)
def sixj(a, b, c, d, e, f):
    """{a b c; d e f} (doubled) -> (sign, square) exactly; (0, 0) if it vanishes."""
    if not (_tri(a, b, c) and _tri(a, e, f) and _tri(d, b, f) and _tri(d, e, c)):
        return 0, Fr(0)
    D2 = _delta2(a, b, c) * _delta2(a, e, f) * _delta2(d, b, f) * _delta2(d, e, c)
    t1 = (a + b + c) // 2; t2 = (a + e + f) // 2; t3 = (d + b + f) // 2; t4 = (d + e + c) // 2
    u1 = (a + b + d + e) // 2; u2 = (b + c + e + f) // 2; u3 = (c + a + f + d) // 2
    S = Fr(0)
    for t in range(max(t1, t2, t3, t4), min(u1, u2, u3) + 1):
        S += Fr((-1) ** t * _fact(t + 1),
                _fact(t - t1) * _fact(t - t2) * _fact(t - t3) * _fact(t - t4) * _fact(u1 - t) * _fact(u2 - t) * _fact(u3 - t))
    if S == 0:
        return 0, Fr(0)
    return (1 if S > 0 else -1), S * S * D2


def sixj_float(*args):
    sg, sq = sixj(*args)
    return sg * float(sq) ** 0.5


def _phase(two_x):
    """(-1)^x for integer x given as doubled 2x"""
    assert two_x % 2 == 0, two_x
    return -1 if (two_x // 2) % 2 else 1


def red_J_first(ja, jb, k2, k2p):
    """<(ja jb)k'||J_a||(ja jb)k> (doubled args) -> (sign, square)"""
    sg, sq = sixj(ja, k2p, jb, k2, ja, 2)
    if sg == 0:
        return 0, Fr(0)
    ph = _phase(ja + jb + k2 + 2)
    sq2 = sq * (k2 + 1) * (k2p + 1) * Fr(ja * (ja + 2) * (ja + 1), 4)
    return ph * sg, sq2


def C0_sq_sign(s2):
    """C0 = (-1)^{2s} {0 s s; 1 s s} sqrt(s(s+1)(2s+1)) -> (sign, square)"""
    sg, sq = sixj(0, s2, s2, 2, s2, s2)
    ph = 1 if s2 % 2 == 0 else -1
    return ph * sg, sq * Fr(s2 * (s2 + 2) * (s2 + 1), 4)


def sites_of(tj):
    """tj = (2j1,2j2,2j3,2j4,2j5); sites (b2, c2) of pairing (23|45) with triangle(b, c, j1)."""
    s2, a, b, c, d = tj
    out = []
    for B in range(abs(a - b), a + b + 1, 2):
        for C in range(abs(c - d), c + d + 1, 2):
            if _tri(B, C, s2):
                out.append((B, C))
    return out


def R_entries(tj):
    """exact entries of R (T = C0/(2i) R): dict {(i, j): (sign, square)} for i<j site indices, R[i,j] = sign*sqrt(square),
    R[j,i] = -R[i,j] (antisymmetry is checked separately in float by building both directions)."""
    s2, j2, j3, j4, j5 = tj
    S = sites_of(tj)
    idx = {p: i for i, p in enumerate(S)}
    ent = {}

    def e_of(ja, jb, k2):
        return Fr(k2 * (k2 + 2) - ja * (ja + 2) - jb * (jb + 2), 8)

    for (B, C), i in idx.items():
        # X: (B, C) -> (B+2, C)  matrix element <B+2, C| R |B, C>
        Bp = B + 2
        if (Bp, C) in idx:
            g6, q6 = sixj(Bp, s2, C, s2, B, 2)
            gJ, qJ = red_J_first(j2, j3, B, Bp)
            de = e_of(j2, j3, Bp) - e_of(j2, j3, B)
            if g6 != 0 and gJ != 0 and de != 0:
                ph = _phase(Bp + C + s2 + 2)
                sgn = ph * g6 * gJ * (1 if de > 0 else -1)
                sq = Fr((s2 + 1) ** 2) * q6 * de * de * 4 * qJ
                ent[(idx[(Bp, C)], i)] = (sgn, sq)
        Cp = C + 2
        if (B, Cp) in idx:
            g6, q6 = sixj(Cp, s2, B, s2, C, 2)
            gJ, qJ = red_J_first(j4, j5, C, Cp)
            de = e_of(j4, j5, Cp) - e_of(j4, j5, C)
            if g6 != 0 and gJ != 0 and de != 0:
                ph = _phase(B + C + s2 + 2)
                sgn = ph * g6 * gJ * (1 if de > 0 else -1)
                sq = Fr((s2 + 1) ** 2) * q6 * de * de * 4 * qJ
                ent[(idx[(B, Cp)], i)] = (sgn, sq)
    return S, ent


def R_entries_reverse(tj):
    """the same matrix elements computed in the opposite direction <B,C|R|B+2,C>, for the antisymmetry check"""
    s2, j2, j3, j4, j5 = tj
    S = sites_of(tj)
    idx = {p: i for i, p in enumerate(S)}
    ent = {}

    def e_of(ja, jb, k2):
        return Fr(k2 * (k2 + 2) - ja * (ja + 2) - jb * (jb + 2), 8)

    for (B, C), i in idx.items():
        Bp = B + 2
        if (Bp, C) in idx:
            g6, q6 = sixj(B, s2, C, s2, Bp, 2)
            gJ, qJ = red_J_first(j2, j3, Bp, B)
            de = e_of(j2, j3, B) - e_of(j2, j3, Bp)
            if g6 != 0 and gJ != 0 and de != 0:
                ph = _phase(B + C + s2 + 2)
                ent[(i, idx[(Bp, C)])] = (ph * g6 * gJ * (1 if de > 0 else -1), Fr((s2 + 1) ** 2) * q6 * de * de * 4 * qJ)
        Cp = C + 2
        if (B, Cp) in idx:
            g6, q6 = sixj(C, s2, B, s2, Cp, 2)
            gJ, qJ = red_J_first(j4, j5, Cp, C)
            de = e_of(j4, j5, C) - e_of(j4, j5, Cp)
            if g6 != 0 and gJ != 0 and de != 0:
                ph = _phase(B + Cp + s2 + 2)
                ent[(i, idx[(B, Cp)])] = (ph * g6 * gJ * (1 if de > 0 else -1), Fr((s2 + 1) ** 2) * q6 * de * de * 4 * qJ)
    return S, ent


def T_matrix(tj):
    """complex Hermitian matrix of T (float), sites"""
    S, ent = R_entries(tj)
    n = len(S)
    R = np.zeros((n, n))
    for (i, j), (sg, sq) in ent.items():
        v = sg * float(sq) ** 0.5
        R[i, j] = v; R[j, i] = -v
    g0, q0 = C0_sq_sign(tj[0])
    C0 = g0 * float(q0) ** 0.5
    return S, (C0 / 2j) * R


def scale_sq(s2):
    """|T entry|^2 = scale_sq * R entry^2  (exact Fraction)"""
    g0, q0 = C0_sq_sign(s2)
    return q0 / 4


def q_values(tj, tol=1e-10):
    """sorted list of nonzero q = 8|eig T| (float), and kernel dimension"""
    S, T = T_matrix(tj)
    if len(S) == 0:
        return [], 0
    w = np.linalg.eigvalsh(T)
    sc = max(1.0, np.abs(w).max())
    nz = np.sort(8 * np.abs(w[np.abs(w) > tol * sc]))
    return list(nz), int(np.sum(np.abs(w) <= tol * sc))


def qmin(tj, tol=1e-10):
    v, k = q_values(tj, tol)
    return (v[0] if v else None), k


def dim_inv(tj):
    return len(sites_of(tj))


# ---------------------------------------------------------------- bipartite structure and exact certification
def colour_classes(tj):
    """two colour classes of the bipartite graph of R (BFS 2-colouring of the actual nonzero links)"""
    S, ent = R_entries(tj)
    n = len(S)
    adj = [[] for _ in range(n)]
    for (i, j) in ent:
        adj[i].append(j); adj[j].append(i)
    col = [-1] * n
    for r in range(n):
        if col[r] >= 0:
            continue
        col[r] = 0; st = [r]
        while st:
            v = st.pop()
            for w in adj[v]:
                if col[w] < 0:
                    col[w] = 1 - col[v]; st.append(w)
                elif col[w] == col[v]:
                    raise RuntimeError('not bipartite')
    return S, ent, col


def is_square_fr(x):
    if x < 0:
        return None
    p, q = x.numerator, x.denominator
    rp, rq = isqrt(p), isqrt(q)
    if rp * rp == p and rq * rq == q:
        return Fr(rp, rq)
    return None

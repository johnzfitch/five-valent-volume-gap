"""Referee library (independent of the author's code).

* sixj(): exact Racah formula on doubled spins (Python integers), float result.
* bc_operator(): signed real antisymmetric matrix of
      A = -4i[alpha J1.(J2xJ3) + beta J1.(J4xJ5)]
  on Inv(j1..j5) in the basis |[((j2 j3)b, j1)c, (j4 j5)c]0>, built from
      X = (i/4)[(J1+J2)^2, (J2+J3)^2]   on the 3-leg space at fixed c,
      D = (i/4)[(J1+J4)^2, (J4+J5)^2]   on the 3-object space ((B j1)c j4) j5 at fixed b,
  with standard (Condon-Shortley) recoupling phases.
* brute(): full tensor-product operators on Inv for small spins.
All spins are passed doubled (integers).
"""
import itertools
from functools import lru_cache
from fractions import Fraction
import math
import numpy as np


@lru_cache(maxsize=None)
def fact(n):
    return math.factorial(n)


def tri(a, b, c):  # doubled
    return (a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b


@lru_cache(maxsize=None)
def delta2(a, b, c):  # Delta(abc)^2 as Fraction, doubled args
    return Fraction(fact((a + b - c) // 2) * fact((a - b + c) // 2) * fact((-a + b + c) // 2),
                    fact((a + b + c) // 2 + 1))


@lru_cache(maxsize=None)
def sixj_exact_sq_and_sign(a, b, c, d, e, f):
    """{a b c; d e f} (doubled) = sign*sqrt(value)."""
    if not (tri(a, b, c) and tri(a, e, f) and tri(d, b, f) and tri(d, e, c)):
        return 0, Fraction(0)
    pre = delta2(a, b, c) * delta2(a, e, f) * delta2(d, b, f) * delta2(d, e, c)
    t1 = (a + b + c) // 2; t2 = (a + e + f) // 2; t3 = (d + b + f) // 2; t4 = (d + e + c) // 2
    s1 = (a + b + d + e) // 2; s2 = (b + c + e + f) // 2; s3 = (a + c + d + f) // 2
    tot = 0
    for t in range(max(t1, t2, t3, t4), min(s1, s2, s3) + 1):
        term = Fraction(fact(t + 1), fact(t - t1) * fact(t - t2) * fact(t - t3) * fact(t - t4)
                        * fact(s1 - t) * fact(s2 - t) * fact(s3 - t))
        tot += (-1) ** t * term
    if tot == 0:
        return 0, Fraction(0)
    sgn = 1 if tot > 0 else -1
    return sgn, pre * tot * tot


def sixj(a, b, c, d, e, f):
    s, v = sixj_exact_sq_and_sign(a, b, c, d, e, f)
    if s == 0:
        return 0.0
    return s * math.sqrt(v)


def ph(x2):
    """(-1)^(x) for doubled integer x2 that must be even."""
    assert x2 % 2 == 0, x2
    return -1 if (x2 // 2) % 2 else 1


def sites(J):
    j1, j2, j3, j4, j5 = J
    bs = [b for b in range(abs(j2 - j3), j2 + j3 + 1, 2)]
    cs = [c for c in range(abs(j4 - j5), j4 + j5 + 1, 2)]
    return [(b, c) for b in bs for c in cs if tri(b, j1, c)]


def bc_parts(J):
    """Return (sites, Xr, Dr): real antisymmetric with X = i*Xr, D = i*Dr."""
    j1, j2, j3, j4, j5 = J
    S = sites(J)
    idx = {s: i for i, s in enumerate(S)}
    n = len(S)
    Xr = np.zeros((n, n)); Dr = np.zeros((n, n))
    cs = sorted({c for _, c in S}); bs = sorted({b for b, _ in S})
    for c in cs:
        bl = [b for b, cc in S if cc == c]
        es = [e for e in range(abs(j1 - j2), j1 + j2 + 1, 2) if tri(e, j3, c)]
        assert len(es) == len(bl)
        if len(bl) < 2:
            continue
        R = np.zeros((len(bl), len(es)))
        for i, b in enumerate(bl):
            for k, e in enumerate(es):
                R[i, k] = (ph(b + j1 - c) * ph(j1 + j2 + j3 + c)
                           * math.sqrt((e + 1) * (b + 1)) * sixj(j1, j2, e, j3, c, b))
        assert np.allclose(R @ R.T, np.eye(len(bl)), atol=1e-12)
        Le = np.diag([e * (e + 2) / 4 for e in es])
        Lb = np.diag([b * (b + 2) / 4 for b in bl])
        M12 = R @ Le @ R.T
        Xc = 0.25 * (M12 @ Lb - Lb @ M12)     # X = (i/4)[(J1+J2)^2,(J2+J3)^2] = i*Xc
        ii = [idx[(b, c)] for b in bl]
        Xr[np.ix_(ii, ii)] = Xc
    for b in bs:
        cl = [c for bb, c in S if bb == b]
        es = [e for e in range(abs(j1 - j4), j1 + j4 + 1, 2) if tri(b, e, j5)]
        assert len(es) == len(cl)
        if len(cl) < 2:
            continue
        Sm = np.zeros((len(cl), len(es)))
        for i, c in enumerate(cl):
            # |[(B j1)c,(j4 j5)c]0> = phi_c |(((B j1)c j4) j5) 0>
            phi = ph(c + j4 + j5) * math.sqrt((j5 + 1) * (c + 1)) * sixj(c, j4, j5, j5, 0, c)
            for k, e in enumerate(es):
                Sm[i, k] = phi * ph(b + j1 + j4 + j5) * math.sqrt((c + 1) * (e + 1)) * sixj(b, j1, c, j4, j5, e)
        assert np.allclose(Sm @ Sm.T, np.eye(len(cl)), atol=1e-12)
        Le = np.diag([e * (e + 2) / 4 for e in es])
        Lc = np.diag([c * (c + 2) / 4 for c in cl])
        M14 = Sm @ Le @ Sm.T
        Dc = 0.25 * (M14 @ Lc - Lc @ M14)     # D = (i/4)[(J1+J4)^2,(J4+J5)^2]
        ii = [idx[(b, c)] for c in cl]
        Dr[np.ix_(ii, ii)] = Dc
    return S, Xr, Dr


def bc_operator(J, alpha, beta):
    """Real antisymmetric matrix M with A = -4i[alpha X + beta D] = 4*(alpha Xr + beta Dr)."""
    S, Xr, Dr = bc_parts(J)
    return S, 4.0 * (alpha * Xr + beta * Dr)


def qvals(M, rel=1e-11):
    if M.shape[0] == 0:
        return np.array([]), 0
    s = np.linalg.svd(M, compute_uv=False)
    thr = rel * max(1.0, s.max() if s.size else 1.0)
    nz = np.sort(s[s > thr])
    return nz, int((s <= thr).sum())


# ---------------- brute force ----------------
def spin_mats(j2):
    d = j2 + 1
    m = np.array([j2 / 2 - k for k in range(d)])
    Jz = np.diag(m)
    Jp = np.zeros((d, d))
    for k in range(1, d):
        mm = m[k]
        Jp[k - 1, k] = math.sqrt(j2 / 2 * (j2 / 2 + 1) - mm * (mm + 1))
    Jx = (Jp + Jp.T) / 2
    Jy = (Jp - Jp.T) / (2j)
    return [Jx.astype(complex), Jy, Jz.astype(complex)]


def brute_inv(J):
    """Return (P, legs) with P an orthonormal basis of Inv (columns), legs[I][a] full-space matrices."""
    mats = [spin_mats(j) for j in J]
    dims = [j + 1 for j in J]
    legs = []
    for I in range(len(J)):
        comp = []
        for a in range(3):
            ops = [np.eye(d) for d in dims]
            ops[I] = mats[I][a]
            M = ops[0]
            for o in ops[1:]:
                M = np.kron(M, o)
            comp.append(M)
        legs.append(comp)
    Jt = [sum(legs[I][a] for I in range(len(J))) for a in range(3)]
    C = sum(Jt[a] @ Jt[a] for a in range(3))
    w, V = np.linalg.eigh(C)
    P = V[:, np.abs(w) < 1e-8]
    return P, legs


def triple(legs, I, J, K):
    a, b, c = legs[I], legs[J], legs[K]
    cross = [b[1] @ c[2] - b[2] @ c[1], b[2] @ c[0] - b[0] @ c[2], b[0] @ c[1] - b[1] @ c[0]]
    return sum(a[k] @ cross[k] for k in range(3))

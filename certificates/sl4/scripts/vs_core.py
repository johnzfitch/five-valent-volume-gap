"""Core for the sharp 4-valent law (independent code; imports nothing from other folders).

Spins are given doubled: J_i = 2 j_i (positive integers). Sorted j1 <= j2 <= j3 <= j4, pairing (12)(34).
Chain c_s = a_{kmin+s}, s = 1..d-1, with
  a_k^2 = [(A1^2-k^2)(k^2-B1^2)(A2^2-k^2)(k^2-B2^2)] / (16(4k^2-1)),
  A1 = j1+j2+1, B1 = |j1-j2|, A2 = j3+j4+1, B2 = |j3-j4|.
Odd/even split: B has diagonal c_1, c_3, ... and subdiagonal c_2, c_4, ...; the nonzero spectrum of Q
is +-sigma(B), so lambda^2 = lambda_min(B^T B).  M := B^T B is m x m tridiagonal, m = floor(n/2), n = d.
"""
from fractions import Fraction
import numpy as np


def kdata(J):
    J1, J2, J3, J4 = J
    kmin2 = max(abs(J1 - J2), abs(J3 - J4))
    kmax2 = min(J1 + J2, J3 + J4)
    if kmin2 > kmax2 or (J1 + J2 - kmin2) % 2 or (J3 + J4 - kmin2) % 2:
        return None
    return kmin2, kmax2, (kmax2 - kmin2) // 2 + 1


def chain2_exact(J):
    """exact c_s^2, s = 1..d-1, as Fractions."""
    kd = kdata(J)
    if kd is None:
        return None
    kmin2, kmax2, d = kd
    J1, J2, J3, J4 = J
    A1, B1 = Fraction(J1 + J2 + 2, 2), Fraction(abs(J1 - J2), 2)
    A2, B2 = Fraction(J3 + J4 + 2, 2), Fraction(abs(J3 - J4), 2)
    out = []
    for s in range(1, d):
        k = Fraction(kmin2, 2) + s
        out.append((A1 * A1 - k * k) * (k * k - B1 * B1) * (A2 * A2 - k * k) * (k * k - B2 * B2) / (16 * (4 * k * k - 1)))
    return out


def chain2_float(J):
    c = chain2_exact(J)
    return None if c is None else np.array([float(x) for x in c])


def M_tridiag(c2):
    """B^T B for the chain with squared entries c2 (list), as (diag, offdiag^2)."""
    n = len(c2) + 1
    m = n // 2
    diag, off2 = [], []
    for i in range(1, m + 1):              # column i <-> c_{2i-1} (diag) and c_{2i} (sub)
        dgi = c2[2 * i - 2]
        if 2 * i - 1 <= len(c2) - 1:
            dgi = dgi + c2[2 * i - 1]
        diag.append(dgi)
        if i < m:
            off2.append(c2[2 * i - 1] * c2[2 * i])
    return diag, off2


def ldl_posdef(diag, off2, mu):
    """exact: is M - mu I positive definite? (all LDL^T pivots > 0)."""
    piv = diag[0] - mu
    if piv <= 0:
        return False
    for i in range(1, len(diag)):
        piv = diag[i] - mu - off2[i - 1] / piv
        if piv <= 0:
            return False
    return True


def lam2_float(J):
    c2 = chain2_float(J)
    if c2 is None or len(c2) == 0:
        return None
    from scipy.linalg import eigh_tridiagonal
    dg, o2 = M_tridiag(list(c2))
    w = eigh_tridiagonal(np.array(dg), np.sqrt(np.array(o2)), eigvals_only=True, select="i", select_range=(0, 0))
    return float(w[0])


def sorted_tuples(N2max, N2min=1):
    """all sorted (J1<=J2<=J3<=J4) doubled-spin tuples with J4 in [N2min, N2max] and d >= 1."""
    for J4 in range(N2min, N2max + 1):
        for J3 in range(1, J4 + 1):
            for J2 in range(1, J3 + 1):
                for J1 in range(1, J2 + 1):
                    if (J1 + J2 + J3 + J4) % 2:
                        continue
                    if kdata((J1, J2, J3, J4)) is None:
                        continue
                    yield (J1, J2, J3, J4)

"""Validate oc_core: closed-form grid vs brute-force tensor product of T = J1.(J2xJ3+J4xJ5) on Inv.
Also: Heron-Stark congruence W = 1/4 S W0 S (exact, rationals) on all tuples 2j<=Mx.
Usage: python oc_validate.py Mbrute Mexact
"""
import sys, os, itertools
import numpy as np
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_core import *


def spin(tj):
    j = tj / 2
    m = j - np.arange(tj + 1)
    Jp = np.zeros((tj + 1, tj + 1))
    for k in range(1, tj + 1):
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    return [(Jp + Jp.T) / 2, (Jp - Jp.T) / 2j, np.diag(m)]


def brute(tj):
    dims = [t + 1 for t in tj]
    ops = []
    for i, t in enumerate(tj):
        Sm = spin(t)
        L = []
        for a in range(3):
            M = np.array([[1.0 + 0j]])
            for k, dk in enumerate(dims):
                M = np.kron(M, Sm[a] if k == i else np.eye(dk))
            L.append(M)
        ops.append(L)
    tot = [sum(o[a] for o in ops) for a in range(3)]
    C2 = sum(t @ t for t in tot)
    w, v = np.linalg.eigh(C2)
    V = v[:, np.abs(w) < 1e-8]
    if V.shape[1] == 0:
        return np.array([])
    def tp(i, j, k):
        A, B, C = ops[i], ops[j], ops[k]
        return (A[0] @ (B[1] @ C[2] - B[2] @ C[1]) + A[1] @ (B[2] @ C[0] - B[0] @ C[2])
                + A[2] @ (B[0] @ C[1] - B[1] @ C[0]))
    T = tp(0, 1, 2) + tp(0, 3, 4)
    Tr = V.conj().T @ T @ V
    return np.sort(np.abs(np.linalg.eigvals(Tr)))


if __name__ == '__main__':
    Mb, Mx = int(sys.argv[1]), int(sys.argv[2])
    worst = 0; n = 0
    for tj in itertools.product(range(1, Mb + 1), repeat=5):
        if sum(tj) % 2:
            continue
        e1 = brute(tj)
        S, W = matrix(tj, 'T')
        e2 = np.sort(np.abs(np.linalg.eigvalsh(W))) if len(S) else np.array([])
        assert len(e1) == len(e2), (tj, len(e1), len(e2))
        if len(e1):
            worst = max(worst, np.abs(e1 - e2).max())
        n += 1
    print(f"brute force vs closed form: {n} tuples with 2j_i<={Mb}, max |spec| difference {worst:.2e}")
    # exact congruence  x^2 = h^2 F G / 16  and sigma construction: checked by construction in links();
    # here verify that the 'T' links equal (1/16) * K-link * sigma*sigma' with sigma^2 = phi(d) omega(u).
    bad = 0; n = 0
    for tj in itertools.product(range(1, Mx + 1), repeat=5):
        if sum(tj) % 2 or tj[1] > tj[2] or tj[3] > tj[4]:
            continue
        s = Fraction(tj[0], 2)
        S, LT = links(tj, 'T'); _, LK = links(tj, 'K')
        if not S:
            continue
        ds = sorted({Fraction(B - C, 2) for B, C in S}); us = sorted({Fraction(B + C, 2) for B, C in S})
        phi = {ds[0]: Fraction(1)}
        for d in ds[1:]:
            phi[d] = F(s, d - Fraction(1, 2)) / phi[d - 1]
        om = {us[0]: Fraction(1)}
        for u in us[1:]:
            om[u] = G(s, u) / om[u - 1]
        sig2 = [phi[Fraction(B - C, 2)] * om[Fraction(B + C, 2)] for B, C in S]
        for (i, k), w in LT.items():
            if 16 * w != LK[(i, k)] * sig2[i] * sig2[k]:
                bad += 1
        n += 1
    print(f"exact congruence x^2 = h^2 sigma^2 sigma'^2/16 on {n} tuples 2j<={Mx}: {bad} failures")
    # known values
    for tj, ref in [((3, 2, 3, 3, 3), 8.86723463711174), ((1, 1, 2, 2, 2), 4.557960739249212)]:
        print(tj, 'q =', 8 * lam(tj), 'ref', ref)

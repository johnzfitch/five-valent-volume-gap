"""Exact bounded-leg slope (Theorem 7.3, Appendix G), revision pass 2b, 22 October 2026.
Independent of the limit-certificate code: reads only the data in ../outputs/limit_cert_K{2,4,6}.json.
(1) compares every certified lower bound sigma_inf with nu0 = sqrt((5-3 sqrt2)/2) in exact arithmetic
    (x >= nu0  <=>  5-2x^2 <= 0 or (5-2x^2)^2 <= 18) and lists the cases below nu0;
(2) recomputes C_K from the data, the largest s0, max 8E(s0), and checks 8 nu0 max(s0) < C_K exactly;
(3) for each case below nu0 builds the exact limit matrix from the link formulas of Lemma G.1
    (X: (1/2) h1(b) sqrt(b^2-g^2), D: (1/2) sqrt(kD)), with its own code, and prints its characteristic
    polynomial and the singular values of the bipartite block Y_inf (sympy, exact);
(4) prints the characteristic polynomial of U_z on the weight +-1/2 spaces of all orderings of (1/2,1,1),
    built from spin matrices.
Usage (from this directory): python lj_exact_slope_check.py > ../outputs/lj_exact_slope_check_out.txt"""
import json, sys
from fractions import Fraction as Fr
import sympy as sp

import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "limit_cert_K%d.json")
QSTAR = Fr(45579607, 10**7)
nu0 = sp.sqrt((5 - 3*sp.sqrt(2))/2)

def ge_nu0(x):  # exact: x >= nu0 for rational x > 0
    a = 5 - 2*x*x          # x>=nu0 <=> 2x^2-5 >= -3sqrt2 <=> a <= 3 sqrt2
    return a <= 0 or a*a <= 18

def limit_matrix(t2, M2):
    j2, j3, j4 = (sp.Rational(x, 2) for x in t2); M = sp.Rational(M2, 2)
    n1 = j2 + j3 + 1; m1 = abs(j2 - j3)
    def h1sq(b): return (n1**2 - b**2)*(b**2 - m1**2)/(4*b**2 - 1)
    S = []
    b = m1
    while b <= j2 + j3:
        g = -M - j4
        while g <= j4 - M:
            if abs(g) <= b: S.append((b, g))
            g += 1
        b += 1
    idx = {p: i for i, p in enumerate(S)}
    A = sp.zeros(len(S))
    for (b, g), i in idx.items():
        if (b - 1, g) in idx:      # X-link, limit (1/2) h1(b) sqrt(b^2-g^2)
            w = sp.sqrt(h1sq(b)*(b**2 - g**2))/2
            A[idx[(b-1, g)], i] = A[i, idx[(b-1, g)]] = w
        if (b, g - 1) in idx:      # D-link, limit (1/2) sqrt(kD)
            kD = (j4 + 1 - M - g)*(g + M + j4)*(b - g + 1)*(b + g)
            if kD > 0:
                w = sp.sqrt(kD)/2
                A[idx[(b, g-1)], i] = A[i, idx[(b, g-1)]] = w
    col = [int((b + g) % 2) for (b, g) in S]
    return S, A, col

x = sp.symbols('x')
for K in (2, 4, 6):
    cases = json.load(open(P % K))
    exc = [c for c in cases if not ge_nu0(Fr(c['sigma_inf']))]
    others = [Fr(c['sigma_inf']) for c in cases if ge_nu0(Fr(c['sigma_inf']))]
    maxs0 = max(Fr(c['s0']) for c in cases)
    max8E = max(8*Fr(c['E_s0']) for c in cases)
    sigK = min(Fr(c['sigma_inf']) for c in cases)
    CK = max(max8E, max(8*sigK*Fr(c['s0']) - QSTAR for c in cases)); CK = int(CK) + 1
    lin_at_maxs0 = 8*nu0*sp.Rational(maxs0.numerator, maxs0.denominator) - CK
    print(f"K={K}: cases {len(cases)}, exceptions {len(exc)}, min other sigma {float(min(others)):.7f}, "
          f"max s0 {float(maxs0)}, max 8E(s0) {float(max8E):.3f}, C_K recomputed {CK}, "
          f"8nu0*maxs0 - C_K = {float(lin_at_maxs0):.2f} (negative: {bool(lin_at_maxs0 < 0)})")
    for c in exc:
        t2 = tuple(c['assign']); M2 = c['M2']
        S, A, col = limit_matrix(t2, M2)
        cp = sp.factor(A.charpoly(x).as_expr())
        n0 = col.count(0); n1 = col.count(1)
        small = [i for i in range(len(S)) if col[i] == (0 if n0 <= n1 else 1)]
        large = [i for i in range(len(S)) if i not in small]
        Y = A.extract(large, small)
        ev = sp.Matrix(Y.T*Y).eigenvals()
        svals = sorted(sp.nsimplify(sp.sqrt(e)) for e in ev)
        print(f"   exception t={c['t']} assign={t2} M0={sp.Rational(M2,2)} sigma_inf={c['sigma_inf']!r} s0={c['s0']} "
              f"sites={S} Y {Y.shape} charpoly {cp}")
        print(f"     singular values of Y_inf: {[sp.N(v, 15) for v in svals]}; smin - nu0 = {sp.simplify(min(svals, key=lambda v: sp.N(v)) - nu0)}")
print("nu0 =", sp.N(nu0, 20), " 8nu0 =", sp.N(8*nu0, 20), " 4sqrt(10-6sqrt2) =", sp.N(4*sp.sqrt(10-6*sp.sqrt(2)), 20))
# also U_z on (1/2 x 1 x 1) weight +-1/2 for all orderings, my own construction
def jmats(j):
    j = sp.Rational(j); d = int(2*j + 1); ms = [j - k for k in range(d)]
    Jz = sp.diag(*ms); Jp = sp.zeros(d)
    for k in range(1, d):
        m = ms[k]; Jp[k-1, k] = sp.sqrt(j*(j+1) - m*(m+1))
    Jm = Jp.T; Jx = (Jp + Jm)/2; Jy = (Jp - Jm)/(2*sp.I)
    return [Jx, Jy, Jz]
from sympy import kronecker_product as kp, eye
def op(js, which, comp):
    mats = [eye(int(2*sp.Rational(j)+1)) for j in js]
    mats[which] = jmats(js[which])[comp]
    out = mats[0]
    for m in mats[1:]: out = kp(out, m)
    return out
for order in [(sp.Rational(1,2),1,1),(1,sp.Rational(1,2),1),(1,1,sp.Rational(1,2))]:
    J = [[op(order, a, c) for c in range(3)] for a in range(3)]
    def cross_z(a, b): return J[a][0]*J[b][1] - J[a][1]*J[b][0]
    Uz = cross_z(0,1) + cross_z(0,2) + cross_z(1,2)
    Gz = J[0][2] + J[1][2] + J[2][2]
    for w in (sp.Rational(1,2), -sp.Rational(1,2)):
        ids = [i for i in range(Gz.shape[0]) if Gz[i,i] == w]
        sub = Uz.extract(ids, ids)
        print("U_z order", order, "weight", w, "charpoly", sp.factor(sub.charpoly(x).as_expr()))

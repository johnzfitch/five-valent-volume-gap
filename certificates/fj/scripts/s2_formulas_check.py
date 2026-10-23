"""Exact check (rationals) of the j1 = 2 chain formulas against the general link formula (fj_stage.link2):
W- links q_b^2, p_b^2 ; M+ links alpha_e^2, beta_e^2 ; the Gram identity M-^T M- = 3/2 W-^T W- + diag(Lambda) with the
closed form of Lambda.  All W2-type (integer n, m) configurations with n1, n2 <= 16.
Usage: python s2_formulas_check.py > ../outputs/s2_formulas_check_out.txt"""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import fj_stage as St
def P(n, m, L): return Fr((n * n - L * L) * (L * L - m * m))
cnt = 0; bad = 0
for n1 in range(3, 17):
    for m1 in range(0, n1 - 2):
        for n2 in range(3, 17):
            for m2 in range(0, n2 - 2):
                G = St.Geometry(4, 2 * n1, 2 * m1, 2 * n2, 2 * m2)
                if not G.sites: continue
                w = lambda r, c: Fr(int(G.w2(r, c).p), int(G.w2(r, c).q))
                # W-: rows (b,b+2) doubled (2b,2b+4); cols (b,b+1)
                for (B, C) in G.diags.get(-4, []):
                    b = B // 2
                    for (cb, cc_), val in (((B, C - 2), P(n2, m2, b + 2) * b / (2 * (2 * b + 3))),
                                           ((B + 2, C), P(n1, m1, b + 1) * (b + 3) / (2 * (2 * b + 3)))):
                        if (cb, cc_) in G.sites:
                            cnt += 1; bad += (w((B, C), (cb, cc_)) != val)
                # M+: rows (e,e) ; cols (e+1,e) [beta] and (e, e-1) [alpha]
                for (B, C) in G.diags.get(0, []):
                    e = B // 2
                    a2 = Fr(3, 4) * P(n2, m2, e) * (e - 1) * (2 * e + 3) / ((2 * e - 1) * (2 * e + 1))
                    b2 = Fr(3, 4) * P(n1, m1, e + 1) * (e + 2) * (2 * e - 1) / ((2 * e + 1) * (2 * e + 3))
                    for (cb, cc_), val in (((B, C - 2), a2), ((B + 2, C), b2)):
                        if (cb, cc_) in G.sites:
                            cnt += 1; bad += (w((B, C), (cb, cc_)) != val)
                # Lambda identity on diag -1 columns
                cols, dW, _ = G.gram(-2, -4); _, dM, _ = G.gram(-2, 0)
                for (B, C), a, m in zip(cols, dW, dM):
                    b = B // 2
                    K = b * (2 * b * b + 6 * b + 5)
                    N = (n1**2 + m1**2 - n2**2 - m2**2) * K + 2 * n1**2 * m1**2 * (b + 3) - 2 * n2**2 * m2**2 * (b - 1) \
                        - 2 * (n2**2 + m2**2) + 2 * (2 * b * b + 4 * b + 1)**2
                    lam = Fr(3 * N, 4 * (2 * b + 1) * (2 * b + 3))
                    got = Fr(int(m.p), int(m.q)) - Fr(3, 2) * Fr(int(a.p), int(a.q))
                    cnt += 1; bad += (got != lam)
print(f"{cnt} exact comparisons (W- links, M+ links, Lambda on diagonal -1), {bad} mismatches")

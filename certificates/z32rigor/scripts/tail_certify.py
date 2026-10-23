"""Rigorous tail certificate (THEOREM.md section 3.9).
For each regime the majorant  C(n) = s* [F0 + (7/3) F1 + W F1]  (sup over the second parameter already taken in the
derivation) is written as a finite sum of monomials  K * L^p * prod_i 1/(n + g_i),  K > 0, L = 1 + ln n,
with at least one factor.  Then:
  (i)  C(n0) <= 1 is evaluated in mpmath interval arithmetic (upper end);
  (ii) every monomial is nonincreasing on [n0, oo): its log-derivative is (p/L - sum_i n/(n+g_i))/n < 0, which holds at
       n0 if p/L(n0) < sum_i 1/(1 + max(g_i,0)/n0); the left side decreases and the right side increases in n.
C(n) <= 1 gives sigma^2 >= s*: with theta = 1 - s* F0,  (1 - s*F0)(1 - s*F1) >= 1 - s*F0 - s*F1 >= s*(4/3 + W) F1,
which is Corollary 5's second condition (THEOREM.md section 3.1).
Usage: python tail_certify.py > ../outputs/tail_certificate_out.txt"""
from mpmath import iv, mpf
iv.prec = 120
S = (iv.mpf(37) / 80) ** 2                          # s* = (3.7/8)^2
PI2_6 = iv.pi ** 2 / 6
LN2 = iv.log(2)
def R(x): return iv.mpf(x)

def terms(name):
    """list of (K, p, [g_i]) : K * L^p / prod (n + g_i)."""
    T = []
    def F1F0W(F0, F1, W):
        # C/s* = F0 + (7/3) F1 + W F1   (products of monomial lists)
        out = list(F0) + [(R(7) / 3 * k, p, g) for (k, p, g) in F1]
        for (kw, pw, gw) in W:
            for (k1, p1, g1) in F1:
                out.append((kw * k1, pw + p1, gw + g1))
        return out
    if name == 'R2-':
        F0 = [(R(32) / 3, 0, [0])]
        F1 = [(R(12), 0, [0]), (R(32) / 3, 1, [0, 2])]
        W = [(R(8), 2, []), (R(12), 1, []), (R(400) * R(1.5), 0, [-10]), (R(400), 1, [-10])]   # 8(1.5+L)L + 400(1.5+L)/(n-10)
        return F1F0W(F0, F1, W)
    if name == 'R2+':
        F0 = [(R(16), 0, [0])]
        F1 = [(R(32) / 3, 0, [0]), (R(32) / 3, 1, [0, 2])]
        W = [(R(16), 2, []), (R(2) / 3, 1, [])]
        return F1F0W(F0, F1, W)
    if name == 'R1':
        # F0 + (7/3) F1max + (W F1 bounded via a S(2a-3) <= 3 + (1/2) ln 2n = (5/2 + ln2/2) + L/2)
        out = [(R(4.22), 0, [0]), (R(7) / 3 * R(16) / 3 * PI2_6, 0, [0])]
        c0 = R(5) / 2 + LN2 / 2
        poly = [(R(2.67), 2), (R(12.6), 1), (R(9.3), 0)]
        for (k, p) in poly:
            out.append((R(16) / 3 * c0 * k, p, [0]))
            out.append((R(16) / 3 * R(1) / 2 * k, p + 1, [0]))
        return out
    if name == 'R4':
        F0 = [(R(16) / 3 * PI2_6, 0, [0]), (R(64) / 3, 1, [0, 3])]
        F1 = [(R(3.52), 0, [0]), (R(32) / 3, 1, [0, 2])]
        return F1F0W(F0, F1, [])
    if name == 'R4c':
        F0 = [(R(16) / 3 * PI2_6, 0, [0]), (R(32) / 3, 1, [0, 2])]
        F1 = [(R(16) * (PI2_6 - 1), 0, [0]), (R(32) / 3, 1, [0, 3])]
        return F1F0W(F0, F1, [])
    if name == 'R1c':
        # F0 = (16/3)(L + ln2)(1 + 2L)/n^2 ; F1 = 32 L^2/(n(n+1)) ; W = (16/225)(L + ln2)(1 + 2L)
        q = [(LN2, 0), (2 * LN2 + 1, 1), (R(2), 2)]             # (L+ln2)(1+2L) = ln2 + (2ln2+1)L + 2L^2
        F0 = [(R(16) / 3 * k, p, [0, 0]) for (k, p) in q]
        F1 = [(R(32), 2, [0, 1])]
        W = [(R(16) / 225 * k, p, []) for (k, p) in q]
        return F1F0W(F0, F1, W)
    if name == 'Tm':
        F0 = [(R(32) / 3, 2, [0, 0])]
        F1 = [(R(16) * (PI2_6 - 1), 0, [0]), (R(32) / 3, 1, [0, 1])]
        return F1F0W(F0, F1, [(R(6), 2, [])])
    if name == 'Tp':
        F0 = [(R(64) / 3, 2, [0, 0])]
        F1 = [(R(16) * (PI2_6 - 1), 0, [0]), (R(32) / 3, 1, [0, 2])]
        return F1F0W(F0, F1, [(R(6), 2, [])])

N0 = {'R1': 4000, 'R2+': 4000, 'R2-': 4000, 'R4': 4000, 'R1c': 100000, 'R4c': 100000, 'Tp': 100000, 'Tm': 100000}
if __name__ == '__main__':
    allok = True
    print("C(n) = s*[F0 + (7/3)F1 + W F1] must be <= 1 at the box edge n0 and nonincreasing beyond it")
    for name, n0 in N0.items():
        n = R(n0); Lv = 1 + iv.log(n)
        C = R(0); mono = True
        for (k, p, g) in terms(name):
            assert len(g) >= 1 and k.a > 0
            val = k * Lv ** p
            for gi in g: val = val / (n + gi)
            C += S * val
            rhs = sum(1 / (1 + max(gi, 0) / mpf(n0)) for gi in g)
            if not (R(p) / Lv).b < rhs: mono = False
        ok = (C.b <= 1) and mono
        allok &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:4s} n0 = {n0}: C(n0) <= {float(C.b):.6f}   all {len(terms(name))} monomials nonincreasing: {mono}")
    print("ALL PASS" if allok else "SOME FAIL")

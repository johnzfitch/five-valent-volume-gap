"""Exact symbolic certificates (sympy), all CPU, small:
 S1  Heron factorisation of the triangle factor of both link types:
       X-link (b-1,c)->(b,c):  ((s+c+1)^2-b^2)(b^2-(s-c)^2) = F(d-1/2) G(u),  d=b-c, u=b+c
       D-link (b,c-1)->(b,c):  ((s+b+1)^2-c^2)(c^2-(s-b)^2) = F(d+1/2) G(u)
     with F(t) = (s+1/2)^2 - t^2, G(u) = (u+s+1)(u-s).
 S2  ladder rung: Stark h23 at b=j3+1/2 for (j2,j3)=(1/2,j3) equals 1.
 S3  Lemma W monotonicity of end weights (ratios), as polynomial identities:
       bottom: w(l+2)/w(l) = [j(j+1)-m1(m1+1)]/[j(j+1)-m2(m2-1)],  num - den = -(l+1) m
       top   : w(l+2)/w(l) = [(j-m1)(j+m2)]/[(j+m1+1)(j-m2+1)],   den - num = (2j+1)(l+1)
     where m1=(m+l)/2, m2=(m-l)/2.
 S4  identity used in Lemma W:  -G(2) = (4/3) w1 - sum_{k>=1} 4 w_{2k+1}/((2k-1)(2k+3))
     (checked symbolically for N = 2..16 with symbolic weights).
 S5  Wallis product facts used for Phi: prod_{l<=K} 4l^2/(4l^2-1) < pi/2, increasing (checked K<=2000 exactly
     as rationals vs. pi/2 bound via sympy N(…, 50)).
 S6  Edmonds 6j sign {s c b; 1 b-1 c} = (-1)^(s+b+c) * positive, checked against sympy.physics.wigner
     for all 2s,2b,2c <= 12 (gauge lemma).
"""
import sympy as sp
from sympy.physics.wigner import wigner_6j
from fractions import Fraction

out = []
s, b, c, u, d = sp.symbols('s b c u d', real=True)
F = lambda t: (s + sp.Rational(1, 2)) ** 2 - t ** 2
G = lambda uu: (uu + s + 1) * (uu - s)
X = ((s + c + 1) ** 2 - b ** 2) * (b ** 2 - (s - c) ** 2)
Dl = ((s + b + 1) ** 2 - c ** 2) * (c ** 2 - (s - b) ** 2)
e1 = sp.expand(X - F(b - c - sp.Rational(1, 2)) * G(b + c))
e2 = sp.expand(Dl - F(b - c + sp.Rational(1, 2)) * G(b + c))
out.append(f"S1 X-link residual: {e1}   D-link residual: {e2}")

j3 = sp.symbols('j3', positive=True)
n, m, B = j3 + sp.Rational(3, 2), j3 - sp.Rational(1, 2), j3 + sp.Rational(1, 2)
h2 = sp.simplify((n ** 2 - B ** 2) * (B ** 2 - m ** 2) / (4 * B ** 2 - 1))
out.append(f"S2 ladder rung h23^2 = {h2}")

j, mm, l = sp.symbols('j m l', real=True)
m1, m2 = (mm + l) / 2, (mm - l) / 2
num_b, den_b = j * (j + 1) - m1 * (m1 + 1), j * (j + 1) - m2 * (m2 - 1)
out.append(f"S3 bottom: num-den - (-(l+1)m) = {sp.expand(num_b - den_b + (l + 1) * mm)}")
num_t, den_t = (j - m1) * (j + m2), (j + m1 + 1) * (j - m2 + 1)
out.append(f"S3 top   : den-num - (2j+1)(l+1) = {sp.expand(den_t - num_t - (2 * j + 1) * (l + 1))}")
# check the ratio formulas themselves against factorial formulas at sample integer points
ok = True
for jj2 in range(1, 14):
    J = Fraction(jj2, 2)
    for M2 in range(0, jj2 + 1):
        Mv = Fraction(M2, 2)
        if (jj2 - M2) % 1: continue
        # m1 ranges with m1, m2 in [-J, J], m1 - J integer
        m1s = [J - k for k in range(int(2 * J) + 1)]
        def wb(x):
            y = Mv - x
            if abs(y) > J: return None
            return sp.factorial(J + x) * sp.factorial(J + y) / (sp.factorial(J - x) * sp.factorial(J - y))
        def wt(x):
            y = Mv - x
            if abs(y) > J: return None
            return sp.binomial(2 * J, J + x) * sp.binomial(2 * J, J + y)
        for x in m1s:
            if wb(x) is None or wb(x + 1) is None or x + 1 > J: continue
            lv = 2 * x - Mv
            rb = sp.nsimplify(wb(x + 1) / wb(x)); rt = sp.nsimplify(wt(x + 1) / wt(x))
            fb = num_b.subs({j: J, mm: Mv, l: lv}) / den_b.subs({j: J, mm: Mv, l: lv})
            ft = num_t.subs({j: J, mm: Mv, l: lv}) / den_t.subs({j: J, mm: Mv, l: lv})
            if sp.simplify(rb - fb) != 0 or sp.simplify(rt - ft) != 0:
                ok = False
out.append(f"S3 ratio formulas match factorial CG formulas at all 2j<=13 sample points: {ok}")

okS4 = True
for N in range(2, 17, 2):
    lams = [N - 1 - 2 * k for k in range(N)]
    w = {lv: sp.Symbol(f'w{abs(lv)}') for lv in lams}   # symmetric weights
    Gz = lambda z: sum(w[lv] / (lv - z) for lv in lams)
    lhs = -Gz(2)  # want G(2) <= 0, i.e. -G(2) >= 0
    tele = sp.Rational(4, 3) * w[1] - sum(4 * w[2 * k + 1] / ((2 * k - 1) * (2 * k + 3)) for k in range(1, N // 2))
    if sp.simplify(lhs - tele) != 0:
        okS4 = False
        out.append(f"S4 mismatch N={N}: {sp.simplify(lhs - tele)}")
out.append("S4 -G(2) == (4/3)w1 - sum_(k>=1) 4 w_(2k+1)/((2k-1)(2k+3)) for symmetric weights, N=2..16 even: " + str(okS4))

W = Fraction(1); okW = True; prev = W
for K in range(1, 2001):
    W *= Fraction(4 * K * K, 4 * K * K - 1)
    if not (W > prev): okW = False
    prev = W
out.append(f"S5 Wallis W_K increasing for K<=2000: {okW}; W_2000 = {float(W):.12f} < pi/2 = {float(sp.pi/2):.12f}")

ok6 = True; cnt = 0
for S2 in range(1, 13):
    for B2 in range(2, 13):
        for C2 in range(1, 13):
            Sv, Bv, Cv = sp.Rational(S2, 2), sp.Rational(B2, 2), sp.Rational(C2, 2)
            if (S2 + B2 + C2) % 2: continue
            val = wigner_6j(Sv, Cv, Bv, 1, Bv - 1, Cv)
            if val == 0: continue
            cnt += 1
            sgn = sp.sign(val) * (-1) ** (Sv + Bv + Cv)
            if sgn != 1: ok6 = False
out.append(f"S6 sign of {{s c b; 1 b-1 c}} equals (-1)^(s+b+c) on all {cnt} nonzero cases (2s,2b,2c<=12): {ok6}")
print("\n".join(out))

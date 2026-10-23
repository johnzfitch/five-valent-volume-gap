"""Symbolic certificates (sympy, exact) for the pointwise inequalities used in the j1 = 3/2 proof.

Method: each claim  E >= 0  is a rational function of the parameters.  After substituting
parameters by  (regime base point) + (nonnegative shift variables), we expand numerator and
denominator as polynomials in the shift variables and check that every coefficient of both has the
same sign (all >= 0, or the numerator all >= 0 and the denominator all > 0).  This proves E >= 0 for
all real nonnegative shifts, a superset of the half-integer lattice actually needed.

Claims (labels as in THEOREM.md section 6):
  S1  Lambda <= 0 on every zz0 column        (R4, R4c)
  S2  same-row ratio r strictly decreasing    (tall chains: zz1 in R2+-,R4,R4c,Tp,Tm; zz0 in R4,R4c)
  S3  same-column ratio s <= 1               (CR chains: R1 zz0/zz1, R1c zz0/zz1, R2+- zz0) [worst case]
  S4  same-row ratio r >= 1                  (RC chains: Tp, Tm zz0)
  S5  Lambda^(2) nonincreasing in n2^2 and m2^2 for c >= 1  (justifies the worst-case substitution)
  S6  Lambda <= Lambdahat := m1^2 [(4c+9) n1^2 + c(4c^2+11c+8)] / (4(2c+1)(2c+3))
      whenever n2 >= n1 + 3/2, c >= 1, n1 >= c+1   (used in the closed-form tails)
Usage: python z32_symbolic.py out.txt
"""
import sys
import sympy as sp

R = sp.Rational
t, c, n1, m1, n2, m2 = sp.symbols('t c n1 m1 n2 m2')
a, x, y, z, w = sp.symbols('a x y z w', nonnegative=True)


def P(n, m, L):
    return (n ** 2 - L ** 2) * (L ** 2 - m ** 2)


# links (squared) at row b of zigzag 0 / 1
q0 = lambda b: R(3, 16) * P(n2, m2, b + R(3, 2)) * b / (b + 1)
p0 = lambda b: R(3, 16) * P(n1, m1, b + 1) * (2 * b + 5) / (2 * b + 3)
q1 = lambda b: R(3, 16) * P(n2, m2, b - R(1, 2)) * (b + 1) / b
p1 = lambda b: R(3, 16) * P(n1, m1, b + 1) * (2 * b - 1) / (2 * b + 1)
L1 = P(n1, m1, c + 1) * c * (c + 2) / ((2 * c + 1) * (2 * c + 3)) - P(n1, m1, c) * (2 * c + 3) / (4 * (2 * c + 1))
L2 = P(n2, m2, c + R(1, 2)) * (2 * c + 3) * (2 * c - 1) / (16 * c * (c + 1)) - P(n2, m2, c + R(3, 2)) * c / (4 * (c + 1))
LAM = L1 + L2
LAMHAT = m1 ** 2 * ((4 * c + 9) * n1 ** 2 + c * (4 * c ** 2 + 11 * c + 8)) / (4 * (2 * c + 1) * (2 * c + 3))

LOG = []


def certify(label, expr, subs):
    """Prove expr >= 0 under subs (dict symbol -> expression in nonnegative shifts)."""
    e = sp.together(sp.sympify(expr).subs(subs, simultaneous=True))
    num, den = sp.fraction(e)
    num = sp.expand(num); den = sp.expand(den)
    syms = sorted((num.free_symbols | den.free_symbols), key=str)
    pn = sp.Poly(num, *syms) if syms else None
    pd = sp.Poly(den, *syms) if syms else None
    cn = [co for _, co in pn.terms()] if pn else [num]
    cd = [co for _, co in pd.terms()] if pd else [den]
    ok = (all(co >= 0 for co in cn) and all(co >= 0 for co in cd) and any(co > 0 for co in cd)) or \
         (all(co <= 0 for co in cn) and all(co <= 0 for co in cd) and any(co < 0 for co in cd))
    msg = f"{'PASS' if ok else 'FAIL'}  {label}: vars {[str(s) for s in syms]}, numerator {len(cn)} terms, denominator {len(cd)} terms"
    print(msg); LOG.append(msg)
    return ok


def certify_univariate(label, expr, var, lo):
    """Prove expr >= 0 for var >= lo (univariate rational function): numerator and denominator have no real
    root in (lo, oo) (exact Sturm counting via sympy) and are positive at lo + 1 (and nonnegative at lo)."""
    e = sp.together(sp.sympify(expr))
    num, den = sp.fraction(e)
    ok = True
    for poly, strict in ((sp.Poly(num, var), False), (sp.Poly(den, var), True)):
        roots = [r for r in sp.real_roots(poly) if r > lo]
        v1 = poly.eval(lo + 1); v0 = poly.eval(lo)
        if roots or v1 <= 0 or (v0 < 0) or (strict and v0 == 0):
            ok = False
    msg = f"{'PASS' if ok else 'FAIL'}  {label}: univariate root count on ({lo}, oo)"
    print(msg); LOG.append(msg)
    return ok


def nested(base):
    """apply substitutions in order (later ones may use earlier symbols)"""
    out = {}
    for k, v in base:
        out[k] = sp.sympify(v).subs(out, simultaneous=True)
    return out


if __name__ == '__main__':
    allok = True
    # ---------------- S1: Lambda <= 0 ----------------
    allok &= certify('S1 R4  Lambda<=0 (c in [m2-1/2, n1-1], n2>=n1+3/2, m2>=m1+3/2)', -LAM,
                     nested([(m1, a), (m2, a + R(3, 2) + w), (c, a + R(3, 2) + w - R(1, 2) + x), (n1, c + 1 + y), (n2, n1 + R(3, 2) + z)]))
    allok &= certify('S1 R4c Lambda<=0 (m1=0,m2=1/2, c in [1,n1-1], n2>=n1+3/2)', -LAM,
                     nested([(m1, 0), (m2, R(1, 2)), (c, 1 + x), (n1, c + 1 + y), (n2, n1 + R(3, 2) + z)]))
    # ---------------- S2: monotone same-row ratio in tall chains ----------------
    r0 = p0(t) / q0(t); r1 = p1(t) / q1(t)
    d0 = r0 - r0.subs(t, t + 1); d1 = r1 - r1.subs(t, t + 1)
    allok &= certify('S2 zz1 R2+ (m2=m1+1/2; rows t,t+1 in [m1+2, n1-2])', d1,
                     nested([(m1, R(1, 2) + a), (m2, m1 + R(1, 2)), (t, m1 + 2 + x), (n1, t + 3 + y), (n2, n1 + R(3, 2) + z)]))
    allok &= certify('S2 zz1 R2- (m2=m1-1/2>=1/2; rows t,t+1 in [m1+1, n1-2])', d1,
                     nested([(m1, 1 + a), (m2, m1 - R(1, 2)), (t, m1 + 1 + x), (n1, t + 3 + y), (n2, n1 + R(3, 2) + z)]))
    allok &= certify('S2 zz1 R4  (rows in [m2+3/2, n1-2])', d1,
                     nested([(m1, a), (m2, a + R(3, 2) + w), (t, m2 + R(3, 2) + x), (n1, t + 3 + y), (n2, n1 + R(3, 2) + z)]))
    allok &= certify('S2 zz0 R4  (rows in [m2-1/2, n1-2])', d0,
                     nested([(m1, a), (m2, a + R(3, 2) + w), (t, m2 - R(1, 2) + x), (n1, t + 3 + y), (n2, n1 + R(3, 2) + z)]))
    allok &= certify('S2 zz1 R4c (rows in [2, n1-2])', d1,
                     nested([(m1, 0), (m2, R(1, 2)), (t, 2 + x), (n1, t + 3 + y), (n2, n1 + R(3, 2) + z)]))
    allok &= certify('S2 zz0 R4c (rows in [1, n1-2])', d0,
                     nested([(m1, 0), (m2, R(1, 2)), (t, 1 + x), (n1, t + 3 + y), (n2, n1 + R(3, 2) + z)]))
    allok &= certify('S2 zz1 Tp  (n2=n1+1/2; rows in [2, n1-2])', d1,
                     nested([(m1, 0), (m2, R(1, 2)), (t, 2 + x), (n1, t + 3 + y), (n2, n1 + R(1, 2))]))
    allok &= certify('S2 zz1 Tm  (n2=n1-1/2; rows in [2, n1-2])', d1,
                     nested([(m1, 0), (m2, R(1, 2)), (t, 2 + x), (n1, t + 3 + y), (n2, n1 - R(1, 2))]))
    # ---------------- S3: CR chains, s<=1 at the worst case ----------------
    s0 = p0(t) / q0(t + 1); s1 = p1(t) / q1(t + 1)
    allok &= certify('S3 R1  zz0 s<=1 (worst n2=n1+3/2, m2=m1-3/2; t in [m1, n1-2], m1>=3/2)', 1 - s0,
                     nested([(m1, R(3, 2) + a), (m2, m1 - R(3, 2)), (t, m1 + x), (n1, t + 2 + y), (n2, n1 + R(3, 2))]))
    allok &= certify('S3 R1  zz1 s<=1 (worst n2=n1+3/2, m2=m1-3/2; t in [m1, n1-2])', 1 - s1,
                     nested([(m1, R(3, 2) + a), (m2, m1 - R(3, 2)), (t, m1 + x), (n1, t + 2 + y), (n2, n1 + R(3, 2))]))
    allok &= certify('S3 R1c zz0 s<=1 (m1=1/2,m2=0, worst n2=n1+3/2; t in [1/2, n1-2])', 1 - s0,
                     nested([(m1, R(1, 2)), (m2, 0), (t, R(1, 2) + x), (n1, t + 2 + y), (n2, n1 + R(3, 2))]))
    allok &= certify('S3 R1c zz1 s<=1 (t in [3/2, n1-2])', 1 - s1,
                     nested([(m1, R(1, 2)), (m2, 0), (t, R(3, 2) + x), (n1, t + 2 + y), (n2, n1 + R(3, 2))]))
    allok &= certify('S3 R2+ zz0 s<=1 (m2=m1+1/2, m1>=1/2, worst n2=n1+3/2; t in [m1, n1-2])', 1 - s0,
                     nested([(m1, R(1, 2) + a), (m2, m1 + R(1, 2)), (t, m1 + x), (n1, t + 2 + y), (n2, n1 + R(3, 2))]))
    allok &= certify('S3 R2- zz0 s<=1 (m2=m1-1/2, m1>=1, worst n2=n1+3/2; t in [m1, n1-2])', 1 - s0,
                     nested([(m1, 1 + a), (m2, m1 - R(1, 2)), (t, m1 + x), (n1, t + 2 + y), (n2, n1 + R(3, 2))]))
    # ---------------- S4: RC chains, r>=1 ----------------
    allok &= certify('S4 Tp zz0 r>=1 (n2=n1+1/2; rows t in [1, n1-2])', r0 - 1,
                     nested([(m1, 0), (m2, R(1, 2)), (t, 1 + x), (n1, t + 2 + y), (n2, n1 + R(1, 2))]))
    allok &= certify('S4 Tm zz0 r>=1 (n2=n1-1/2; rows t in [1, n1-3])', r0 - 1,
                     nested([(m1, 0), (m2, R(1, 2)), (t, 1 + x), (n1, t + 3 + y), (n2, n1 - R(1, 2))]))
    # ---------------- S5: monotonicity of Lambda^(2) ----------------
    N2s, M2s = sp.symbols('N2s M2s', nonnegative=True)
    L2s = L2.subs({n2: sp.sqrt(N2s), m2: sp.sqrt(M2s)})
    allok &= certify('S5 dLambda2/d(n2^2) <= 0 for c>=1', -sp.diff(L2s, N2s), nested([(c, 1 + x), (M2s, w), (N2s, z)]))
    allok &= certify('S5 dLambda2/d(m2^2) <= 0 for c>=1 and n2^2 >= 1', -sp.diff(L2s, M2s), nested([(c, 1 + x), (N2s, 1 + z), (M2s, w)]))
    # ---------------- S6: Lambda <= Lambdahat ----------------
    allok &= certify('S6 Lambda <= Lambdahat (c>=1, n1>=c+1, n2=n1+3/2+z, m2>=0, m1>=0)', LAMHAT - LAM,
                     nested([(m1, a), (m2, w), (c, 1 + x), (n1, c + 1 + y), (n2, n1 + R(3, 2) + z)]))
    # ---------------- S7: telescoping ratio bounds (same-column ratios of CR chains) ----------------
    TT = t - m1 + 1
    q0w = lambda b: R(3, 16) * P(n1 + R(3, 2), m1 - R(3, 2), b + R(3, 2)) * b / (b + 1)
    q1w = lambda b: R(3, 16) * P(n1 + R(3, 2), m1 - R(3, 2), b - R(1, 2)) * (b + 1) / b
    allok &= certify('S7 R1 zz0  s <= T/(T+1)  (worst n2=n1+3/2, m2=m1-3/2)', TT / (TT + 1) - p0(t) / q0w(t + 1),
                     nested([(m1, R(3, 2) + a), (t, m1 + x), (n1, t + 2 + y)]))
    allok &= certify('S7 R1 zz1  s <= T/(T+1)', TT / (TT + 1) - p1(t) / q1w(t + 1),
                     nested([(m1, R(3, 2) + a), (t, m1 + x), (n1, t + 2 + y)]))
    q0m = lambda b: R(3, 16) * P(n1 + R(3, 2), m1 - R(1, 2), b + R(3, 2)) * b / (b + 1)
    psi = lambda s_: ((s_ + R(3, 2)) ** 2 - (m1 - R(1, 2)) ** 2) / ((2 * s_ + 3) * (s_ + 1))
    allok &= certify('S7 R2- zz0 s <= T/(T+1) psi(t)/psi(t+1)', TT / (TT + 1) * psi(t) / psi(t + 1) - p0(t) / q0m(t + 1),
                     nested([(m1, 1 + a), (t, m1 + x), (n1, t + 2 + y)]))
    kk = sp.Symbol('kk')
    allok &= certify('S7 R2- psi(c)/psi(k) <= 5/4 for k >= c >= m1 >= 1', 5 * psi(kk) - 4 * psi(t),
                     nested([(m1, 1 + a), (t, m1 + x), (kk, t + y)]))
    q0p = lambda b: R(3, 16) * P(n1 + R(3, 2), m1 + R(1, 2), b + R(3, 2)) * b / (b + 1)
    phi = lambda s_: (s_ + 1 + m1) * (s_ + 2 + m1) / ((2 * s_ + 3) * (s_ + 1))
    allok &= certify('S7 R2+ zz0 s <= T/(T+1) phi(t)/phi(t+1)', TT / (TT + 1) * phi(t) / phi(t + 1) - p0(t) / q0p(t + 1),
                     nested([(m1, R(1, 2) + a), (t, m1 + x), (n1, t + 2 + y)]))
    allok &= certify('S7 R2+ phi nonincreasing', phi(t) - phi(t + 1), nested([(m1, R(1, 2) + a), (t, m1 + x)]))
    allok &= certify('S7 R2+ phi > 1/2', phi(t) - R(1, 2), nested([(m1, R(1, 2) + a), (t, m1 + x)]))
    # ---------------- S8: elementary bounds used for Lambdahat ----------------
    allok &= certify_univariate('S8 (4c+9)/(4(2c+1)(2c+3)) <= 1.016/(4c) for c >= 1',
                     R(1016, 1000) / (4 * c) - (4 * c + 9) / (4 * (2 * c + 1) * (2 * c + 3)), c, 1)
    allok &= certify('S8 c(4c^2+11c+8)/(4(2c+1)(2c+3)) <= (c+1)/4 for c >= 0',
                     (c + 1) / 4 - c * (4 * c ** 2 + 11 * c + 8) / (4 * (2 * c + 1) * (2 * c + 3)), {c: x})
    LOG.append('ALL PASS' if allok else 'SOME FAIL')
    print(LOG[-1])
    if len(sys.argv) > 1:
        open(sys.argv[1], 'w').write('\n'.join(LOG) + '\n')

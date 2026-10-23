"""Exact symbolic certificates (sympy) for the polynomial facts used in the tail lemmas of THEOREM.md section 3.
Method: each claim 'X >= 0 on a domain' is reduced, by substituting nonnegative shift variables for the domain
constraints (e.g. c = a + u, n = c + 1 + v, a = a0 + w), to a polynomial whose coefficients are all >= 0
(after clearing positive denominators).  Identities are checked by simplification to 0.
Usage: python tail_symbolic.py > ../outputs/tail_symbolic_out.txt"""
import sympy as sp
n, a, c, t, u, v, w = sp.symbols('n a c t u v w', nonnegative=True)
H, K = sp.Rational(1, 2), sp.Rational(3, 2)
RES = []

def P(N, M, L): return (N**2 - L**2) * (L**2 - M**2)
def q0(n2, m2, b): return sp.Rational(3, 16) * P(n2, m2, b + K) * b / (b + 1)
def p0(n1, m1, b): return sp.Rational(3, 16) * P(n1, m1, b + 1) * (2*b + 5) / (2*b + 3)
def q1(n2, m2, b): return sp.Rational(3, 16) * P(n2, m2, b - H) * (b + 1) / b
def p1(n1, m1, b): return sp.Rational(3, 16) * P(n1, m1, b + 1) * (2*b - 1) / (2*b + 1)
def Lam(n1, m1, n2, m2, x):
    L1 = P(n1, m1, x + 1) * x * (x + 2) / ((2*x + 1) * (2*x + 3)) - P(n1, m1, x) * (2*x + 3) / (4 * (2*x + 1))
    L2 = P(n2, m2, x + H) * (2*x + 3) * (2*x - 1) / (16 * x * (x + 1)) - P(n2, m2, x + K) * x / (4 * (x + 1))
    return L1 + L2

def nonneg(label, expr, subs, positive_den_ok=True):
    """expr >= 0 after subs (dict of shift substitutions); clear denominators (asserted positive by construction)."""
    e = expr
    for k_, v_ in subs.items(): e = e.subs(k_, v_)          # sequential, in the order given
    e = sp.together(sp.expand(e))
    num, den = sp.fraction(e)
    num = sp.expand(num); den = sp.expand(den)
    pn = sp.Poly(num, *sorted(num.free_symbols, key=str)) if num.free_symbols else None
    pd = sp.Poly(den, *sorted(den.free_symbols, key=str)) if den.free_symbols else None
    okn = (num >= 0) if pn is None else all(co >= 0 for co in pn.coeffs())
    if pd is None: okd = den > 0
    else: okd = all(co >= 0 for co in pd.coeffs()) and pd.eval({s: 0 for s in pd.gens}) > 0 if False else all(co >= 0 for co in pd.coeffs())
    ok = bool(okn and okd)
    RES.append((label, ok)); print(f"{'PASS' if ok else 'FAIL'}  {label}")
    return ok

def ident(label, expr):
    ok = sp.simplify(expr) == 0
    RES.append((label, ok)); print(f"{'PASS' if ok else 'FAIL'}  {label}")
    return ok

def majorize_a_by_c(poly_ac):
    """for 0 <= a <= c: upper bound obtained by replacing a -> c in positive monomials and dropping negative monomials
    that contain a (each step can only increase the value)."""
    p = sp.Poly(sp.expand(poly_ac), a, c); out = 0
    for (i, j), k in p.terms():
        if k > 0: out += k * c**(i + j)
        elif i == 0: out += k * c**j
    return sp.expand(out)

def univariate_nonneg(label, expr, var, x0):
    """expr(var) >= 0 for all real var >= x0: no real root in (x0, oo) (sympy real_roots) and value at x0 >= 0
    or leading behaviour checked; a polynomial with no root in (x0, oo), positive at x0 + 1, is positive there."""
    e = sp.Poly(sp.expand(expr), var)
    roots = [r for r in sp.real_roots(e) if r > x0]
    ok = (len(roots) == 0) and (e.eval(x0) >= 0) and (e.eval(x0 + 1) > 0)
    RES.append((label, ok)); print(f"{'PASS' if ok else 'FAIL'}  {label}")
    return ok

# ---------------- link factorisations (worst-case n2 = n + 3/2 unless stated)
n2 = n + K
ident("R2-: q0(c) = (3/16)(n-c)(n+c+3)(c-a+2)(c+1+a) c/(c+1)", q0(n2, a - H, c) - sp.Rational(3, 16)*(n-c)*(n+c+3)*(c-a+2)*(c+1+a)*c/(c+1))
ident("R2+: q0(c) = (3/16)(n-c)(n+c+3)(c+1-a)(c+2+a) c/(c+1)", q0(n2, a + H, c) - sp.Rational(3, 16)*(n-c)*(n+c+3)*(c+1-a)*(c+2+a)*c/(c+1))
ident("R1:  q0(c) = (3/16)(n-c)(n+c+3)(c+3-a)(c+a) c/(c+1)", q0(n2, a - K, c) - sp.Rational(3, 16)*(n-c)*(n+c+3)*(c+3-a)*(c+a)*c/(c+1))
ident("p0(c) = (3/16)(n-c-1)(n+c+1)(c+1-a)(c+1+a)(2c+5)/(2c+3)", p0(n, a, c) - sp.Rational(3, 16)*(n-c-1)*(n+c+1)*(c+1-a)*(c+1+a)*(2*c+5)/(2*c+3))
ident("R2-: q1(i) = (3/16)(n-i+2)(n+i+1)(i-a)(i+a-1)(i+1)/i", q1(n2, a - H, c) - sp.Rational(3, 16)*(n-c+2)*(n+c+1)*(c-a)*(c+a-1)*(c+1)/c)
ident("R2+: q1(i) = (3/16)(n-i+2)(n+i+1)(i-1-a)(i+a)(i+1)/i", q1(n2, a + H, c) - sp.Rational(3, 16)*(n-c+2)*(n+c+1)*(c-1-a)*(c+a)*(c+1)/c)
ident("R1:  q1(c) = (3/16)(n-c+2)(n+c+1)(c+1-a)(c+a-2)(c+1)/c", q1(n2, a - K, c) - sp.Rational(3, 16)*(n-c+2)*(n+c+1)*(c+1-a)*(c+a-2)*(c+1)/c)

# ---------------- same-column ratios s_t = p0(t)/q0(t+1) (zigzag 0) and s'_t = p1(t)/q1(t+1) (zigzag 1, R1)
T = t - a + 1
nu = (n + t + 1) / (n + t + 4); phi = (2*t + 5) * (t + 2) / ((2*t + 3) * (t + 1))
ident("R1  s_t = T/(T+3) nu phi", p0(n, a, t) / q0(n2, a - K, t + 1) - T/(T+3)*nu*phi)
ident("R2+ s_t = T/(T+1) (t+1+a)/(t+3+a) nu phi", p0(n, a, t) / q0(n2, a + H, t + 1) - T/(T+1)*(t+1+a)/(t+3+a)*nu*phi)
ident("R2- s_t = T/(T+2) (t+1+a)/(t+2+a) nu phi", p0(n, a, t) / q0(n2, a - H, t + 1) - T/(T+2)*(t+1+a)/(t+2+a)*nu*phi)
nup = (n - t - 1) * (n + t + 1) / ((n - t + 1) * (n + t + 2))
ident("R1  s'_t = T/(T+1) (t+1+a)/(t-1+a) (2t-1)(t+1)/((2t+1)(t+2)) nu'", p1(n, a, t) / q1(n2, a - K, t + 1) - T/(T+1)*(t+1+a)/(t-1+a)*(2*t-1)*(t+1)/((2*t+1)*(t+2))*nup)
# s <= 1 on t >= a, t <= n-2 :  t = a+u, n = t+2+v
# nu, nu' <= 1 for t <= n-2 is immediate; the n-free factors are certified <= 1:
nonneg("R1  T/(T+3) phi <= 1  (a >= 3/2, t >= a)", 1 - T/(T+3)*phi, {t: a + u, a: K + w})
nonneg("R2+ T/(T+1) (t+1+a)/(t+3+a) phi <= 1  (a >= 1/2, t >= a)", 1 - T/(T+1)*(t+1+a)/(t+3+a)*phi, {t: a + u, a: H + w})
nonneg("R2- T/(T+2) (t+1+a)/(t+2+a) phi <= 1  (a >= 1, t >= a)", 1 - T/(T+2)*(t+1+a)/(t+2+a)*phi, {t: a + u, a: 1 + w})
nonneg("R1  T/(T+1) (t+1+a)/(t-1+a) (2t-1)(t+1)/((2t+1)(t+2)) <= 1  (a >= 3/2, t >= a)",
       1 - T/(T+1)*(t+1+a)/(t-1+a)*(2*t-1)*(t+1)/((2*t+1)*(t+2)), {t: a + u, a: K + w})
nonneg("nu' <= 1  (t <= n-2)", 1 - nup, {n: t + 2 + v})
# R1c chains (m1 = 1/2, m2 = 0)
ident("R1c s_t = nu (t+1/2)(t+2)/((t+5/2)(t+1))", p0(n, H, t) / q0(n2, 0, t + 1) - nu*(t+H)*(t+2)/((t+sp.Rational(5,2))*(t+1)))
nonneg("R1c (t+1/2)(t+2) <= (t+5/2)(t+1)", (t+sp.Rational(5,2))*(t+1) - (t+H)*(t+2), {t: u})
ident("R1c s'_t = nu' (t+3/2)(t-1/2)(t+1)/((t+1/2)^2 (t+2))", p1(n, H, t) / q1(n2, 0, t + 1) - nup*(t+K)*(t-H)*(t+1)/((t+H)**2*(t+2)))
nonneg("R1c (t+3/2)(t-1/2)(t+1) <= (t+1/2)^2 (t+2)", (t+H)**2*(t+2) - (t+K)*(t-H)*(t+1), {t: u})

# ---------------- Lambda: monotone in m2^2 (c >= 1), worst case m2 = 0
m2s = sp.symbols('m2s', nonnegative=True)
L2part = lambda M2SQ: ((n2**2 - (c+H)**2)*((c+H)**2 - M2SQ)*(2*c+3)*(2*c-1)/(16*c*(c+1)) - (n2**2 - (c+K)**2)*((c+K)**2 - M2SQ)*c/(4*(c+1)))
nonneg("Lambda^2 nonincreasing in m2^2 (c >= 1, c <= n2 - 3/2)", -sp.diff(L2part(m2s), m2s), {n: c + v, c: 1 + u})

den = 16*c*(c+1)*(2*c+1)*(2*c+3)
def num_of(expr): return sp.expand(sp.cancel(expr * den))
# R2-
N2m = num_of(Lam(n, a, n2, a - H, c))
A2 = sp.expand(N2m.coeff(n, 2)); A1 = sp.expand(N2m.coeff(n, 1)); A0 = sp.expand(N2m.coeff(n, 0))
ident("R2- numerator is quadratic in n", N2m - (A2*n**2 + A1*n + A0))
nonneg("R2- A2 <= a^2(32c^2+48c+9) + a(16c^3+20c^2)   (0 <= a <= c)", a**2*(32*c**2+48*c+9) + a*(16*c**3+20*c**2) - A2, {c: a + u})
univariate_nonneg("R2- A1 <= -48 c^5   (0 <= a <= c, c >= 1; after a -> c majorisation)", -48*c**5 - majorize_a_by_c(A1), c, 1)
univariate_nonneg("R2- A0 <= 48c^6 + 479c^5   (0 <= a <= c, c >= 1; after majorisation)", 48*c**6 + 479*c**5 - majorize_a_by_c(A0), c, 1)
# R2+
N2p = num_of(Lam(n, a, n2, a + H, c))
B2 = sp.expand(N2p.coeff(n, 2)); B1 = sp.expand(N2p.coeff(n, 1)); B0 = sp.expand(N2p.coeff(n, 0))
nonneg("R2+ B2 <= a^2(32c^2+48c+9) + a(12c+9)   (0 <= a <= c)", a**2*(32*c**2+48*c+9) + a*(12*c+9) - B2, {c: a + u})
univariate_nonneg("R2+ B1 <= -48 c^5   (0 <= a <= c, c >= 1/2; after majorisation)", -48*c**5 - majorize_a_by_c(B1), c, H)
univariate_nonneg("R2+ B0 <= 32c^6 + 255c^5   (0 <= a <= c, c >= 1; after majorisation)", 32*c**6 + 255*c**5 - majorize_a_by_c(B0), c, 1)
# common denominator facts (c >= 1/2 or >= 1)
nonneg("(32c^2+48c+9) * 2c^2 <= den  (c >= 0)", den - (32*c**2+48*c+9)*2*c**2, {c: u})
nonneg("(16c^3+20c^2) * 4c <= den   (c >= 0)", den - (16*c**3+20*c**2)*4*c, {c: u})
nonneg("(12c+9) * 4c <= den   (c >= 1/2)", den - (12*c+9)*4*c, {c: H + u})
nonneg("den >= 64 c^4", den - 64*c**4, {c: u})
# R2+ column c = 1/2 (a = 1/2, m2 = 1): Lambda_{1/2} <= n^2/16 for n2 = n + 3/2
lam_half = sp.simplify(Lam(n, H, n2, 1, H))
nonneg("R2+ Lambda_{1/2} <= n^2/16  (n >= 0)", n**2/16 - lam_half, {n: u})
# R1 (m2 = 0 worst case)
D1 = 64*c*(c+1)*(2*c+1)*(2*c+3)
N1 = sp.expand(sp.cancel(Lam(n, a, n2, 0, c) * D1))
C2 = sp.expand(N1.coeff(n, 2)); C1 = sp.expand(N1.coeff(n, 1)); C0 = sp.expand(N1.coeff(n, 0))
nonneg("R1 C2 <= a^2(64c^3+208c^2+144c)", a**2*(64*c**3+208*c**2+144*c) - C2, {c: u})
nonneg("R1 (64c^2+208c+144)/(64(c+1)(2c+1)(2c+3)) <= (c+1)/(4c^2)", (c+1)*64*(c+1)*(2*c+1)*(2*c+3) - 4*c**2*(64*c**2+208*c+144), {c: u})
nonneg("R1 C1 <= -(3/4) c D1", -sp.Rational(3, 4)*c*D1 - C1, {c: u})
X_ = sp.expand(C0.coeff(a, 2)); Y_ = sp.expand(C0 - a**2*X_)
ident("R1 C0 = a^2 X(c) + Y(c)", C0 - (a**2*X_ + Y_))
univariate_nonneg("R1 X(c) <= 256 c^4 (c/4 + 2)   (c >= 3/2)", 256*c**4*(c/4 + 2) - X_, c, K)
nonneg("R1 Y(c) <= 256 c^4 (c^2/2 + 27c/16 + 25/16)", 256*c**4*(c**2/2 + sp.Rational(27, 16)*c + sp.Rational(25, 16)) - Y_, {c: u})
nonneg("R1 D1 >= 256 c^4", D1 - 256*c**4, {c: u})
# R1c
N1c = sp.expand(sp.cancel(Lam(n, H, n2, 0, c) * 64*c*(c+1)))
ident("R1c numerator = n^2(4c-3) - n(48c^3+84c^2+24c+9) + 32c^4+48c^3-5c^2-13c-6",
      N1c - (n**2*(4*c-3) - n*(48*c**3+84*c**2+24*c+9) + 32*c**4+48*c**3-5*c**2-13*c-6))
nonneg("R1c Lambda <= n^2(4c-3)/(64c(c+1))  (c >= 1, n >= c+1)", sp.expand(n**2*(4*c-3) - N1c), {n: c + 1 + v, c: 1 + u})
univariate_nonneg("R1c (4c-3)/(c(c+1)) <= 0.84   (c >= 0)", sp.Rational(84, 100)*c*(c+1) - (4*c-3), c, 0)
nonneg("R1c Lambda_{1/2} <= 0  (n >= 1)", -sp.simplify(Lam(n, H, n2, 0, H)), {n: 1 + u})
# Tm / Tp closed forms and bounds (m1 = 0, m2 = 1/2, n2 = n -+ 1/2)
for name, n2f, sgn in (('Tm', n - H, 1), ('Tp', n + H, -1)):
    cf = (32*c**4 + 104*c**3 + 101*c**2 + 25*c + sgn*n*(2*c+1)*(2*c+3)*(4*c+3) - (4*c+9)*n**2) / (16*(2*c+1)*(2*c+3))
    ident(f"{name} Lambda closed form", Lam(n, 0, n2f, H, c) - cf)
    bnd = c**2/2 + c + (n*(4*c+3)/16 if sgn == 1 else 0)
    nonneg(f"{name} Lambda <= c^2/2 + c" + (" + n(4c+3)/16" if sgn == 1 else "") + "   (c >= 0)", bnd - cf, {c: u, n: v})
# Tm / Tp exact products of the RC chain (rows i, columns i+1 matched by p0)
i_ = sp.symbols('i', nonnegative=True)
for name, n2f in (('Tm', n - H), ('Tp', n + H)):
    r = sp.simplify(q0(n2f, H, t) / p0(n, 0, t))
    target = {'Tm': (n-t-2)*t*(t+2)*(2*t+3)/((n-t-1)*(t+1)**2*(2*t+5)),
              'Tp': (n+t+2)*t*(t+2)*(2*t+3)/((n+t+1)*(t+1)**2*(2*t+5))}[name]
    ident(f"{name} RC ratio q0/p0 closed form", r - target)

print("ALL PASS" if all(ok for _, ok in RES) else "SOME FAIL")

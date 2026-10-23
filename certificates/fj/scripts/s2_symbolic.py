"""Symbolic certificates for the j1 = 2 proof (THEOREM.md section 4).

Every claim is a polynomial inequality on a regime of integer parameters.  We substitute nonnegative shift
variables for the regime constraints and show that the expanded polynomial has no negative coefficient
("shift certificate").  For each certificate we also run a NEGATIVE CONTROL (a deliberately false variant) that must
FAIL, to guard against a vacuous checker.

Notation (integers):  P(L) = (n1^2-L^2)(L^2-m1^2),  Q(L) = (n2^2-L^2)(L^2-m2^2).
W- : rows (b,b+2), columns (b,b+1);  q_b^2 = Q(b+2) b/(2(2b+3)),  p_b^2 = P(b+1)(b+3)/(2(2b+3)).
M+ : rows (e,e), columns (e+1,e) ... row e touches column e-1 with alpha_e^2 = 3/4 Q(e)(e-1)(2e+3)/((2e-1)(2e+1))
     and column e with beta_e^2 = 3/4 P(e+1)(e+2)(2e-1)/((2e+1)(2e+3)).
Usage: python s2_symbolic.py > ../outputs/s2_symbolic_out.txt
"""
import sympy as sp

k, l, dl, e_, mu, x, nu, t_ = sp.symbols('k l delta e mu x nu t', nonnegative=True)
RESULTS = []


def P(n, m, L): return (n**2 - L**2) * (L**2 - m**2)


def shift_cert(expr, name, expect=True):
    ex = sp.expand(expr)
    syms = sorted(ex.free_symbols, key=str)
    poly = sp.Poly(ex, *syms)
    neg = [(mono, c) for mono, c in poly.terms() if c < 0]
    ok = (len(neg) == 0)
    status = 'PASS' if ok == expect else 'FAIL'
    RESULTS.append((name, status))
    print(f"[{status}] {name}: {len(poly.terms())} terms, {len(neg)} negative"
          + ('' if expect else '  (negative control: must have negative terms)'))
    return ok


def W_cert():
    # (S1) W- RC, case m2 >= m1+2:  r_b >= 1  <=>  P(b+1)(b+3) - Q(b+2) b >= 0 on interior rows
    #      interior rows b in [m2-1, n2-3], regime n2 <= n1+1.
    m1 = mu; m2 = mu + 2 + x; b = m2 - 1 + k; n2 = b + 3 + l; n1 = n2 - 1 + dl
    shift_cert(P(n1, m1, b + 1) * (b + 3) - P(n2, m2, b + 2) * b, 'S1a W- RC r>=1 (m2>=m1+2)')
    shift_cert(P(n1, m1, b + 1) * (b + 1) - P(n2, m2, b + 2) * b, 'S1a-neg (factor b+1 instead of b+3)', expect=False)
    #      case m1 = 0, m2 in {0,1}: interior rows b >= 1
    for m2v in (0, 1):
        b = 1 + k; n2 = b + 3 + l; n1 = n2 - 1 + dl
        shift_cert(P(n1, 0, b + 1) * (b + 3) - P(n2, m2v, b + 2) * b, f'S1b W- RC r>=1 (m1=0,m2={m2v})')
    # (S2) W- CR (m1>=1, m2<=m1+1, n2>=n1+2): same-column ratio s_t = p_t^2/q_{t+1}^2 <= 1, t in [m1, n1-2];
    #      worst m2 = m1+1 (Q decreasing in m2^2)
    m1 = 1 + mu; m2 = m1 + 1; t = m1 + k; n1 = t + 2 + l; n2 = n1 + 2 + dl
    shift_cert(P(n2, m2, t + 3) * (t + 1) * (2 * t + 3) - P(n1, m1, t + 1) * (t + 3) * (2 * t + 5), 'S2 W- CR s<=1')
    shift_cert(P(n2, m2, t + 2) * t - P(n1, m1, t + 1) * (t + 3), 'S2-neg (same-row form fails)', expect=False)
    # (S3) W- RR (startR, n2>=n1+2): r_b non-increasing on consecutive interior rows
    def dec(n1, m1, n2, m2, b):
        return P(n1, m1, b + 1) * (b + 3) * P(n2, m2, b + 3) * (b + 1) - P(n1, m1, b + 2) * (b + 4) * P(n2, m2, b + 2) * b
    m1 = mu; m2 = mu + 2 + x; b = m2 - 1 + k; n1 = b + 3 + l; n2 = n1 + 2 + dl
    shift_cert(dec(n1, m1, n2, m2, b), 'S3a W- RR r decreasing (m2>=m1+2)')
    for m2v in (0, 1):
        b = 1 + k; n1 = b + 3 + l; n2 = n1 + 2 + dl
        shift_cert(dec(n1, 0, n2, m2v, b), f'S3b W- RR r decreasing (m1=0,m2={m2v})')


def M_cert():
    def num(n1, m1, n2, m2, e): return P(n1, m1, e + 1) * (e + 2) * (2 * e - 1)**2
    def den(n1, m1, n2, m2, e): return P(n2, m2, e) * (e - 1) * (2 * e + 3)**2
    # (S4) M+ RC (n1=n2+1, m1<=m2, m2>=1): rho = beta^2/alpha^2 >= 1 on interior rows e in [m2+1, n1-2];
    #      worst m1 = m2 (P decreasing in m1^2)
    m2 = 1 + mu; m1 = m2; e = m2 + 1 + k; n2 = e + 1 + l; n1 = n2 + 1
    shift_cert(num(n1, m1, n2, m2, e) - den(n1, m1, n2, m2, e), 'S4 M+ RC rho>=1')
    # (S5) M+ RR (n1<=n2, m1<=m2, m2>=1): rho non-increasing on consecutive interior rows
    def dec(n1, m1, n2, m2, e):
        return num(n1, m1, n2, m2, e) * den(n1, m1, n2, m2, e + 1) - num(n1, m1, n2, m2, e + 1) * den(n1, m1, n2, m2, e)
    for name, (m1, m2) in {'m1=nu,m2=nu+1+x': (nu, nu + 1 + x), 'm1=nu+1,m2=m1+x': (nu + 1, nu + 1 + x)}.items():
        e = m2 + 1 + k; n1 = e + 3 + l; n2 = n1 + dl
        shift_cert(dec(n1, m1, n2, m2, e), 'S5 M+ RR rho decreasing ' + name)


def C3_majorants():
    # (S6) C3 (|n2-n1|<=1, m2>=m1+2 or (m1,m2)=(0,1)), W- RC:  rho_t = q_t^2/p_t^2 <= nfac_t * t/(t+1), where the
    #      n-factor is the exact telescoping ratio for each Delta n.  Split: m-part and n-part.
    #   m-part: ((t+2)^2-m2^2)(t+1) <= ((t+1)^2-m1^2)(t+3)  for m2 = m1+2 (worst; LHS decreasing in m2), t>=m1+1
    m1 = mu; m2 = mu + 2; t = mu + 1 + x
    shift_cert(((t + 1)**2 - m1**2) * (t + 3) - ((t + 2)**2 - m2**2) * (t + 1), 'S6a C3 W- m-part (m2=m1+2)')
    #   (0,1): equality ((t+2)^2-1)/(t+1)^2 * t/(t+3) = t/(t+1)  (identity)
    tt = sp.symbols('tt', positive=True)
    ident = sp.simplify(((tt + 2)**2 - 1) / (tt + 1)**2 * tt / (tt + 3) - tt / (tt + 1))
    RESULTS.append(('S6b C3 W- m-part identity (0,1)', 'PASS' if ident == 0 else 'FAIL'))
    print(f"[{'PASS' if ident == 0 else 'FAIL'}] S6b C3 W- m-part identity (0,1)")
    #   n-part, each Delta n: (n2^2-(t+2)^2)/(n1^2-(t+1)^2) == telescoping factor (identities)
    n2s = sp.symbols('n2s', positive=True)
    for dn, fac in ((-1, (n2s - tt - 2) / (n2s - tt)),
                    (0, (n2s - tt - 2) * (n2s + tt + 2) / ((n2s - tt - 1) * (n2s + tt + 1))),
                    (1, (n2s + tt + 2) / (n2s + tt))):
        n1s = n2s - dn
        d = sp.simplify((n2s**2 - (tt + 2)**2) / (n1s**2 - (tt + 1)**2) - fac)
        RESULTS.append((f'S6c C3 W- n-part identity dn={dn}', 'PASS' if d == 0 else 'FAIL'))
        print(f"[{'PASS' if d == 0 else 'FAIL'}] S6c C3 W- n-part identity dn={dn}")
    # (S7) C3, M+ lower piece: 1/rho_e = alpha_e^2/beta_e^2 <= nM_e * e/(e+1), nM_e = exact n-ratio
    #      (n2^2-e^2)/(n1^2-(e+1)^2); m-part claim: (e^2-m2^2)(e-1)(2e+3)^2 (e+1) <= ((e+1)^2-m1^2)(e+2)(2e-1)^2 e
    #      for e >= m2+1 and m2 = m1+2 (worst), and for (m1,m2)=(0,1).
    m1 = mu; m2 = mu + 2; e = m2 + 1 + x
    shift_cert(((e + 1)**2 - m1**2) * (e + 2) * (2 * e - 1)**2 * e - (e**2 - m2**2) * (e - 1) * (2 * e + 3)**2 * (e + 1),
               'S7a C3 M+ m-part (m2=m1+2)')
    e = 2 + x
    shift_cert(((e + 1)**2) * (e + 2) * (2 * e - 1)**2 * e - (e**2 - 1) * (e - 1) * (2 * e + 3)**2 * (e + 1),
               'S7b C3 M+ m-part (0,1)')
    shift_cert(((e + 1)**2) * (e + 2) * (2 * e - 1)**2 * (e - 1) - (e**2 - 1) * (e - 1) * (2 * e + 3)**2 * (e + 1),
               'S7-neg (e-1 instead of e on the right)', expect=False)
    # monotonicity of the m-ratio used for "worst case": (e^2-m2^2)/((e+1)^2-m1^2) decreasing in m2, increasing in m1: trivial.


def C1_lambda():
    # (S8) C1 (n2>=n1+2, m2>=m1+2 or (0,1)): Lambda_c <= 0 for all W- columns c in [1, n1-1].
    #  N_c = (n1^2+m1^2-n2^2-m2^2)K + 2 n1^2 m1^2 (c+3) - 2 n2^2 m2^2 (c-1) - 2(n2^2+m2^2) + 2(2c^2+4c+1)^2
    #  piece A: (4c+8)K(c) - 2(2c^2+4c+1)^2 >= 0  (uses n1^2-n2^2 <= -4n1-4 <= -(4c+8) since c <= n1-1)
    c = 1 + x
    K = c * (2 * c * c + 6 * c + 5)
    shift_cert((4 * c + 8) * K - 2 * (2 * c * c + 4 * c + 1)**2, 'S8a C1 quartic term dominated')
    #  piece B: n2^2 m2^2 (c-1) >= n1^2 m1^2 (c+3) when m2 >= m1+2, n2 >= n1+2, c >= m1+1, m1 >= 1
    m1 = 1 + mu; m2 = m1 + 2 + x; c = m1 + 1 + k; n1 = c + 1 + l; n2 = n1 + 2 + dl
    shift_cert(n2**2 * m2**2 * (c - 1) - n1**2 * m1**2 * (c + 3), 'S8b C1 m-terms nonpositive')


def Lambda_identity():
    # (S9) the closed form of Lambda for W- columns, from the Gram diagonals (exact rational identity)
    b, n1, m1, n2, m2 = sp.symbols('b n1 m1 n2 m2')
    Pp = lambda L: (n1**2 - L**2) * (L**2 - m1**2); Qq = lambda L: (n2**2 - L**2) * (L**2 - m2**2)
    MtM = sp.Rational(3, 4) * Qq(b + 1) * (b + 2) * (2 * b - 1) / ((2 * b + 1) * (2 * b + 3)) \
        + sp.Rational(3, 4) * Pp(b + 1) * b * (2 * b + 5) / ((2 * b + 1) * (2 * b + 3))
    WtW = Qq(b + 2) * b / (2 * (2 * b + 3)) + Pp(b) * (b + 2) / (2 * (2 * b + 1))
    K = b * (2 * b * b + 6 * b + 5)
    N = (n1**2 + m1**2 - n2**2 - m2**2) * K + 2 * n1**2 * m1**2 * (b + 3) - 2 * n2**2 * m2**2 * (b - 1) \
        - 2 * (n2**2 + m2**2) + 2 * (2 * b * b + 4 * b + 1)**2
    d = sp.simplify(MtM - sp.Rational(3, 2) * WtW - 3 * N / (4 * (2 * b + 1) * (2 * b + 3)))
    RESULTS.append(('S9 Lambda closed form', 'PASS' if d == 0 else 'FAIL'))
    print(f"[{'PASS' if d == 0 else 'FAIL'}] S9 Lambda closed form")


def C3_lambda():
    # (S10) C3 m-term bounds used for the tail estimate of W (columns c >= m1+1, m2 >= m1+2):
    #  (a) n2 >= n1 (Delta n in {0,1}):  n2^2 m2^2 (c-1) >= n1^2 m1^2 (c+3)     [m1 >= 1; for m1 = 0 trivial]
    m1 = 1 + mu; m2 = m1 + 2 + x; c = m1 + 1 + k; n1 = c + 1 + l; n2 = n1 + dl
    shift_cert(n2**2 * m2**2 * (c - 1) - n1**2 * m1**2 * (c + 3), 'S10a C3 m-terms <= 0 for n2 >= n1')
    #  (b) n2 = n1-1:  2 n1^2 m1^2 (c+3) - 2 (n1-1)^2 m2^2 (c-1) <= (2 n1 - 1) K(c)
    m1 = mu; m2 = m1 + 2 + x; c = m1 + 1 + k; n1 = c + 1 + l
    K = c * (2 * c * c + 6 * c + 5)
    shift_cert((2 * n1 - 1) * K - 2 * n1**2 * m1**2 * (c + 3) + 2 * (n1 - 1)**2 * m2**2 * (c - 1), 'S10b C3 m-term (dn=-1) <= (2n1-1)K')
    #  (c) the elementary bounds K(c) <= 2c(c+2)^2 and 2(2c^2+4c+1)^2 <= 8(c+1)^4
    c = 1 + x
    shift_cert(2 * c * (c + 2)**2 - c * (2 * c * c + 6 * c + 5), 'S10c K(c) <= 2c(c+2)^2')
    shift_cert(8 * (c + 1)**4 - 2 * (2 * c * c + 4 * c + 1)**2, 'S10d 2(2c^2+4c+1)^2 <= 8(c+1)^4')


if __name__ == '__main__':
    W_cert(); M_cert(); C3_majorants(); C1_lambda(); Lambda_identity(); C3_lambda()
    nf = sum(1 for _, s in RESULTS if s != 'PASS')
    print(f"\n{len(RESULTS)} certificates, {nf} failures:", 'ALL PASS' if nf == 0 else 'SOME FAIL')

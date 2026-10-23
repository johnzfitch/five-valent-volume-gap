"""Referee checks of the analytic lemmas (valence 6). Independent of the author's code.

Parts
 6J   Edmonds' spin-1/2 6j closed forms (as used in the referee's derivation) vs sympy wigner_6j;
      the four alpha/beta closed forms vs exact sympy 6j for all n <= NMAX6J, all e.
 DET  symbolic: |det| of each transfer step, the two-step factor (n+3+i)/(n+5+i), telescoping.
 PHI  rigorous: Phi0^2 = ||M1(0)||_F^2 ||M2(1)||_F^2 <= 900 for all n >= 7 (polynomial positivity).
 CONE endpoint inequalities of Lemma C, cone invariance constants, S bound.
 SIGN random-sign ladders (magnitudes of Prop. L, random signs): odd-n kernel trivial,
      sigma_min*8 <= B(n), at high precision.
 B    B(n) vs exact q_min (read from ref_full_exact_out.txt), elementary bound, asymptotics.
usage: python ref_lemmas.py NMAX6J out.txt
"""
import sys, random, math, re
import sympy as sp
from sympy.physics.wigner import wigner_6j
import mpmath as mp

R = sp.Rational
out = None


def w(s):
    print(s); out.write(s + "\n"); out.flush()


def edmonds_plus(a, b, c):   # {a b c; 1/2 c-1/2 b+1/2}
    s = a+b+c
    return (-1)**s*sp.sqrt((s-2*b)*(s-2*c+1)/((2*b+1)*(2*b+2)*2*c*(2*c+1)))


def edmonds_minus(a, b, c):  # {a b c; 1/2 c-1/2 b-1/2}
    s = a+b+c
    return (-1)**s*sp.sqrt((s+1)*(s-2*a)/(2*b*(2*b+1)*2*c*(2*c+1)))


def part_6j(nmax):
    rnd = random.Random(1)
    bad = 0; cnt = 0
    for _ in range(300):
        a = R(rnd.randint(0, 12), 2); b = R(rnd.randint(1, 12), 2); c = R(rnd.randint(1, 12), 2)
        if not (abs(a-b) <= c <= a+b and (a+b+c).is_integer):
            continue
        for sgn, f in [(1, edmonds_plus), (-1, edmonds_minus)]:
            lo = b + sgn*R(1, 2)
            if lo < 0:
                continue
            ex = wigner_6j(a, b, c, R(1, 2), c-R(1, 2), lo)
            fo = f(a, b, c)
            cnt += 1
            if sp.simplify(ex - fo) != 0:
                bad += 1
    w(f"6J  Edmonds spin-1/2 formulas vs sympy wigner_6j: {cnt} random cases, mismatches {bad}")
    bad = 0; cnt = 0
    for n in range(1, nmax+1):
        cm, cp, g, j6 = R(n-1, 2), R(n+1, 2), R(n, 2), R(n+1, 2)
        for i in range(0, n+1):
            e = i + R(1, 2)
            for c in (cm, cp):
                for h, kind in ((e-R(1, 2), 'L'), (e+R(1, 2), 'R')):
                    if not (abs(c-h) <= j6 <= c+h):
                        continue
                    ex = (2*g+1)*(2*h+1)*wigner_6j(c, R(1, 2), g, e, j6, h)**2
                    if c == cm:
                        cf = R((n+2+i)*i, n*(2*i+2)) if kind == 'L' else R((n-i)*(i+2), n*(2*i+2))
                    else:
                        cf = R(n+1-i, 2*n+4) if kind == 'L' else R(n+3+i, 2*n+4)
                    cnt += 1
                    if sp.nsimplify(ex) != cf:
                        bad += 1
    w(f"6J  alpha/beta closed forms vs exact sympy 6j, n=1..{nmax}, all e: {cnt} entries, mismatches {bad}")


n, i, j = sp.symbols('n i j', positive=True)
am2 = lambda k: (n+2+k)*k/(n*(2*k+2))
bm2 = lambda k: (n-k)*(k+2)/(n*(2*k+2))
ap2 = lambda k: (n+1-k)/(2*n+4)
bp2 = lambda k: (n+3+k)/(2*n+4)
x2 = n*(n+2)/16
y2 = lambda h: (n+1-h)*(n+2+h)*h*(h+1)/16


def part_det():
    # step M1(k): W_k -> U_{k+1}; |A1 B1| ; step M2(k): U_k -> W_{k+1}; |A2 B2|
    d1sq = lambda k: am2(k+1)*bm2(k)/(ap2(k+1)*bm2(k+1))
    d2sq = lambda k: ap2(k+1)*bp2(k)/(am2(k+1)*bp2(k+1))
    two = sp.factor(sp.simplify(d1sq(i)*d2sq(i+1)))
    ok = sp.simplify(two - (n+3+i)/(n+5+i)) == 0
    w(f"DET two-step |det M2(i+1)M1(i)|^2 = {two}; equals (n+3+i)/(n+5+i): {ok}")
    # telescoping, exact for many n
    good = True
    for nn in range(5, 402, 2):
        prod = sp.Integer(1)
        for ii in range(2, nn-2, 2):
            prod *= (d1sq(ii)*d2sq(ii+1)).subs({n: nn, i: ii}) if False else (d1sq(sp.Integer(ii))*d2sq(sp.Integer(ii+1))).subs(n, nn)
        if sp.nsimplify(prod) != R(nn+5, 2*nn+2):
            good = False
    w(f"DET |det T_rest|^2 = (n+5)/(2n+2) exactly for all odd n in [5,401]: {good}")
    tl = sp.factor(sp.simplify(d1sq(0)*d2sq(1)))
    w(f"DET |det T_left|^2 = {tl}")


def part_phi():
    # M1(0): A1 = y1/(x ap(1)), B1 = x am(1) bm(0)/(y1 bm(1)), C1 = y1/(x ap(1) bm(1))
    A1 = y2(1)/(x2*ap2(1)); B1 = x2*am2(1)*bm2(0)/(y2(1)*bm2(1)); C1 = y2(1)/(x2*ap2(1)*bm2(1))
    A2 = y2(2)/(x2*am2(2)); B2 = x2*ap2(2)*bp2(1)/(y2(2)*bp2(2)); C2 = y2(2)/(x2*am2(2)*bp2(2))
    F1 = sp.together(A1+B1+C1); F2 = sp.together(A2+B2+C2)
    Phi2 = sp.factor(sp.together(F1*F2))
    num, den = sp.fraction(sp.together(900 - Phi2))
    m = sp.symbols('m', nonnegative=True)
    pn = sp.Poly(sp.expand(num.subs(n, m+7)), m); pd = sp.Poly(sp.expand(den.subs(n, m+7)), m)
    sgn_n = set(sp.sign(c) for c in pn.all_coeffs()); sgn_d = set(sp.sign(c) for c in pd.all_coeffs())
    w(f"PHI Phi0^2 = {Phi2}")
    w(f"PHI 900 - Phi0^2 = num/den; with n = 7+m, coefficient signs of num: {sgn_n}, of den: {sgn_d} "
      f"(all one sign each => Phi0 <= 30 for every n >= 7)")
    lim = sp.limit(Phi2, n, sp.oo)
    w(f"PHI lim Phi0^2 = {lim} (Phi0 -> {sp.sqrt(lim)} = {float(sp.sqrt(lim)):.4f}); Phi0(7) = {float(sp.sqrt(Phi2.subs(n,7))):.4f}, Phi0(3)={float(sp.sqrt(Phi2.subs(n,3))):.4f}, Phi0(5)={float(sp.sqrt(Phi2.subs(n,5))):.4f}")
    return Phi2


def part_cone():
    r = sp.factor(y2(j)/x2)
    e3 = sp.factor(sp.expand(12*(n-2)*(n+5) - 9*n*(n+2)))
    e4 = sp.factor(sp.expand(2*(n-1)*(2*n+1) - 9*(n+2)))
    w(f"CONE y_j^2/x^2 = {r}; endpoint j=3: 12(n-2)(n+5)-9n(n+2) = {e3}; j=n-1: 2(n-1)(2n+1)-9(n+2) = {e4} (both >= 0 for n >= 4)")
    w("CONE invariance of |P|<=2|r| needs beta+ (1+2e1) <= 2(1-e2-2e1) with e<=1/9: "
      f"{float(1+2/9):.4f} <= {float(2*(1-3/9)):.4f}")
    # S bound and log inequality
    t = sp.symbols('t')
    w(f"CONE ln(1-t)+1.5t at t=1/3: {float(sp.log(sp.Rational(2,3))+0.5):.4f} (>=0, concave => >=0 on [0,1/3])")
    worst = 0; worstS = 0
    for nn in list(range(5, 2002, 2)) + [20001]:
        S = sum(mp.mpf(nn*(nn+2))/(jj*(jj+1)*(nn+1-jj)*(nn+2+jj)) for jj in range(3, nn))
        bound = mp.mpf(1)/3 + 2*(1+mp.log(nn))/(nn+2)
        worst = max(worst, S - bound); worstS = max(worstS, S)
    w(f"CONE S = sum_{{j=3}}^{{n-1}} x^2/y_j^2: max over odd n<=2001 and 20001 = {float(worstS):.4f}; max(S - (1/3+2(1+ln n)/(n+2))) = {float(worst):.4f} (<0 means bound holds)")


def ladder(nn, signs, prec):
    """Prop. L ladder with arbitrary signs, mpmath. returns K (real antisymmetric)."""
    mp.mp.dps = prec
    idx = {}
    for ii in range(1, nn+1): idx[('L', '-', ii)] = len(idx)
    for ii in range(0, nn): idx[('R', '-', ii)] = len(idx)
    for ii in range(0, nn+1): idx[('L', '+', ii)] = len(idx)
    for ii in range(0, nn+1): idx[('R', '+', ii)] = len(idx)
    d = len(idx)
    K = mp.zeros(d, d)
    x = mp.sqrt(nn*(nn+2))/4
    def s(): return signs.choice((-1, 1))
    for ii in range(0, nn+1):
        um = mp.zeros(d, 1); up = mp.zeros(d, 1)
        a2 = mp.mpf((nn+2+ii)*ii)/(nn*(2*ii+2)); b2 = mp.mpf((nn-ii)*(ii+2))/(nn*(2*ii+2))
        if ('L', '-', ii) in idx: um[idx[('L', '-', ii)]] = s()*mp.sqrt(a2)
        if ('R', '-', ii) in idx: um[idx[('R', '-', ii)]] = s()*mp.sqrt(b2)
        up[idx[('L', '+', ii)]] = s()*mp.sqrt(mp.mpf(nn+1-ii)/(2*nn+4))
        up[idx[('R', '+', ii)]] = s()*mp.sqrt(mp.mpf(nn+3+ii)/(2*nn+4))
        K += s()*x*(up*um.T - um*up.T)
    for c in '-+':
        for h in range(1, nn+1):
            y = mp.sqrt(mp.mpf((nn+1-h)*(nn+2+h)*h*(h+1)))/4*s()
            a = idx[('R', c, h-1)]; b = idx[('L', c, h)]
            K[a, b] -= y; K[b, a] += y
    return K


def smin(K):
    Ki = mp.inverse(K)
    v = mp.matrix([1]*K.rows)
    lam = 0
    for _ in range(60):
        v = Ki.T*(Ki*v); nv = mp.norm(v); v = v/nv
        lam = nv
    return 1/mp.sqrt(lam)


def Bn(nn):
    mp.mp.dps = 50
    nn = mp.mpf(nn); N = int(nn)
    x = mp.sqrt(nn*(nn+2))/4
    y = lambda h: mp.sqrt((nn+1-h)*(nn+2+h)*h*(h+1))/4
    am = lambda k: mp.sqrt((nn+2+k)*k/(nn*(2*k+2))); bm = lambda k: mp.sqrt((nn-k)*(k+2)/(nn*(2*k+2)))
    ap = lambda k: mp.sqrt((nn+1-k)/(2*nn+4)); bp = lambda k: mp.sqrt((nn+3+k)/(2*nn+4))
    A1 = y(1)/(x*ap(1)); B1 = x*am(1)*bm(0)/(y(1)*bm(1)); C1 = y(1)/(x*ap(1)*bm(1))
    A2 = y(2)/(x*am(2)); B2 = x*ap(2)*bp(1)/(y(2)*bp(2)); C2 = y(2)/(x*am(2)*bp(2))
    Phi0 = mp.sqrt(A1**2+B1**2+C1**2)*mp.sqrt(A2**2+B2**2+C2**2)
    dT = mp.sqrt((nn+5)/(2*nn+2))
    den = mp.mpf(1)
    for ii in range(2, N-2, 2):
        c1 = y(ii+1)/(x*ap(ii+1)*bm(ii+1)); c2 = y(ii+2)/(x*am(ii+2)*bp(ii+2))
        den *= c1*c2*(1 - x**2/y(ii+2)**2 - 2*x**2/y(ii+1)**2)
    pref = 8*mp.sqrt(3)*(y(N) + mp.sqrt(2)*x**2/y(N))
    B = pref*Phi0*dT/den
    # weaker sign-free version with C1C2 >= y y/x^2 (what the cone lemma literally gives)
    prod = mp.mpf(1)
    for jj in range(3, N):
        prod *= y(jj)/x
    fac = mp.mpf(1)
    for ii in range(2, N-2, 2):
        fac *= (1 - x**2/y(ii+2)**2 - 2*x**2/y(ii+1)**2)
    Bweak = pref*30*dT/(prod*fac)
    B2a = 22700*((nn+1)*mp.sqrt(2*nn)/4 + 1)*mp.sqrt(12)*(nn+2)**((nn-3)/2)/mp.factorial(nn-2)**mp.mpf(1.5)
    corr = y(N) + mp.sqrt(2)*x**2/y(N) - ((nn+1)*mp.sqrt(2*nn)/4 + 1)
    return B, Bweak, B2a, Phi0, corr, prod


def part_sign(nlist, ndraw):
    rnd = random.Random(7)
    for nn in nlist:
        B = Bn(nn)[0]
        res = []
        for dr in range(ndraw):
            prec = int(3*nn*math.log10(nn+1)) + 40
            K = ladder(nn, rnd, prec)
            dt = mp.det(K)
            q = 8*smin(K)
            res.append((dt != 0 and abs(dt) > mp.mpf(10)**(-prec+20), q))
        qs = [float(mp.log10(q)) for _, q in res]
        w(f"SIGN n={nn}: {ndraw} random sign patterns: det K != 0 in all: {all(r[0] for r in res)}; "
          f"log10(8 sigma_min) range [{min(qs):.3f}, {max(qs):.3f}]; log10 B(n) = {float(mp.log10(B)):.3f}; "
          f"all <= B: {all(q <= B for _, q in res)}")


def part_B(path_exact):
    qs = {}
    try:
        for line in open(path_exact, encoding='utf-8'):
            m = re.search(r"sigma-reduced n=(\d+):.*kernel dim=(\d+).*q_min = \[([0-9.e+-]+) \+/-", line)
            if m and m.group(2) == '0':
                qs[int(m.group(1))] = mp.mpf(m.group(3))
    except FileNotFoundError:
        pass
    mp.mp.dps = 50
    rows = []
    for nn in sorted(qs):
        if nn < 5 or nn % 2 == 0:
            continue
        B, Bweak, B2a, Phi0, corr, prod = Bn(nn)
        q = qs[nn]
        nl = nn*mp.log(nn)
        rows.append((nn, q))
        w(f"B   n={nn}: log10 q_min(exact) = {float(mp.log10(q)):9.3f}; log10 B = {float(mp.log10(B)):9.3f}; "
          f"log10 Bweak = {float(mp.log10(Bweak)):9.3f}; log10 B2(as stated) = {float(mp.log10(B2a)):9.3f}; q<=B<=Bweak: {q <= B <= Bweak}; "
          f"(y_n+sqrt2 x^2/y_n)-(y_n+1) = {float(corr):+.3f}; (ln q + n ln n)/n = {float((mp.log(q)+nl)/nn):.4f}; (ln q + 1.5 n ln n)/n = {float((mp.log(q)+1.5*nl)/nn):.4f}")
    # fit ln q = -a n ln n + b n + c ln n + d over odd n >= 15
    pts = [(nn, q) for nn, q in rows if nn >= 15]
    if len(pts) >= 4:
        import numpy as np
        A = np.array([[float(-nn*math.log(nn)), nn, math.log(nn), 1.0] for nn, _ in pts])
        yv = np.array([float(mp.log(q)) for _, q in pts])
        coef = np.linalg.lstsq(A, yv, rcond=None)[0]
        w(f"B   least-squares fit ln q_min = -a n ln n + b n + c ln n + d over odd n in [15,{pts[-1][0]}]: a={coef[0]:.4f} b={coef[1]:.4f} c={coef[2]:.4f} d={coef[3]:.4f}")
    # asymptotics of B itself, analytically: ln prod_{j=3}^{n-1} y_j/x
    for nn in [101, 1001, 10001]:
        mp.mp.dps = 30
        lp = mp.mpf(0)
        for jj in range(3, nn):
            lp += 0.5*mp.log(mp.mpf(jj)*(jj+1)*(nn+1-jj)*(nn+2+jj)/(nn*(nn+2)))
        w(f"B   n={nn}: ln prod_(j=3)^(n-1) y_j/x divided by n ln n = {float(lp/(nn*mp.log(nn))):.4f} (tends to 1, not 3/2)")


def main():
    global out
    nmax6j = int(sys.argv[1]); out = open(sys.argv[2], "w", encoding="utf-8")
    part_6j(nmax6j)
    part_det()
    part_phi()
    part_cone()
    part_sign([7, 9, 11, 13, 15, 21], 6)
    part_B(sys.argv[3] if len(sys.argv) > 3 else "../outputs/ref_full_exact_out.txt")


if __name__ == "__main__":
    main()

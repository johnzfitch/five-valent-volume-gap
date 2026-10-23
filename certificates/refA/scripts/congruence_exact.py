"""Exact (rational) spot check of the Heron-Stark congruence [CL] Theorem 3 and of Lemma 5.

For each tuple (pairing 23|45):
 (T3a) every closed-form link of W satisfies  x^2 == sigma_i^2 sigma_j^2 h(L)^2 / 16  exactly, where h is the Stark link
       of the pair whose label changes (L = upper label), sigma^2 = phi(d) omega(u) built by the recursions of Thm 3;
 (T3b) the links of W are exactly the links of K = H1 (+) H2 with both ends in R (no link of K inside R has zero
       weight, none is missing);
 (L5)  mu := min_{colour 0} sigma^2 * min_{colour 1} sigma^2  >=  (2s+2) Phi_s  (exact), and Phi_s >= 8s(s+1)/pi ... /(2s+2)
       i.e. (2s+2)Phi_s >= 8 s (s+1)/pi checked with a rational upper bound for pi.
Also the polynomial identities P(b;s,c) = F(d-1/2) G(u) (X-link) and P(c;s,b) = F(d+1/2) G(u) (D-link) in sympy.
Coverage: every valid doubled tuple with all entries <= M_ALL, plus N_RAND random tuples with entries <= 38."""
import sys, random, time
from fractions import Fraction as Fr
from refcore import region, links_closed_form, heron_sigma2, stark_h2, Phi_s, colour, valid

PI_UP = Fr(314159266, 100000000)


def check(tj):
    s, (n1, m1), (n2, m2), rect, R = region(tj)
    if len(R) < 2:
        return None
    L = links_closed_form(tj)
    sig2, phi, om = heron_sigma2(tj)
    Rs = set(R)
    # (T3b) K-links inside R
    klinks = set()
    for (b, c) in R:
        if (b - 1, c) in Rs: klinks.add(((b - 1, c), (b, c)))
        if (b, c - 1) in Rs: klinks.add(((b, c - 1), (b, c)))
    assert set(L) == klinks, "link sets differ"
    for (p, q), w2 in L.items():
        assert w2 > 0, ("zero link inside R", tj, p, q)
        if p[1] == q[1]:   # X-link, b changes, upper label q[0]
            h2 = stark_h2(n1, m1, q[0])
        else:
            h2 = stark_h2(n2, m2, q[1])
        assert h2 > 0
        assert w2 == sig2[p] * sig2[q] * h2 / 16, ("congruence fails", tj, p, q)
    cols = {}
    for (b, c) in R:
        cols.setdefault(colour(s, b, c), []).append(sig2[(b, c)])
    if len(cols) < 2:
        return ('onecolour', None)
    mu = min(cols[0]) * min(cols[1])
    bound = (2 * s + 2) * Phi_s(s)
    assert mu >= bound, ("Lemma 5 fails", tj, mu, bound)
    assert bound * PI_UP >= 8 * s * (s + 1), ("Wallis bound fails", s)
    return ('ok', mu / bound)


def sympy_identities():
    import sympy as sp
    b, c, s = sp.symbols('b c s')
    F = lambda t: (s + sp.Rational(1, 2)) ** 2 - t ** 2
    G = lambda u: (u + s + 1) * (u - s)
    d, u = b - c, b + c
    PX = ((s + c + 1) ** 2 - b ** 2) * (b ** 2 - (s - c) ** 2)
    PD = ((s + b + 1) ** 2 - c ** 2) * (c ** 2 - (s - b) ** 2)
    e1 = sp.expand(PX - F(d - sp.Rational(1, 2)) * G(u))
    e2 = sp.expand(PD - F(d + sp.Rational(1, 2)) * G(u))
    return e1 == 0, e2 == 0


if __name__ == '__main__':
    M_ALL = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    N_RAND = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    t0 = time.time()
    i1, i2 = sympy_identities()
    print(f"sympy: P(b;s,c) == F(d-1/2)G(u): {i1};  P(c;s,b) == F(d+1/2)G(u): {i2}")
    n = 0; worst = None
    for S in range(1, M_ALL + 1):
        for A in range(1, M_ALL + 1):
            for B in range(1, M_ALL + 1):
                for C in range(1, M_ALL + 1):
                    for D in range(1, M_ALL + 1):
                        tj = (S, A, B, C, D)
                        if not valid(tj): continue
                        r = check(tj)
                        if r is None: continue
                        n += 1
                        if r[0] == 'ok' and (worst is None or r[1] < worst[0]): worst = (r[1], tj)
    print(f"all valid ordered tuples with doubled spins <= {M_ALL}: {n} nonempty (dim>=2) checked exactly; "
          f"T3a, T3b, Lemma 5 all hold; min mu/((2s+2)Phi_s) = {float(worst[0]):.6f} at {worst[1]}  ({time.time()-t0:.0f}s)")
    random.seed(20261021); m = 0; t1 = time.time(); wr = None
    while m < N_RAND:
        tj = tuple(random.randint(1, 38) for _ in range(5))
        if not valid(tj): continue
        r = check(tj)
        if r is None: continue
        m += 1
        if r[0] == 'ok' and (wr is None or r[1] < wr[0]): wr = (r[1], tj)
    print(f"random tuples with doubled spins <= 38: {m} nonempty checked exactly; all hold; "
          f"min mu/((2s+2)Phi_s) = {float(wr[0]):.6f} at {wr[1]}  ({time.time()-t1:.0f}s)")
    # sharpness of Lemma 5 for the full strip: s integer and half-integer, rectangle large enough that R is the full strip
    for S in (1, 2, 3, 4, 5, 7, 9, 12, 15):
        tj = (S, 40, 40, 40, 40 + S % 2)
        r = check(tj)
        print(f"  large-rectangle tuple {tj}: mu/((2s+2)Phi_s) = {float(r[1]):.6f}")

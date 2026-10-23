"""Checks of [Z] (j1 = 3/2) with independent code.

[A] Exact, against the Racah operator rb_core: every zigzag link of the [Z] Sec. 1 table (q, p, q', p', n+, n-) and the
    coupling identity of Prop. 4,  N^T N = (4/3) D0^T D0 + diag(Lambda),  with Lambda = Lambda1 + Lambda2 in closed form,
    including the signs of the off-diagonal products (flux), on all tuples with legs 2j <= MAXA in the pairing chosen
    by the regime table.
[B] Regime census from rb_core's site sets: regime by the gap rules of Lemma 2, zigzag row/column ranges and chain
    types compared with the table; all zigzag links nonzero (connected paths); B block lower-triangular;
    imbalances in {0,1}.  Legs 2j <= MAXB.
[C] Exact re-evaluation (Fractions) of the box bound of Sec. 7-8.1 (Corollary 5 with the Lemma 6/7/8 forms, worst-case
    substitution of Sec. 5) for small parameters of every regime, written from the text; the optimal tau is computed in
    50-digit arithmetic and then rationalised, so the reported number is an exact lower bound of the stated form.
    Compared with outputs/box_summary.txt of [Z].
[D] Rounding: the same bound for the families Tm, Tp, R1c, R4c at n1 = 10^3, 10^4, 10^5 in binary64 and in 40-digit
    mpmath, to measure the actual floating error that the a-priori allowance (1e-9) must cover, and the size of F0
    (which controls the cancellation in 1 - tau(4/3+W)).
Usage: python rb_z32.py MAXA MAXB > ../outputs/rb_z32_out.txt
"""
import sys, os, math, time, itertools
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C

H = Fr(1, 2); K32 = Fr(3, 2)


def P(n, m, L):
    return (n * n - L * L) * (L * L - m * m)


def regime(sp):
    """sp = sorted doubled legs (a<=b<=c<=d); returns (name, ordered pairing as doubled (A,B,C,D))"""
    a, b, c, d = sp
    g1, g2, g3 = Fr(b - a, 2), Fr(c - b, 2), Fr(d - c, 2)
    if g1 - g3 >= K32: return 'R1', (a, c, b, d)
    if g1 - g3 == H and g3 >= H: return 'R2-', (a, c, b, d)
    if g1 == H and g3 == 0 and g2 >= H: return 'R1c', (a, b, c, d)
    if g1 == H and g2 == 0 and g3 == 0: return 'Tm', (b, c, a, d)
    if g3 - g1 >= K32: return 'R4', (a, c, b, d)
    if g3 - g1 == H and g1 >= H: return 'R2+', (a, c, b, d)
    if g1 == 0 and g3 == H and g2 >= H: return 'R4c', (a, b, c, d)
    if g1 == 0 and g3 == H and g2 == 0: return 'Tp', (a, b, c, d)
    raise ValueError(sp)


def table(name, n1, m1, n2, m2):
    """[Z] Lemma 2 table: (type0, rows0, cols0, type1, rows1, cols1) with ranges as (lo, hi) Fractions"""
    if name == 'R1':  return ('CR', (m1, n1 - 1), (m1, n1 - 1), 'CR', (m1, n1 - 1), (m1, n1 - 1))
    if name == 'R2-': return ('CR', (m1, n1 - 1), (m1, n1 - 1), 'RR', (m1, n1 - 1), (m1 + 1, n1 - 1))
    if name == 'R1c': return ('CR', (H, n1 - 1), (H, n1 - 1), 'CR', (K32, n1 - 1), (K32, n1 - 1))
    if name == 'Tm':  return ('RC', (0, n1 - 3), (1, n1 - 2), 'RR', (1, n1 - 1), (2, n1 - 1))
    if name == 'R4':  return ('RR', (m2 - K32, n1 - 1), (m2 - H, n1 - 1), 'RR', (m2 + H, n1 - 1), (m2 + K32, n1 - 1))
    if name == 'R2+': return ('CR', (m1, n1 - 1), (m1, n1 - 1), 'RR', (m1 + 1, n1 - 1), (m1 + 2, n1 - 1))
    if name == 'R4c': return ('RR', (0, n1 - 1), (1, n1 - 1), 'RR', (1, n1 - 1), (2, n1 - 1))
    if name == 'Tp':  return ('RC', (0, n1 - 2), (1, n1 - 1), 'RR', (1, n1 - 1), (2, n1 - 1))


def zig(tj):
    """my own zigzag decomposition from rb_core sites: dict d -> sorted list of b"""
    S = C.sites_of(tj)
    z = {}
    for (B, Cc) in S:
        z.setdefault(Fr(B - Cc, 2), []).append(Fr(B, 2))
    return {k: sorted(v) for k, v in z.items()}, S


def ctype(rows, cols, drow, dcol):
    """chain type of a zigzag from its row/col b-lists: row (b, b-drow), col (b, b-dcol); compare first/last u"""
    if not rows and not cols: return 'empty'
    if not cols: return 'RR'          # a lone row (empty column range) is a degenerate tall chain
    if not rows: return 'CC'
    u = lambda b, dd: 2 * b - dd
    st = 'C' if u(cols[0], dcol) < u(rows[0], drow) else 'R'
    en = 'R' if u(rows[-1], drow) > u(cols[-1], dcol) else 'C'
    return st + en


def rng_(lst):
    return (lst[0], lst[-1]) if lst else None


def partA_B(MAXA, MAXB):
    t0 = time.time()
    nA = badlink = badid = badsign = 0; nB = badB = 0; regcount = {}
    for sp in itertools.combinations_with_replacement(range(1, MAXB + 1), 4):
        if (3 + sum(sp)) % 2:
            continue
        name, (A_, B_, C_, D_) = regime(sp)
        tj = (3, A_, B_, C_, D_)
        S = C.sites_of(tj)
        if len(S) < 2:
            continue
        nB += 1; regcount[name] = regcount.get(name, 0) + 1
        a, b, c, d = (Fr(x, 2) for x in (A_, B_, C_, D_))
        n1, m1, n2, m2 = a + b + 1, abs(a - b), c + d + 1, abs(c - d)
        z, _ = zig(tj)
        R0, C0, R1, C1 = z.get(-K32, []), z.get(-H, []), z.get(H, []), z.get(K32, [])
        tab = table(name, n1, m1, n2, m2)
        mine = (ctype(R0, C0, -K32, -H), rng_(R0), rng_(C0), ctype(R1, C1, H, K32), rng_(R1), rng_(C1))
        ok = True
        # compare ranges where nonempty in the table
        for k in (1, 2, 4, 5):
            lo, hi = tab[k]
            # lattice: b = m1 mod 1
            lo2 = lo if (lo - m1).denominator == 1 else lo + H
            if lo2 > hi:
                ok &= mine[k] is None
            else:
                ok &= mine[k] == (lo2, hi)
        if mine[4] is not None:
            ok &= mine[3] == tab[3]
        ok &= mine[0] == tab[0]
        imb0 = len(R0) - len(C0); imb1 = len(R1) - len(C1)
        ok &= imb0 in (0, 1) and imb1 in (0, 1)
        if sp[-1] <= MAXA:
            # exact links: connected paths, table formulas, Prop 4
            _, ent = C.R_entries(tj)
            sc = C.scale_sq(3)
            idx = {p: i for i, p in enumerate(S)}
            L = {}
            for (i, j), (sg, sq) in ent.items():
                L[frozenset((S[i], S[j]))] = (sg, sq * sc)
            st = lambda b_, c_: (int(2 * b_), int(2 * c_))
            def lk(p1, p2):
                return L.get(frozenset((st(*p1), st(*p2))), (0, Fr(0)))
            nA += 1
            for bb in R0:
                if bb in C0:
                    sg, v = lk((bb, bb + H), (bb, bb + K32)); badlink += v != Fr(3, 16) * P(n2, m2, bb + K32) * bb / (bb + 1)
                    badlink += v == 0
                if bb + 1 in C0:
                    sg, v = lk((bb + 1, bb + K32), (bb, bb + K32)); badlink += v != Fr(3, 16) * P(n1, m1, bb + 1) * (2 * bb + 5) / (2 * bb + 3)
                    badlink += v == 0
            for bb in R1:
                if bb in C1:
                    sg, v = lk((bb, bb - K32), (bb, bb - H)); badlink += v != Fr(3, 16) * P(n2, m2, bb - H) * (bb + 1) / bb
                    badlink += v == 0
                if bb + 1 in C1:
                    sg, v = lk((bb, bb - H), (bb + 1, bb - H)); badlink += v != Fr(3, 16) * P(n1, m1, bb + 1) * (2 * bb - 1) / (2 * bb + 1)
                    badlink += v == 0
                if bb in C0:
                    sg, v = lk((bb, bb - H), (bb, bb + H)); badlink += v != P(n2, m2, bb + H) * (2 * bb + 3) * (2 * bb - 1) / (16 * bb * (bb + 1))
                if bb - 1 in C0:
                    sg, v = lk((bb, bb - H), (bb - 1, bb - H)); badlink += v != P(n1, m1, bb) * (bb * bb - 1) / (4 * bb * bb - 1)
            # Prop 4 on the full zigzag-0 column set, exact (with signs)
            for cc in C0:
                q = lk((cc, cc + H), (cc, cc + K32)); pm = lk((cc, cc + H), (cc - 1, cc + H))      # q_c, p_{c-1}
                npl = lk((cc, cc + H), (cc, cc - H)); nmi = lk((cc, cc + H), (cc + 1, cc + H))    # n+_c, n-_{c+1}
                lam = npl[1] + nmi[1] - Fr(4, 3) * (q[1] + pm[1])
                cf = (P(n1, m1, cc + 1) * cc * (cc + 2) / ((2 * cc + 1) * (2 * cc + 3)) - P(n1, m1, cc) * (2 * cc + 3) / (4 * (2 * cc + 1))
                      + P(n2, m2, cc + H) * (2 * cc + 3) * (2 * cc - 1) / (16 * cc * (cc + 1)) - P(n2, m2, cc + K32) * cc / (4 * (cc + 1)))
                badid += lam != cf
                if cc + 1 in C0:
                    a1 = lk((cc, cc + K32), (cc, cc + H)); a2 = lk((cc, cc + K32), (cc + 1, cc + K32))     # q_c, p_c at R_c
                    b1 = lk((cc + 1, cc + H), (cc, cc + H)); b2 = lk((cc + 1, cc + H), (cc + 1, cc + Fr(3, 2)))  # n-_{c+1}, n+_{c+1} at R'_{c+1}
                    if a1[1] and a2[1]:
                        badid += (b1[1] * b2[1]) != Fr(16, 9) * (a1[1] * a2[1])
                        # signs: off-diagonal of N^T N and D0^T D0 must have the same sign (as entries of R)
                        badsign += (b1[0] * b2[0]) != (a1[0] * a2[0])
        if not ok:
            badB += 1
            if badB < 6: print('   census mismatch', name, sp, mine, tab)
    print(f"[A] exact zigzag links vs [Z] table and Prop. 4 identity (legs 2j<={MAXA}): {nA} tuples; "
          f"link mismatches/zero links {badlink}, identity mismatches {badid}, sign mismatches {badsign}")
    print(f"[B] regime census (legs 2j<={MAXB}): {nB} tuples, mismatches with the Lemma 2 table {badB}; per regime {regcount}  ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------- [C] box bound, exact
def q0(n2, m2, b): return Fr(3, 16) * P(n2, m2, b + K32) * b / (b + 1)
def p0(n1, m1, b): return Fr(3, 16) * P(n1, m1, b + 1) * (2 * b + 5) / (2 * b + 3)
def q1(n2, m2, b): return Fr(3, 16) * P(n2, m2, b - H) * (b + 1) / b
def p1(n1, m1, b): return Fr(3, 16) * P(n1, m1, b + 1) * (2 * b - 1) / (2 * b + 1)


def Lam(c, n1, m1, n2, m2):
    return (P(n1, m1, c + 1) * c * (c + 2) / ((2 * c + 1) * (2 * c + 3)) - P(n1, m1, c) * (2 * c + 3) / (4 * (2 * c + 1))
            + P(n2, m2, c + H) * (2 * c + 3) * (2 * c - 1) / (16 * c * (c + 1)) - P(n2, m2, c + K32) * c / (4 * (c + 1)))


def CR(lo, hi, ell, pp, lam):
    cs = [lo + k for k in range(int(hi - lo) + 1)]
    Fc = sum((hi - c + 1) / ell(c) for c in cs)
    Phi = Fr(0); Ft = Fr(0); pprev = Fr(0)
    for c in cs:
        Phi = (pprev * Phi + 1) / ell(c); Ft += Phi; pprev = pp(c)
    W = None
    if lam is not None:
        Wc = sum(max(lam(c), 0) * (hi - c + 1) / ell(c) for c in cs)
        R = Fr(0); Wt = Fr(0)
        for c in reversed(cs):
            R = (1 + (pp(c) if c < hi else 0) * R) / ell(c); Wt += max(lam(c), 0) * R
        W = min(Wc, Wt)
    return min(Fc, Ft), W


def RR(lo, hi, pp, qq):
    F = Fr(0)
    i = lo
    while i <= hi:
        if i < hi: F += (hi - i) / pp(i)
        if i > lo: F += (i - lo) / qq(i)
        i += 1
    return F


def RCb(n, hi, n2, lam):
    Fc = Fr(0); Ft = Fr(0); Wc = Fr(0); Wt = Fr(0); Rc = Fr(0); acc = Fr(0)
    for k in range(int(hi) + 1):
        i = Fr(k)
        pi = p0(n, 0, i); qi = q0(n2, H, i)
        Fc += (hi - i + 1) / pi; Rc += 1 / pi; acc = acc * (qi / pi) + 1 / pi; Ft += acc
        l = max(lam(i + 1), 0)
        Wc += l * Rc; Wt += l * acc
    return min(Fc, Ft), min(Wc, Wt)


def bound(F0, F1, W):
    """max over tau of min(A(1-tau k), tau B/(1+tau)); tau from 50 digits, rationalised; exact value returned"""
    F0, F1, W = (Fr(x) if isinstance(x, float) else x for x in (F0, F1, W))   # binary64 inputs converted exactly
    if F0 == 0:                                  # zigzag 0 has no columns: only the D1 constraint remains
        return (1 / F1) if F1 else Fr(10 ** 9)
    A = 1 / F0; k = Fr(4, 3) + W
    if F1 == 0:
        return A
    B = 1 / F1
    mp.mp.dps = 50
    Am, km, Bm = mp.mpf(A.numerator) / A.denominator, mp.mpf(k.numerator) / k.denominator, mp.mpf(B.numerator) / B.denominator
    bb = Bm - Am + Am * km
    tau = (-bb + mp.sqrt(bb * bb + 4 * Am * Am * km)) / (2 * Am * km)
    t = Fr(mp.nstr(tau, 40))
    return min(A * (1 - t * k), t * B / (1 + t))


def regime_bound(name, n, a):
    """exact bound of the [Z] box for one parameter set (n = n1, a = m1, or m2 for R4); returns 8 sigma lower bound"""
    if name == 'R1':
        m2w = a - K32; mu = Fr(0) if (a + H) % 1 == 0 else H
        F0, W = CR(a, n - 1, lambda c: q0(n + K32, m2w, c), lambda c: p0(n, a, c), lambda c: Lam(c, n, a, n + K32, mu))
        F1, _ = CR(a, n - 1, lambda c: q1(n + K32, m2w, c), lambda c: p1(n, a, c), None)
    elif name in ('R2+', 'R2-'):
        m2 = a + H if name == 'R2+' else a - H
        F0, W = CR(a, n - 1, lambda c: q0(n + K32, m2, c), lambda c: p0(n, a, c), lambda c: Lam(c, n, a, n + K32, m2))
        lo1 = a + 1 if name == 'R2+' else a
        F1 = RR(lo1, n - 1, lambda i: p1(n, a, i), lambda i: q1(n + K32, m2, i))
    elif name == 'R4':
        m2 = a; b0 = m2 - K32
        F0 = RR(b0, n - 1, lambda i: p0(n, b0, i), lambda i: q0(n + K32, m2, i))
        F1 = RR(m2 + H, n - 1, lambda i: p1(n, b0, i), lambda i: q1(n + K32, m2, i))
        W = Fr(0)
    elif name == 'R1c':
        F0, W = CR(H, n - 1, lambda c: q0(n + K32, 0, c), lambda c: p0(n, H, c), lambda c: Lam(c, n, H, n + K32, 0))
        F1, _ = CR(K32, n - 1, lambda c: q1(n + K32, 0, c), lambda c: p1(n, H, c), None)
    elif name == 'R4c':
        F0 = RR(Fr(0), n - 1, lambda i: p0(n, 0, i), lambda i: q0(n + K32, H, i))
        F1 = RR(Fr(1), n - 1, lambda i: p1(n, 0, i), lambda i: q1(n + K32, H, i))
        W = Fr(0)
    else:
        n2 = n + H if name == 'Tp' else n - H
        hi0 = n - 2 if name == 'Tp' else n - 3
        F0, W = RCb(n, hi0, n2, lambda c: Lam(c, n, 0, n2, H))
        F1 = RR(Fr(1), n - 1, lambda i: p1(n, 0, i), lambda i: q1(n2, H, i))
    s2 = bound(F0, F1, W)
    return 8 * math.sqrt(float(s2)) if s2 > 0 else float('nan'), s2


def partC(NC2, NC1):
    reported = {'R1': (10.873596, (3.5, 1.5)), 'R2+': (7.125869, (4.5, 0.5)), 'R2-': (4.723103, (3.0, 1.0)), 'R4': (6.507914, (2.0, 1.5)),
                'R1c': (8.219253, (2.5, 0)), 'R4c': (4.621864, (3.0, 0)), 'Tm': (3.777423, (4.0, 0)), 'Tp': (4.386733, (4.0, 0))}
    t0 = time.time()
    for name in ('R1', 'R2+', 'R2-', 'R4'):
        best = (1e9, None)
        amin = {'R1': K32, 'R2+': H, 'R2-': Fr(1), 'R4': K32}[name]
        for n2x in range(4, 2 * NC2 + 1):
            n = Fr(n2x, 2)
            if name == 'R4':
                a = n + H
                while a >= K32:
                    v, _ = regime_bound(name, n, a)
                    if v == v and v < best[0]: best = (v, (float(n), float(a)))
                    a -= 1
            else:
                a = n - 2
                while a >= amin:
                    v, _ = regime_bound(name, n, a)
                    if v == v and v < best[0]: best = (v, (float(n), float(a)))
                    a -= 1
        print(f"[C] {name:4s} exact, n1 <= {NC2}: min 8sigma bound {best[0]:.6f} at {best[1]};  [Z] box reports {reported[name][0]} at {reported[name][1]}", flush=True)
    for name in ('R1c', 'R4c', 'Tp', 'Tm'):
        start = {'R1c': Fr(5, 2), 'R4c': Fr(2), 'Tp': Fr(2), 'Tm': Fr(3)}[name]
        best = (1e9, None); n = start; vals = []
        while n <= NC1:
            v, _ = regime_bound(name, n, 0)
            vals.append((float(n), v))
            if v == v and v < best[0]: best = (v, float(n))
            n += 1
        print(f"[C] {name:4s} exact, n1 <= {NC1}: min 8sigma bound {best[0]:.6f} at n1={best[1]};  [Z] box reports {reported[name][0]} at {reported[name][1]};"
              f"  first values {[(x, round(y, 4)) for x, y in vals[:5]]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------- [D] rounding at large n (families)
def family_float(name, n, prec=None):
    """same bound as regime_bound for 1-D families, in binary64 (prec=None) or mpmath (prec digits); returns (s2, F0, F1, W)"""
    if prec:
        mp.mp.dps = prec; num = mp.mpf
    else:
        num = float
    h = num(1) / 2; k3 = num(3) / 2
    nn = num(n)
    Pn = lambda N_, M_, L_: (N_ * N_ - L_ * L_) * (L_ * L_ - M_ * M_)
    q0f = lambda N2, M2, b: num(3) / 16 * Pn(N2, M2, b + k3) * b / (b + 1)
    p0f = lambda N1, M1, b: num(3) / 16 * Pn(N1, M1, b + 1) * (2 * b + 5) / (2 * b + 3)
    q1f = lambda N2, M2, b: num(3) / 16 * Pn(N2, M2, b - h) * (b + 1) / b
    p1f = lambda N1, M1, b: num(3) / 16 * Pn(N1, M1, b + 1) * (2 * b - 1) / (2 * b + 1)

    def lamf(c, N1, M1, N2, M2):
        return (Pn(N1, M1, c + 1) * c * (c + 2) / ((2 * c + 1) * (2 * c + 3)) - Pn(N1, M1, c) * (2 * c + 3) / (4 * (2 * c + 1))
                + Pn(N2, M2, c + h) * (2 * c + 3) * (2 * c - 1) / (16 * c * (c + 1)) - Pn(N2, M2, c + k3) * c / (4 * (c + 1)))

    def RRf(lo, hi, pp, qq):
        F = num(0); i = lo
        while i <= hi:
            if i < hi: F += (hi - i) / pp(i)
            if i > lo: F += (i - lo) / qq(i)
            i += 1
        return F
    zero = num(0)
    if name in ('Tp', 'Tm'):
        n2 = nn + h if name == 'Tp' else nn - h
        hi0 = int(n) - 2 if name == 'Tp' else int(n) - 3
        Fc = Ft = Wc = Wt = Rc = acc = zero
        for kk in range(hi0 + 1):
            i = num(kk)
            pi = p0f(nn, zero, i); qi = q0f(n2, h, i)
            Fc += (hi0 - i + 1) / pi; Rc += 1 / pi; acc = acc * (qi / pi) + 1 / pi; Ft += acc
            l = max(lamf(i + 1, nn, zero, n2, h), zero)
            Wc += l * Rc; Wt += l * acc
        F0, W = min(Fc, Ft), min(Wc, Wt)
        F1 = RRf(num(1), nn - 1, lambda i: p1f(nn, zero, i), lambda i: q1f(n2, h, i))
    elif name == 'R4c':
        F0 = RRf(zero, nn - 1, lambda i: p0f(nn, zero, i), lambda i: q0f(nn + k3, h, i))
        F1 = RRf(num(1), nn - 1, lambda i: p1f(nn, zero, i), lambda i: q1f(nn + k3, h, i)); W = zero
    A = 1 / F0; kk_ = num(4) / 3 + W; B = 1 / F1
    bb = B - A + A * kk_
    tau = (-bb + (bb * bb + 4 * A * A * kk_) ** h) / (2 * A * kk_) if not prec else (-bb + mp.sqrt(bb * bb + 4 * A * A * kk_)) / (2 * A * kk_)
    s2 = min(A * (1 - tau * kk_), tau * B / (1 + tau))
    return s2, F0, F1, W, 1 - tau * kk_


def partD():
    for name in ('Tm', 'Tp', 'R4c'):
        for n in (1000, 10000, 100000):
            t0 = time.time()
            sf, F0f, F1f, Wf, cf = family_float(name, n)
            sm, F0m, F1m, Wm, cm = family_float(name, n, prec=40)
            rel = abs(mp.mpf(sf) - sm) / sm
            relF = max(abs(mp.mpf(F0f) - F0m) / F0m, abs(mp.mpf(F1f) - F1m) / F1m, (abs(mp.mpf(Wf) - Wm) / Wm) if Wm else 0)
            print(f"[D] {name} n1={n}: 8sigma bound {8*math.sqrt(sf):.4f};  rel. error of sigma^2 (binary64 vs 40 digits) {float(rel):.2e};"
                  f"  max rel. error of F0,F1,W {float(relF):.2e};  F0 = {F0f:.3e};  1 - tau(4/3+W) = {cf:.3e}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    MAXA, MAXB = int(sys.argv[1]), int(sys.argv[2])
    partA_B(MAXA, MAXB)
    partC(20, 60)
    partD()

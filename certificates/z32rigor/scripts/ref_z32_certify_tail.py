"""Tail certificate (beyond the computed box) for the j1 = 3/2 theorem.  Derivations: THEOREM.md section 7.

Target: 8 sigma >= TSTAR (sigma = sigma_min(B)), i.e. sigma^2 >= s* = (TSTAR/8)^2.  With tau = theta/(4/3 + W),
  sigma^2 >= min( (1-theta)/F0 , theta/((4/3 + W + theta) F1) ),
so it suffices that  F0 <= (1-theta)/s*  and  G := (4/3 + theta + W) F1 <= theta/s*.   theta = 0.95.

Every bound below is a finite sum of terms  K (1+ln n)^e (1+ln 2n)^f / (n+g)^d  with K,g >= 0, d >= 1, e+f <= 2,
times factors (1+2/n) or S(n) that are nonincreasing, or a constant (factors n/(n+g) are bounded by 1 before evaluation);
each such term is nonincreasing for n >= 100 (log-derivative e/(n(1+ln n)) + f/(n(1+ln 2n)) - d/(n+g) < 0).
Hence checking the inequality at the box edge n0 proves it for all n >= n0.
Floating evaluation of a handful of numbers; margins are reported.

Usage: python z32_certify_tail.py out.txt
"""
import sys, math

TSTAR = 3.7
THETA = 0.95
KAP = 4.0 / 3.0
SSTAR = (TSTAR / 8) ** 2
F0MAX = (1 - THETA) / SSTAR
GMAX = THETA / SSTAR
Z2 = math.pi ** 2 / 6
LN2 = math.log(2)
L = lambda n: 1 + math.log(n)          # >= H_n
L2 = lambda n: 1 + math.log(2 * n)
OUT = []


def report(label, n0, G, F0):
    ok = (G <= GMAX) and (F0 <= F0MAX)
    msg = (f"{'PASS' if ok else 'FAIL'}  {label}: at n0={n0}:  G <= {G:.4f} (need <= {GMAX:.4f}),  "
           f"F0 <= {F0:.5f} (need <= {F0MAX:.4f})")
    print(msg); OUT.append(msg)
    return ok


def S(x):        # >= sum_{T>=1} 1/(T(T+x)),  x > 0
    return 1 / (1 + x) + math.log(1 + x) / x


# ------------------------------------------------------------------ R1  (a >= 3/2, telescoping, C0 = C1 = 1)
def tail_R1(n):
    """a-uniform bound for n >= 100.
    F1 <= 8.774/((n+a+1)(2a-2));  W <= (4/3)(1+1/a)[1.016 n S + a Z2/2],  S = min(Z2/2, 2(1/2 + a ln2) H_N/(N+1)).
    Case a <= n/2:  n/(N+1) <= 2,  (1+1/a)(1/2 + a ln 2)/(2a-2) <= 2.5664 (a >= 3/2, decreasing).
    Case a >  n/2:  S <= Z2/2, (1+1/a)/(2a-2) <= (1+2/n)/(n-2), n/(n+a+1) <= 2/3.
    W-second part: (4/3)(1+1/a)(a Z2/2) F1 = (2/3) Z2 * 8.774 (a+1)/((n+a+1)(2a-2)) <= (2/3) Z2 8.774 * 2.5/(n+2.5)."""
    c1 = (4 / 3) * 1.016 * 2 * 2 * 2.5664 * 8.774 * L(n) / (n + 2.5)
    c2 = (4 / 3) * 1.016 * (Z2 / 2) * 8.774 * (2 / 3) * (1 + 2 / n) / (n - 2)
    w2 = (2 / 3) * Z2 * 8.774 * 2.5 / (n + 2.5)
    t = (KAP + THETA) * 8.774 / (n + 2.5)
    return max(c1, c2) + w2 + t, 3.5105 / (n + 4.5)


# ------------------------------------------------------------------ R2-  (a >= 1, telescoping, C0 = 5/4)
def tail_R2m(n):
    """F1 <= 16 S(2a)/(n+a+1) + F1s,  F1s = (32/3)(1+ln 2n)/((n+a+1)(n+a+2));
    W <= (5/3)(1+1/a)[1.016 n S_W + a Z2/2],  S_W = min(Z2/2, 2 G H_N/(N+1)),  G <= 1 + a ln 2.
    (1+1/a) G <= 2.6932 + 0.6932 a;  (2.6932 + 0.6932a) S(2a) <= 2.6932 S(2) + 0.3466 (1 + ln(1+2a));
    (a+1) S(2a) <= 1 + ln(1+2a);  ln(1+2a) <= ln 2n;  (2.6932+0.6932a)/(n+a+1) <= 0.6932 (n >= 3)."""
    S2 = S(2.0)
    # case a <= n/2
    w1m = (5 / 3) * 1.016 * 4 * 16 * (2.6932 * S2 + 0.3466 * L2(n)) * L(n) / (n + 2)
    w1s = (5 / 3) * 1.016 * 4 * 0.6932 * L(n) * (32 / 3) * L2(n) / (n + 2)
    # case a > n/2: S_W <= Z2/2, 1+1/a <= 1+2/n, S(2a) <= S(n), n/(n+a+1) <= 2/3
    c2 = (5 / 3) * 1.016 * (Z2 / 2) * (1 + 2 / n) * (16 * S(n) * (2 / 3) + (32 / 3) * L2(n) / (1.5 * n + 2) * (2 / 3))
    w2 = (5 / 6) * Z2 * (16 + 32 / 3) * L2(n) / (n + 2)
    F1max = 16 * S2 / (n + 2) + (32 / 3) * L2(n) / ((n + 2) * (n + 3))
    t = (KAP + THETA) * F1max
    return max(w1m + w1s, c2) + w2 + t, 5.3334 / (n + 4)


# ------------------------------------------------------------------ R2+  (a >= 1/2, count form)
def series_R2p(a, Tmax=200000):
    """sigma0(a) = sum_T (c+1)/(T(c+a+2)c),  Omega(a) = sum_{T: c>=1} a^2(c+1)/(c^2 T (c+a+2)),
    Sp(2a) = sum_T 1/((T+1)(T+1+2a)), with c = a+T-1; truncated at Tmax plus rigorous tail bounds
    (each summand <= 2/T^2, <= 8a^2/T^3, <= 1/T^2 respectively for T > Tmax >= 2a+2; float summation
    of <= 2e5 positive terms has relative error < 1e-10, covered by the factor 1 + 1e-9)."""
    s0 = om = sp = 0.0
    for T in range(1, Tmax + 1):
        c = a + T - 1
        s0 += (c + 1) / (T * (c + a + 2) * c)
        if c >= 1:
            om += a * a * (c + 1) / (c * c * T * (c + a + 2))
        sp += 1 / ((T + 1) * (T + 1 + 2 * a))
    s0 += 2 / Tmax; om += 4 * a * a / Tmax ** 2; sp += 1 / Tmax
    return s0 * (1 + 1e-9), om * (1 + 1e-9), sp * (1 + 1e-9)


def tail_R2p_a(n, a, s0, om, sp):
    """count form, fixed a:  F0 <= (16/3) s0/(n+a+3);  F1 <= (32/3) sp/(n+a+2) + (32/3)(1+ln2n)/((n+a+3)(n+a+2));
    W <= [a=1/2] n/3 + (4/3)[1.016 n Omega + (a/2)(1+1/cmin) H_N],  cmin = max(a,1)."""
    cmin = max(a, 1.0)
    F1 = (32 / 3) * sp / (n + a + 2) + (32 / 3) * L2(n) / ((n + a + 3) * (n + a + 2))
    # W*F1 with n W-terms bounded via n/(n+a+2) <= 1
    WF1 = ((1 / 3 if abs(a - 0.5) < 1e-9 else 0) + (4 / 3) * 1.016 * om) * (32 / 3) * sp \
        + ((1 / 3 if abs(a - 0.5) < 1e-9 else 0) + (4 / 3) * 1.016 * om) * (32 / 3) * L2(n) / (n + a + 2) \
        + (4 / 3) * (a / 2) * (1 + 1 / cmin) * L(n) * F1
    return (KAP + THETA) * F1 + WF1, (16 / 3) * s0 / (n + a + 3)


def tail_R2p_large(n, A):
    """a > A >= 60, uniform: s0 <= 2 S(2a+1)... we use  F0 <= (16/3)(1+1/a) S(2a+1)/(n+a+3) <= tiny,
    Omega(a) <= (1+1/a)(0.5 (1 + ln a) + 0.51),  Sp(2a) <= ln(1+2a)/(2a);
    the product Omega(a) Sp(2a) <= (1+1/A)(0.5(1+ln a)+0.51) ln(1+2a)/(2a) is decreasing for a >= A (checked at A),
    and ln(1+2a) <= ln 2n, a <= n."""
    a = A
    om_sp = (1 + 1 / A) * (0.5 * (1 + math.log(a)) + 0.51) * math.log(1 + 2 * a) / (2 * a)
    F1 = (32 / 3) * (math.log(1 + 2 * A) / (2 * A)) / (n + A + 2) + (32 / 3) * L2(n) / ((n + A + 3) * (n + A + 2))
    WF1 = (4 / 3) * 1.016 * (32 / 3) * om_sp \
        + (4 / 3) * 1.016 * (1 + 1 / A) * (0.5 * L(n) + 0.51) * (32 / 3) * L2(n) / (n + A + 3) \
        + (4 / 3) * 0.5 * (1 + 1 / A) * L(n) * (32 / 3) * (math.log(2 * n) / 2) / (n + A + 2) \
        + (4 / 3) * 0.5 * (1 + 1 / A) * L(n) * (32 / 3) * L2(n) / (n + A + 3)
    return (KAP + THETA) * F1 + WF1, (16 / 3) * (1 + 1 / A) * S(2 * A + 1) / (n + A + 3)


# ------------------------------------------------------------------ R4  (W = 0, count)
def tail_R4(n):
    """b0 = m2 - 3/2 >= 0.  F0 <= (16/3) Z2/(n+b0+1) + (32/3)(1+ln n)/((n+b0+4)(b0+4));
    F1 <= 3.511/(n+b0+3) + (16/3)(1+ln n)/((n+b0+4)(2b0+4)).  Worst b0 = 0."""
    F0 = (16 / 3) * Z2 / (n + 1) + (32 / 3) * L(n) / ((n + 4) * 4)
    F1 = 3.511 / (n + 3) + (16 / 3) * L(n) / ((n + 4) * 4)
    return (KAP + THETA) * F1, F0


# ------------------------------------------------------------------ R1c (count, 1-D)
def tail_R1c(n):
    """F0 <= (16/3)(1+2ln2)(2/3)/(n+3.5);  F1 <= (16/3) Z2/(n+2.5);
    W <= (1/3)[1.016 * 0.69 n + (5/3 + ln n)/(n+4.5)]  (c=1/2 column has Lambda <= 0)."""
    F0 = (16 / 3) * (2 / 3) * (1 + 2 * LN2) / (n + 3.5)
    F1 = (16 / 3) * Z2 / (n + 2.5)
    # W = (1/3)(1.016*0.69 n + (5/3 + ln n)/(n+4.5));  W*F1 <= (1/3)(1.016*0.69)(16/3)Z2  (n/(n+2.5) <= 1) + decreasing rest
    G = (KAP + THETA) * F1 + (1 / 3) * 1.016 * 0.69 * (16 / 3) * Z2 + (1 / 3) * (5 / 3 + math.log(n)) / (n + 4.5) * F1
    return G, F0


# ------------------------------------------------------------------ R4c (W = 0, count, 1-D)
def tail_R4c(n):
    F0 = (16 / 3) * (Z2 / (n + 1) + 2 * L(n + 1) / ((n + 2) * (n + 4)))
    F1 = (16 / 3) * (3 * (Z2 - 1) / (n + 2) + 2 * L(n + 1) / ((n + 3) ** 2))
    return (KAP + THETA) * F1, F0


# ------------------------------------------------------------------ Tp, Tm (telescoping, 1-D)
def tail_T(n, which):
    """F0 <= (64/15)(1+ln n)^2/(n+1) [Tm] (x2 for Tp);  F1 <= (16/3)(3(Z2-1))/(n+2) + (32/3)(1+ln n)/(n+1)^2;
    W <= (20/3)(1+ln n)^2  [Tm: Lambda_c <= (n(4c+3) + 8c^2+10c+3)/16];  W <= (16/3)(1+ln n)^2 (4n+6)/(4(n+2)) [Tp]."""
    F0 = (64 / 15) * L(n) ** 2 / (n + 1) * (2 if which == 'Tp' else 1)
    F1 = (16 / 3) * 3 * (Z2 - 1) / (n + 2) + (32 / 3) * L(n) / (n + 1) ** 2
    W = (20 / 3) * L(n) ** 2 if which == 'Tm' else (16 / 3) * L(n) ** 2
    return (KAP + THETA + W) * F1, F0


if __name__ == '__main__':
    allok = True
    N2D, N1D = 4000.0, 100000.0
    OUT.append(f'TSTAR = {TSTAR}, theta = {THETA}: need F0 <= {F0MAX:.4f}, G <= {GMAX:.4f}')
    allok &= report('R1  (all a >= 3/2)', N2D, *tail_R1(N2D))
    allok &= report('R2- (all a >= 1)', N2D, *tail_R2m(N2D))
    A = 60.0
    worst = (0, None)
    a = 0.5
    while a <= A + 1e-9:
        s0, om, sp = series_R2p(a)
        G, F0 = tail_R2p_a(N2D, a, s0, om, sp)
        if G > worst[0]:
            worst = (G, a, F0)
        a += 0.5
    allok &= report(f'R2+ (a <= {A}, per-a series; worst a = {worst[1]})', N2D, worst[0], worst[2])
    allok &= report(f'R2+ (a > {A}, uniform)', N2D, *tail_R2p_large(N2D, A))
    allok &= report('R4  (all m2)', N2D, *tail_R4(N2D))
    allok &= report('R1c', N1D, *tail_R1c(N1D))
    allok &= report('R4c', N1D, *tail_R4c(N1D))
    allok &= report('Tp', N1D, *tail_T(N1D, 'Tp'))
    allok &= report('Tm', N1D, *tail_T(N1D, 'Tm'))
    OUT.append('ALL PASS' if allok else 'SOME FAIL')
    print(OUT[-1])
    if len(sys.argv) > 1:
        open(sys.argv[1], 'w').write('\n'.join(OUT) + '\n')

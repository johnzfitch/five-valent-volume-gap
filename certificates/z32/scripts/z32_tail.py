"""Closed-form tail bounds (large n = n1) for each regime; derivations in THEOREM.md section 7.

Each tail_X(n, a) returns upper bounds (F0, F1, W) valid for every parameter set of regime X with
the given n and a (a = m1, or b0 = m2 - 3/2 for R4).  Each Gbar_X(n) is an a-uniform upper bound on
   G(n,a) = (kappa + W + theta) F1   (kappa = 4/3, theta = 0.95)
together with the matching bound on F0; it is nonincreasing in n for n >= 10 (shape A(1+ln n)^k/(n+c)).
The final requirement for 8 sigma >= Tstar with sigma*^2 = (Tstar/8)^2 is
   F0 <= (1-theta)/sigma*^2   and   G <= theta/sigma*^2.
"""
import math

Z2 = math.pi ** 2 / 6            # >= H_N^(2)
LN2 = math.log(2.0)
KAP = 4.0 / 3.0
THETA = 0.95


def Hup(N):
    """H_N <= 1 + ln N (N >= 1)."""
    return 1.0 + math.log(max(N, 1.0))


def S_series(x):
    """>= sum_{T>=1} 1/(T(T+x)) for x>0:  1/(1+x) + ln(1+x)/x  (first term + integral)."""
    return 1 / (1 + x) + math.log(1 + x) / x


def Sprime(x):
    """>= sum_{T>=1} 1/((T+1)(T+1+x)) = S(x) - 1/(1+x)  <=  ln(1+x)/x  (x>0)."""
    return math.log(1 + x) / x


# ---------------------------------------------------------------- R1
def tail_R1(n, a):
    N = n - a
    F0 = 3.5105 / (n + a + 3)
    F1 = 8.774 / ((n + a + 1) * (2 * a - 2))
    G = 0.5 + a * LN2
    S = min(0.5 * Z2, 2 * G * Hup(N) / (N + 1))
    W = (4 / 3) * (1 + 1 / a) * (1.016 * n * S + 0.5 * a * Z2)
    return F0, F1, W


def Gbar_R1(n):
    """a-uniform:  case N >= a (a <= n/2):  n/(N+1) < 2,  (1/2 + a ln2)(1+1/a)/(2a-2) <= 2.5642 (a>=3/2, decreasing),
                   (n+a+1) >= n + 2.5, H_N <= 1 + ln n;
       case N < a  (a > n/2 >= 5):  S <= Z2/2, (1+1/a)/(2a-2) <= 1.1/(n-2), n/(n+a+1) <= 2/3.
       W-term2 and kappa,theta terms:  (4/3)(Z2/2)(a+1)/(2a-2) <= (4/3)(Z2/2)(2.5);  (KAP+THETA) F1 <= (KAP+THETA)*8.774/(n+2.5)."""
    c1 = (4 / 3) * 1.016 * 8.774 * 2 * 2 * 2.5642 * Hup(n) / (n + 2.5)
    c2 = (4 / 3) * 1.016 * 0.5 * Z2 * 8.774 * 1.1 / (n - 2) * (2 / 3)
    t2 = (4 / 3) * 0.5 * Z2 * 2.5 * 8.774 / (n + 2.5)
    t3 = (KAP + THETA) * 8.774 / (n + 2.5)
    return max(c1, c2) + t2 + t3, 3.5105 / (n + 4.5)


# ---------------------------------------------------------------- R2-
def tail_R2m(n, a):
    N = n - a
    F0 = 5.3334 / (n + a + 3)
    F1 = 16 * S_series(2 * a) / (n + a + 1) + (32 / 3) * Hup(2 * n) / ((n + a + 1) * (n + a + 2))
    G = 0.5 + (a * a / (a + 1)) * math.log((2 * a + 1) / a)
    S = min(0.5 * Z2, 2 * G * Hup(N) / (N + 1))
    W = 1.25 * (4 / 3) * (1 + 1 / a) * (1.016 * n * S + 0.5 * a * Z2)
    return F0, F1, W


def Gbar_R2m(n):
    """a-uniform (a>=1):  G(1+1/a) <= 1.0493 + a ln2 * (1+1/a) ... we use  G <= 1/2 + a ln(3)  and (1+1/a)<=2;
       S_series(2a) <= (1 + ln(1+2a))/(1+2a) * 2  for a>=1 ... we use the explicit monotone bound
       S_series(2a) <= (1/3 + ln(3)/2) * 3/(1+2a) * (1 + ln(1+2a))/(1+ln 3)  [decreasing-ratio argument, see text];
       here implemented as the direct a-sup over a in [1, n-2] of the tail expression is NOT used; instead:
         case N >= a:  W1 <= 1.25(4/3)(1+1/a) 1.016 n 2G H_N/(N+1),  n/(N+1)<2
                       W1*F1main <= 1.25(4/3)*1.016*2*2 * [G(1+1/a) * 16 S(2a)] * H/(n+a+1)
                       with  G(1+1/a)*S(2a) <= 0.9 * (1 + ln(1+2a)) <= 0.9 (1 + ln(2n))   (checked, see text)
         case N < a:  S <= Z2/2,  W1*F1main <= 1.25(4/3)*1.016*(Z2/2)*(1+1/a)*16*S(2a)*n/(n+a+1) with a > n/2
       remaining: W2 = 1.25(4/3)(1+1/a)(a/2)Z2,  W2*F1main <= 1.25(4/3)(Z2/2)*16*[(a+1) S(2a)]/(n+a+1),
                  (a+1) S(2a) <= 2*(1/3 + ln(3)/2) for a >= 1 (decreasing) ;  F1 second part and kappa terms."""
    H = Hup(n)
    F1sec = (32 / 3) * Hup(2 * n) / ((n + 2) * (n + 3))
    Smax = S_series(2.0)                              # S(2a) <= S(2) for a >= 1
    F1max = 16 * Smax / (n + 2) + F1sec
    c1 = 1.25 * (4 / 3) * 1.016 * 2 * 2 * 16 * 0.9 * (1 + math.log(2 * n)) * H / (n + 2)
    am = n / 2
    c2 = 1.25 * (4 / 3) * 1.016 * 0.5 * Z2 * (1 + 1 / am) * 16 * S_series(2 * am) * n / (n + am + 1)
    w2 = 1.25 * (4 / 3) * 0.5 * Z2 * 16 * 2 * Smax / (n + 2)
    w1sec = 1.25 * (4 / 3) * 2 * 1.016 * n * 0.5 * Z2 * F1sec
    w2sec = 1.25 * (4 / 3) * 2 * 0.5 * n * Z2 * F1sec
    t3 = (KAP + THETA) * F1max
    return max(c1, c2) + w2 + w1sec + w2sec + t3, 5.3334 / (n + 4)


def sigma2(F0, F1, W, kap=KAP):
    """max_tau min((1 - tau(kap+W))/F0, tau/(1+tau)/F1)  (bisection, float)."""
    k = kap + W
    if F1 <= 0:
        return 1 / F0
    lo, hi = 0.0, 1 / k
    for _ in range(100):
        t = (lo + hi) / 2
        if (1 - t * k) / F0 > t / (1 + t) / F1:
            lo = t
        else:
            hi = t
    t = lo
    return min((1 - t * k) / F0, t / (1 + t) / F1)

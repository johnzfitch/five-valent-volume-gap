"""Inequality (9) of Appendix B (even windows), revision pass 2b, 22 October 2026.
Exact rational check, written for the argument now printed in Appendix B:
  Gamma_N := sum_{i=1}^K 4(1-rho_i)/((2i-1)(2i+3)) + 2N/(N^2-1) >= 2 R_N^2 (N-1)/(N(N+1)),
  N = 2P, K = P-1, rho_i = C(N-1,K-i)/C(N-1,K) = prod_{r<=i} (P-r)/(P+r),
  R_N = 2^(2K+1) K! (K+1)! / ((2K+1)!!)^2.
(a) the inequality itself for every even N = 2..400 (P <= 200);
(b) on the overlap 100 <= P <= 200 with the analytic range, each step of the printed argument
    in exact arithmetic: rho_i <= 1/2 for all i >= k = ceil(sqrt P); the telescoped lower bound
    L = (1/(2k-1)+1/(2k+1))/2 + (1/(2P-1)+1/(2P+1))/2 satisfies Gamma_N > L > 1/(2k+1) > 4/P > RHS;
    P > 8 sqrt(P) + 12 (as (P-12)^2 > 64P with P > 12);
(c) R_2 = 2 and R_{N+2}/R_N = 4P(P+1)/(2P+1)^2 < 1 for P = 1..1000.
Usage (from this directory): python sw_even_window_exact.py > ../outputs/sw_even_window_exact_out.txt"""
from fractions import Fraction as Fr
from math import comb, factorial, isqrt


def dfact(n):
    r = 1
    for x in range(n, 0, -2):
        r *= x
    return r


def R(P):
    K = P - 1
    return Fr(2 ** (2 * K + 1) * factorial(K) * factorial(K + 1), dfact(2 * K + 1) ** 2)


def gamma(P):
    N, K = 2 * P, P - 1
    return sum(Fr(4, (2 * i - 1) * (2 * i + 3)) * (1 - Fr(comb(N - 1, K - i), comb(N - 1, K)))
               for i in range(1, K + 1)) + Fr(2 * N, N * N - 1)


def rhs(P):
    N = 2 * P
    return 2 * R(P) ** 2 * Fr(N - 1, N * (N + 1))


ok_a = True; worst = None; eq = []
for P in range(1, 201):
    g, r = gamma(P), rhs(P)
    if g < r: ok_a = False
    if g == r: eq.append(2 * P)
    elif worst is None or g / r < worst[0]: worst = (g / r, 2 * P)
print(f"(a) even N = 2..400: Gamma_N >= RHS for all: {ok_a}; equality at N = {eq}; "
      f"smallest ratio elsewhere {float(worst[0]):.6f} at N = {worst[1]}")

ok_b = True
for P in range(100, 201):
    k = isqrt(P - 1) + 1                      # ceil(sqrt P) for P >= 2
    assert (k - 1) ** 2 < P <= k ** 2
    rho = Fr(1)
    for i in range(1, P):
        rho *= Fr(P - i, P + i)
        if i >= k and rho > Fr(1, 2): ok_b = False
    L = (Fr(1, 2 * k - 1) + Fr(1, 2 * k + 1)) / 2 + (Fr(1, 2 * P - 1) + Fr(1, 2 * P + 1)) / 2
    chain = gamma(P) > L > Fr(1, 2 * k + 1) > Fr(4, P) > rhs(P)
    big = P > 12 and (P - 12) ** 2 > 64 * P
    if not (chain and big): ok_b = False
print(f"(b) 100 <= P <= 200: rho_i <= 1/2 for i >= ceil(sqrt P), Gamma_N > L > 1/(2k+1) > 4/P > RHS, "
      f"and P > 8 sqrt P + 12: {ok_b}")

ok_c = R(1) == 2 and all(R(P + 1) / R(P) == Fr(4 * P * (P + 1), (2 * P + 1) ** 2) < 1 for P in range(1, 1001))
print(f"(c) R_2 = 2 and R_(N+2)/R_N = 4P(P+1)/(2P+1)^2 < 1 for P = 1..1000: {ok_c}")
print("ALL PASS" if (ok_a and ok_b and ok_c) else "SOME FAIL")

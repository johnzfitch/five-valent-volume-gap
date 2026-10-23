"""Lemma Gamma (even N = 2P, K = P-1), exact rational check:
  Gamma_N := sum_{i=1}^K 4(1-rho_i)/((2i-1)(2i+3)) + 2N/(N^2-1)  >=  2 R_N^2 (N-1)/(N(N+1)),
  rho_i = C(N-1,K-i)/C(N-1,K),  R_N = 2^(2K+1) K! (K+1)! / ((2K+1)!!)^2.
Proved analytically in THEOREM.md for P >= 25; checked here exactly for N = 2..400 (covers P <= 24 with room).
Also checks the analytic tail inequality (1/sqrt P)(arctan sqrt P - 2/sqrt(P-1)) >= 4/P at P = 25 and R_N <= 8K(K+1)/(2K+1)^2."""
from fractions import Fraction as Fr
from math import comb, factorial, atan, sqrt
def dfact(n): 
    r = 1
    for x in range(n, 0, -2): r *= x
    return r
lines = []; worst = None; ok = True; okR = True
for N in range(2, 401, 2):
    K = N // 2 - 1
    G = sum(Fr(4) * (1 - Fr(comb(N - 1, K - i), comb(N - 1, K))) / ((2*i - 1) * (2*i + 3)) for i in range(1, K + 1)) + Fr(2 * N, N * N - 1)
    R = Fr(2 ** (2*K + 1) * factorial(K) * factorial(K + 1), dfact(2*K + 1) ** 2)
    rhs = 2 * R * R * (N - 1) / (N * (N + 1))
    ratio = G / rhs
    if ratio < 1: ok = False
    if K >= 1 and R > Fr(8 * K * (K + 1), (2*K + 1) ** 2): okR = False
    if worst is None or ratio < worst[0]: worst = (ratio, N)
    if N in (2, 4, 6, 8, 10, 20, 48, 50, 100, 400): lines.append(f"N={N}: Gamma_N / RHS = {float(ratio):.6f}")
lines.append(f"all even N in [2,400]: Gamma_N >= RHS: {ok}; minimum ratio {float(worst[0]):.6f} at N={worst[1]} (equality only at N=2)")
lines.append(f"R_N <= 8K(K+1)/(2K+1)^2 for K>=1, N<=400: {okR}")
P = 25
lines.append(f"analytic tail at P=25: (arctan 5 - 2/sqrt 24)/5 = {(atan(5) - 2/sqrt(24))/5:.5f} >= 4/25 = {4/25}")
print("\n".join(lines)); open("../outputs/sw_even_lemma_out.txt", "w").write("\n".join(lines) + "\n")

"""Rigorous certificate (interval arithmetic) of Conjecture SW at the sharp level s* = (15-sqrt 97)/6 for odd N in [5, Nmax],
window a = 0, b = N-2 (all other windows follow by R1, R2), uniformly in real m >= 0.
Exact criterion (THEOREM.md, Prop. O):  sigma_min^2 >= s  <=>  s * S0(m) * SN(m) <= Xi_N(s), where
  S0(m) = sum_k r0(k;m) rho_k,  SN(m) = sum_k rN(k;m) rho_k,  rho_k = 1/((lam_k-1)^2 - s),  lam_k = N-1-2k,
  r0, rN = end weights relative to the central weight (explicit products), Xi_N(s) = (1/4) P'(0)^2 (1/P(1+t)+1/P(1-t))^2.
S0 is nonincreasing and SN nondecreasing in m, so on [ma, mb] it suffices that s*S0(ma)*SN(mb) <= Xi; the last interval
[M, inf) uses SN(inf). Intervals are bisected until the inequality holds with interval arithmetic.
Usage: python sw_odd_cert.py Nmin Nmax"""
import sys, time
from fractions import Fraction as Fr
from mpmath import iv, mp
iv.dps = 40
import os
s = (15 - iv.sqrt(97)) / 6 if "SW_S" not in os.environ else iv.mpf(os.environ["SW_S"]); t = iv.sqrt(s)
def r_list(N, m, top):
    """relative weights r(k) for k = 0..N-1 as exact Fractions; m a Fraction or None (= infinity)."""
    J = (N - 1) // 2; r = [None] * N; r[J] = Fr(1)
    for k in range(J, 0, -1):      # r(k-1) = r(k) * ratio(k)
        if m is None: ratio = Fr(k, N - k)
        elif top: ratio = Fr(k) * (m + k) / ((N + m - k) * (N - k))
        else: ratio = Fr(k) * (N + m - k) / ((m + k) * (N - k))
        r[k - 1] = r[k] * ratio
    for k in range(J + 1, N): r[k] = r[N - 1 - k]
    return r
def S(N, rl, rho): return sum(iv.mpf(x.numerator) / x.denominator * rho[k] for k, x in enumerate(rl))
def cert(N, Mmax=Fr(2) ** 24, maxdepth=60):
    lam = [N - 1 - 2 * k for k in range(N)]; J = (N - 1) // 2
    rho = [1 / ((l - 1) ** 2 - s) for l in lam]
    P = lambda z: iv.fprod([z - l for l in lam])
    Pp = iv.mpf(1)
    for k in range(N):
        if k != J: Pp *= -lam[k]
    Xi = Pp ** 2 * (1 / P(1 + t) + 1 / P(1 - t)) ** 2 / 4
    cacheS0 = {}; cacheSN = {}
    def S0(m):
        if m not in cacheS0: cacheS0[m] = S(N, r_list(N, m, False), rho)
        return cacheS0[m]
    def SN(m):
        key = 'inf' if m is None else m
        if key not in cacheSN: cacheSN[key] = S(N, r_list(N, m, True), rho)
        return cacheSN[key]
    def good(a, b): 
        val = s * S0(a) * SN(b)
        return val.b < Xi.a     # upper end of LHS < lower end of RHS
    # last interval
    if not good(Mmax, None): return False, 'tail', 0
    stack = [(Fr(0), Mmax, 0)]; nint = 1
    while stack:
        a, b, d = stack.pop()
        if good(a, b): nint += 1; continue
        if d > maxdepth: return False, (a, b), nint
        mid = (a + b) / 2 if b - a > 1 or a == 0 else (a + b) / 2
        if a > 0 and b / a > 2: mid = Fr(int((a * b) ** 0.5 * 2 ** 20), 2 ** 20)   # geometric split on wide intervals
        stack += [(a, mid, d + 1), (mid, b, d + 1)]
    return True, None, nint
if __name__ == '__main__':
    N0, N1 = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time(); allok = True; lines = []
    for N in range(N0, N1 + 1, 2):
        ok, where, n = cert(N)
        allok &= ok
        lines.append(f"N={N}: certified={ok} with {n} m-intervals" + ("" if ok else f" FAILED at {where}"))
        print(lines[-1], flush=True)
    lines.append(f"all odd N in [{N0},{N1}] certified: {allok}; wall {time.time()-t0:.0f}s")
    print(lines[-1]); open(f"../outputs/sw_odd_cert_{N0}_{N1}.txt", "w").write("\n".join(lines) + "\n")

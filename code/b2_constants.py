"""Interval certificate for the two numerical constants in the proof of Theorem B2 (j1 = 1/2).

Even chains (doubled n >= 4, K = floor((n-1)/2)):  q >= sqrt(8n / (H_K^2 + H_K^(2))).
Odd chains: q >= min(6 sqrt2/pi, sqrt(4(n+4)/H_K)).
We show, in mpmath interval arithmetic, that both right-hand sides are >= 6 sqrt2 / pi for
4 <= n <= NMAX, and analytically beyond: for n >= NMAX both are increasing functions bounded
below by elementary expressions (see the paper, Appendix C).
"""
import sys
import mpmath as mp
from mpmath import iv

iv.dps = 40
NMAX = 20000
target = iv.mpf(6) * iv.sqrt(2) / iv.pi
H = [iv.mpf(0)]
H2 = [iv.mpf(0)]
for k in range(1, NMAX):
    H.append(H[-1] + iv.mpf(1) / k)
    H2.append(H2[-1] + iv.mpf(1) / (k * k))
worst_even = (iv.mpf(1e9), None)
worst_odd = (iv.mpf(1e9), None)
for n in range(3, NMAX + 1):
    K = (n - 1) // 2
    be = iv.sqrt(8 * iv.mpf(n) / (H[K] ** 2 + H2[K]))
    bo = iv.sqrt(4 * iv.mpf(n + 4) / H[K])
    assert be.a > target.b and bo.a > target.b, n
    if be.a < worst_even[0].a:
        worst_even = (be, n)
    if bo.a < worst_odd[0].a:
        worst_odd = (bo, n)
# tail: for n >= NMAX, H_K <= 1 + ln n and H_K^(2) <= pi^2/6, and 8n/((1+ln n)^2 + pi^2/6) is increasing
tail_even = iv.sqrt(8 * iv.mpf(NMAX) / ((1 + iv.log(NMAX)) ** 2 + iv.pi ** 2 / 6))
tail_odd = iv.sqrt(4 * iv.mpf(NMAX + 4) / (1 + iv.log(NMAX)))
with open(sys.argv[1], "w") as f:
    print(f"target 6 sqrt2/pi in {target}", file=f)
    print(f"even-chain bound, 3 <= n <= {NMAX}: minimum {worst_even[0]} at n = {worst_even[1]}", file=f)
    print(f"odd-chain bound,  3 <= n <= {NMAX}: minimum {worst_odd[0]} at n = {worst_odd[1]}", file=f)
    print(f"tails at n = {NMAX}: even >= {tail_even.a}, odd >= {tail_odd.a} (both increasing in n)", file=f)
    print("ALL PASS", file=f)

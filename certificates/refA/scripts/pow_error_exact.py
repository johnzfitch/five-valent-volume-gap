"""Exact worst-case relative error of Bh = float(Fraction(r))**0.5 against sqrt(r), over every squared-link value r
that can occur in (a) the finite certificate (all doubled spins <= 38) and (b) the j1 = 5/2 recheck (2j1 = 5,
other doubled spins <= 70).  The certificates assume |Bh - sqrt r| <= 2.0001 u sqrt r.  For each value we bound
e = |Bh/sqrt(r) - 1| exactly: e <= E  <=>  r (1-E)^2 <= Bh^2 <= r (1+E)^2 (rational arithmetic).
Reports the max over all values of the error in units of u, computed exactly to 1e-6 u resolution by bisection
only for values where x**0.5 != math.sqrt(x) (for the others Bh is the correctly rounded sqrt of the correctly
rounded r, so e <= (1+u)^(1/2)(1+u) - 1 < 1.51 u)."""
import math, sys, time
from fractions import Fraction as Fr
U = Fr(1, 2 ** 53)
def link_pairs(Svals, M):
    vals = set()
    for n_ in range(4, 2 * M + 3):
        for m in range(0, M + 1):
            if (n_ - m) % 2 or m >= n_ - 2: continue
            for k in range(m + 2, n_, 2):
                f1 = (n_ * n_ - k * k) * (k * k - m * m)
                if f1 <= 0: continue
                for S in Svals:
                    for C in range(0, 2 * M + 1):
                        if (S + C + k) % 2: continue
                        A2, B2 = S + C + 2, abs(S - C)
                        f2 = (A2 * A2 - k * k) * (k * k - B2 * B2)
                        if f2 > 0: vals.add((f1 * f2, 4096 * (k * k - 1)))
    return vals
def rel_err_units(r, y):
    """exact |y/sqrt(r) - 1| / u, to resolution 1e-6 (upper end returned)."""
    y2 = Fr(y) ** 2
    lo, hi = Fr(0), Fr(8)
    for _ in range(40):
        mid = (lo + hi) / 2; E = mid * U
        if r * (1 - E) ** 2 <= y2 <= r * (1 + E) ** 2: hi = mid
        else: lo = mid
    return float(hi)
for label, Svals, M in [("finite cert, doubled spins <= 38", range(1, 39), 38), ("j1=5/2 recheck, doubled spins <= 70", [5], 70)]:
    t0 = time.time(); vals = link_pairs(Svals, M)
    worst = 0.0; nbad = 0; wv = None
    for (a, b) in vals:
        x = float(Fr(a, b)); y = x ** 0.5
        if y != math.sqrt(x):
            nbad += 1; e = rel_err_units(Fr(a, b), y)
            if e > worst: worst, wv = e, (a, b)
    print(f"{label}: {len(vals)} distinct squared links; {nbad} with x**0.5 != sqrt(x); worst exact error among those "
          f"{worst:.6f} u at r={wv}; bound assumed by the certificate 2.0001 u -> {'HOLDS' if worst <= 2.0001 else 'VIOLATED'}  ({time.time()-t0:.0f}s)", flush=True)

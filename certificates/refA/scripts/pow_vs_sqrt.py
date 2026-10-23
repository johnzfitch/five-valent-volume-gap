"""Does CPython's x**0.5 (C-library pow) agree with the correctly rounded math.sqrt on this machine?
The finite certificate builds Bh with float(Fraction)**0.5 but its error model assumes a correctly rounded sqrt.
Test 1: random doubles over a wide exponent range.  Test 2: every squared link value float(link2) that occurs
for doubled spins <= 38 (superset: all parameter combinations of the closed form, deduplicated)."""
import math, random, sys, time
from fractions import Fraction as Fr
random.seed(1)
bad = 0; n = 0; t0 = time.time()
for i in range(3_000_000):
    x = math.ldexp(random.random() + 0.5, random.randint(-60, 80))
    n += 1
    if x ** 0.5 != math.sqrt(x): bad += 1
print(f"random doubles: {n} tested, {bad} disagreements x**0.5 vs math.sqrt  ({time.time()-t0:.1f}s)")
# all squared links: num/(4096(k^2-1)), num = (n^2-k^2)(k^2-m^2)(A^2-k^2)(k^2-B^2) with doubled labels <= 78
vals = set(); t0 = time.time()
M = 38
for n_ in range(4, 2 * M + 3):            # n = j+j'+2 doubled (2..78)
    for m in range(0, M + 1):
        if (n_ - m) % 2: continue
        if m >= n_ - 2: continue
        for k in range(m + 2, n_, 2):      # upper label strictly inside (m, n)
            f1 = (n_ * n_ - k * k) * (k * k - m * m)
            if f1 <= 0: continue
            for S in range(1, M + 1):
                for C in range(0, 2 * M + 1):
                    if (S + C + k) % 2: continue
                    A2, B2 = S + C + 2, abs(S - C)
                    f2 = (A2 * A2 - k * k) * (k * k - B2 * B2)
                    if f2 <= 0: continue
                    vals.add((f1 * f2, 4096 * (k * k - 1)))
print(f"distinct (num,den) link pairs for doubled spins <= {M}: {len(vals)}  ({time.time()-t0:.1f}s)")
bad2 = 0
for (a, b) in vals:
    x = float(Fr(a, b))
    if x ** 0.5 != math.sqrt(x): bad2 += 1
print(f"squared-link values: {bad2} disagreements x**0.5 vs math.sqrt  ({time.time()-t0:.1f}s)")

"""Conjecture SW for ALL windows at once via the top-down Riccati recursion (riccati.V_seq):
SW at level s for every window [a, N-2] of H(n=m+N, m)  <=>  V_k(s) - s*I is PSD for k = 1..N-1.
(Windows with b < N-2 follow by R1 of the closing note.) The 2x2 PSD test is done with a relative tolerance.
Also reports, per (N,m), the smallest eigenvalue of V_k - s over k, and cross-checks against direct SVD for N <= 40.
Usage: python sw_riccati_scan.py Nmax"""
import sys, time
import numpy as np
from riccati import V_seq
from stark import Bwin, smin
s = (15 - np.sqrt(97)) / 6
Nmax = int(sys.argv[1]); t0 = time.time()
ms = [x / 2 for x in range(0, 81)] + [100.0, 1e3, 1e6]
bad = []; mism = 0; worst = (9e9, None); nwin = 0
for N in range(2, Nmax + 1):
    for m in ms:
        V = V_seq(N, m, s)
        lo = min(np.linalg.eigvalsh(V[k]).min() - s for k in V)
        nwin += N - 1
        if lo < -1e-9 * max(1, N): bad.append((N, m, lo))
        if lo < worst[0]: worst = (lo, (N, m))
        if N <= 40 and m in (0.0, 0.5, 3.0, 17.5):
            direct = min(smin(Bwin(N, m, a)) for a in range(N - 1)) ** 2 - s
            if (direct >= -1e-9) != (lo >= -1e-9): mism += 1
print(f"windows tested: {nwin} (all a, b=N-2), N<= {Nmax}, m in 0..40 step 1/2 and 100, 1e3, 1e6")
print(f"violations of V_k >= s*: {len(bad)} {bad[:5]}")
print(f"smallest min-eig(V_k - s*) = {worst[0]:.3e} at (N,m) = {worst[1]}  (0 at the equality case N=3, m=0)")
print(f"Riccati vs direct SVD sign mismatches (N<=40, 4 m values): {mism}")
print(f"wall {time.time()-t0:.0f}s")

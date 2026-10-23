"""Compare T with Omega_off: same graph, links |T| = (b/2)|Omega| (X) and (c/2)|Omega| (D).
Omega_off links = 2|T|/b or 2|T|/c from the closed form; spectrum of Omega_off vs Omega - diag.
Usage: python explore_omega_off.py SMIN SMAX LMAX"""
import sys, itertools, numpy as np
from t1core import grid, valid, qmin

S0, S1, L = map(int, sys.argv[1:4])
rat = []
for S in range(S0, S1 + 1):
    for legs in itertools.combinations_with_replacement(range(1, L + 1), 4):
        if not valid(S, *legs):
            continue
        st, links = grid(S, *legs)
        if len(st) < 2:
            continue
        n = len(st)
        Om = np.zeros((n, n))
        for i, k, w2 in links:
            (b1, c1), (b2, c2) = st[i], st[k]
            fac = (b2 / 2) if b2 != b1 else (c2 / 2)    # upper label
            Om[i, k] = Om[k, i] = 2 * np.sqrt(w2) / fac
        ev = np.abs(np.linalg.eigvalsh(Om))
        nz = ev[ev > 1e-9 * max(1, ev.max())]
        q, dk, nn = qmin(S, *legs)
        if q is None or len(nz) == 0:
            continue
        rat.append((q / 8 / nz.min(), q, nz.min(), S, legs))
rat.sort()
print('min (q/8)/lambda(Omega_off):', rat[0]); print('max:', rat[-1])
for r in rat[:10]:
    print(r)

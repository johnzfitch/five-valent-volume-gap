"""Validate the closed-form grid against the full tensor-product construction.
Usage: python validate_core.py MAXDOUBLED   (all tuples with every doubled spin <= MAX)"""
import sys, itertools, numpy as np
from t1core import brute, Tmatrix, valid

MX = int(sys.argv[1]) if len(sys.argv) > 1 else 3
worst = 0.0; count = 0; dimfail = 0
for t in itertools.product(range(1, MX + 1), repeat=5):
    if not valid(*t):
        continue
    eb = brute(*t)
    st, W = Tmatrix(*t)
    eg = np.sort(np.linalg.eigvalsh(W)) if len(st) else np.array([])
    if len(eb) != len(eg):
        dimfail += 1
        print('dim mismatch', t, len(eb), len(eg)); continue
    if len(eb) == 0:
        continue
    d = np.max(np.abs(np.sort(np.abs(eb)) - np.sort(np.abs(eg))))
    worst = max(worst, d / max(1, np.abs(eb).max())); count += 1
print(f'tuples compared {count}, dim mismatches {dimfail}, max rel |spec| diff {worst:.2e}')

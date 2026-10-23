"""Spectrum of auxiliary operators on D = rectangle cap strip |b-c|<=s (corner NOT imposed):
 K_F : links h1(b) sqrt(F(d-1/2)), h2(c) sqrt(F(d+1/2))
 K_G : links h1(b) sqrt(G(u)), ... on u>=s  (for comparison)
Usage: python explore_KF.py S a2 a3 a4 a5"""
import sys, numpy as np
from fractions import Fraction as Fr
from t1core import pair_params, h2stark

S, a2, a3, a4, a5 = map(int, sys.argv[1:6])
n1, m1 = pair_params(a2, a3); n2, m2 = pair_params(a4, a5)
s = Fr(S, 2)
F = lambda t: (s + Fr(1, 2)) ** 2 - t * t
D = [(b, c) for b in range(m1, n1 - 1, 2) for c in range(m2, n2 - 1, 2) if abs(b - c) <= S and (b + c + S) % 2 == 0]
idx = {x: i for i, x in enumerate(D)}
n = len(D); K = np.zeros((n, n))
for (b, c), i in idx.items():
    d = Fr(b - c, 2)
    if (b - 2, c) in idx:
        w = float(h2stark(n1, m1, b) * F(d - Fr(1, 2))) ** 0.5
        K[i, idx[(b - 2, c)]] = K[idx[(b - 2, c)], i] = w
    if (b, c - 2) in idx:
        w = float(h2stark(n2, m2, c) * F(d + Fr(1, 2))) ** 0.5
        K[i, idx[(b, c - 2)]] = K[idx[(b, c - 2)], i] = w
ev = np.sort(np.linalg.eigvalsh(K))
print('dim', n, 'spec K_F:', np.round(ev, 5))
print('spec K_F^2 :', np.round(np.sort(ev ** 2), 4))

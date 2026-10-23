"""Where do low modes live? Print the mass of the lowest nonzero singular vector of T's block
and of W0's block on each diagonal d and in u-bands.  Usage: python explore_modes.py S a2 a3 a4 a5"""
import sys, numpy as np
from t1core import Tmatrix, colour
from w0core import W0_block, pairings

S, a2, a3, a4, a5 = map(int, sys.argv[1:6])


def report(B, cols, rows, label):
    U, sv, Vt = np.linalg.svd(B)
    tol = 1e-9 * max(1, sv.max())
    k = np.where(sv > tol)[0][-1]
    v = Vt[k]; u = U[:, k]
    print(label, 'sigma_min_nonzero =', sv[k], ' #cols', len(cols), '#rows', len(rows), 'rank', k + 1)
    for name, vec, pts in (('col-vector', v, cols), ('row-vector', u, rows)):
        md = {}; mu = {}
        for x, (b, c) in zip(vec, pts):
            d = (b - c) / 2; uu = (b + c) / 2
            md[d] = md.get(d, 0) + x * x
            band = int((uu - S / 2) // 4)
            mu[band] = mu.get(band, 0) + x * x
        print('  ', name, 'by d:', ' '.join(f'{d:+.1f}:{m:.2f}' for d, m in sorted(md.items())))
        print('  ', name, 'by u-band(4):', ' '.join(f'{b}:{m:.2f}' for b, m in sorted(mu.items())[:12]))


st, W = Tmatrix(S, a2, a3, a4, a5)
cols = [i for i, x in enumerate(st) if colour(S, x) == 0]
rows = [i for i, x in enumerate(st) if colour(S, x) == 1]
report(W[np.ix_(rows, cols)], [st[i] for i in cols], [st[i] for i in rows], 'T (23|45):')
for p1, p2 in pairings(a2, a3, a4, a5):
    for col in (0, 1):
        B0, c_, r_ = W0_block(S, p1, p2, col)
        if B0.size:
            report(B0, c_, r_, f'W0 {p1}|{p2} col={col}:')

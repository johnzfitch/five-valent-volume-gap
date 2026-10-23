"""Profile of the lowest nonzero mode of W0 over the diagonals d and over u (geometric bins).
Usage: python mode_profile_u.py S a2 a3 a4 a5   (doubled)"""
import sys
import numpy as np, scipy.sparse.linalg as spl
from w0sparse import build
S = int(sys.argv[1]); legs = tuple(map(int, sys.argv[2:6]))
sites, W = build(S, *legs)
vals, vecs = spl.eigsh(W.tocsc(), k=8, sigma=1e-3, which='LM')
o = np.argsort(np.abs(vals))
for i in o:
    if abs(vals[i]) < 1e-7: continue
    v = vecs[:, i]; print('eig', vals[i]); break
d = np.array([(b - c) / 2 for b, c in sites]); u = np.array([(b + c) / 2 for b, c in sites])
for dd in sorted(set(d)):
    print('d=%+.1f mass %.3f' % (dd, (v[d == dd] ** 2).sum()))
# u profile in bins
umax = u.max(); s = S / 2
edges = np.unique(np.round(np.geomspace(s, umax + 1, 25)))
for a, bb in zip(edges[:-1], edges[1:]):
    m = (u >= a) & (u < bb)
    print('u in [%g,%g): mass %.4f  per-unit %.5f' % (a, bb, (v[m] ** 2).sum(), (v[m] ** 2).sum() / max(1, bb - a)))

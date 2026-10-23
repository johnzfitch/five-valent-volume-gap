"""Exact sign structure of X = J1.(J2xJ3) and D = J1.(J4xJ5) on the grid basis.
Leg order in hv_core chain coupling: (j2, j3, j1, j4 | j5)  => a2 = b (spin of J2+J3), a3 = c = r (spin of
J1+J2+J3 = spin of J4+J5), last leg j5.  In this ordering hv_core legs are 0:j2 1:j3 2:j1 3:j4.
X = J1.(J2xJ3) = qhat(2,0,1)/(-4i)  (cyclic), D' = J1.((J2+J3)xJ4) = qhat(2,0,3)+qhat(2,1,3)  (= J1.(J4xJ5) on Inv).
Check: (i) X is zero except b->b+-1 at fixed c; D zero except c->c+-1 at fixed b; (ii) the product of the four
signed hoppings around every unit plaquette; (iii) candidate gauge formula for the signs.
Float entries (hv_core, float64 from exact signed squares); signs are exact since magnitudes are >~1e-3."""
import sys, itertools
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from hv_core import Vertex

def ops(J):  # J = doubled (j1,j2,j3,j4,j5)
    j1, j2, j3, j4, j5 = J
    v = Vertex((j2, j3, j1, j4, j5))
    if v.dim == 0: return None
    X = v.qhat(2, 0, 1) / 4.0          # real antisym; -4i J1.(J2xJ3) = qhat  -> use qhat/4 as 'X' up to the factor -i
    D = (v.qhat(2, 0, 3) + v.qhat(2, 1, 3)) / 4.0
    sites = [(c[1], c[2]) for c in v.basis]   # (2b, 2c)
    return v, X, D, sites

bad_struct = 0; flux = {}; ncase = 0
for J in itertools.product(range(1, 6), repeat=5):
    if sum(J) % 2: continue
    r = ops(J)
    if r is None: continue
    v, X, D, sites = r
    ncase += 1
    idx = {s: i for i, s in enumerate(sites)}
    for i, (b, c) in enumerate(sites):
        for k, (b2, c2) in enumerate(sites):
            if abs(X[i, k]) > 1e-12 and not (c2 == c and abs(b2 - b) == 2): bad_struct += 1
            if abs(D[i, k]) > 1e-12 and not (b2 == b and abs(c2 - c) == 2): bad_struct += 1
    for (b, c) in sites:
        p = [(b, c), (b + 2, c), (b + 2, c + 2), (b, c + 2)]
        if all(q in idx for q in p):
            i0, i1, i2, i3 = [idx[q] for q in p]
            prod = X[i1, i0] * D[i2, i1] * X[i2, i3] * D[i3, i0]   # around: (b,c)->(b+1,c)->(b+1,c+1)<-(b,c+1)<-(b,c)
            if abs(prod) < 1e-12:
                key = 'zero'
            else:
                key = int(np.sign(prod))
            flux[key] = flux.get(key, 0) + 1
print(f"cases (all 2j_i<=5, integer total): {ncase}; entries outside the grid pattern: {bad_struct}")
print("sign of X(b->b+1,c) D(b+1,c->c+1) / [X(b->b+1,c+1) D(b,c->c+1)] over all unit plaquettes:", flux)

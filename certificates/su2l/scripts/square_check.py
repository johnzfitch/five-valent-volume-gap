"""How often is the best zigzag reading square (L = B^{-1} exactly, so 8/||L|| = q trivially)?"""
import numpy as np, sys
from zigzag import readings, leftinv, tdiag
from t1core import qmin
from w0core import pairings, covered
S, count, lo, hi, seed = (int(x) for x in sys.argv[1:6])
rng = np.random.default_rng(seed)
n = sq = sq1 = tall = tallr = 0; worst_tall = 9
while n < count:
    legs = tuple(sorted(int(x) for x in rng.integers(lo, hi + 1, 4)))
    if (S + sum(legs)) % 2 or 1 in legs or sum(1 for a in legs if a <= 6) >= 3: continue
    if covered(S, *legs) not in ('W2', 'open'): continue
    q = qmin(S, *legs)[0]
    if q is None: continue
    best = None
    for (p1, p2) in pairings(*legs):
        lg = (p1[0], p1[1], p2[0], p2[1]); data = tdiag(S, lg)
        for lab, st in readings(S).items():
            b = leftinv(S, lg, st, data)
            if b is None: continue
            ncol = sum(len(data[2][c]) for (_, c, _) in st); nrow = sum(len(data[2][r]) for (r, _, _) in st)
            if best is None or b > best[0]: best = (b, ncol == nrow)
    n += 1
    if best is None: continue
    if best[1]:
        sq += 1; sq1 += abs(best[0] / q - 1) < 1e-8
    else:
        tall += 1; worst_tall = min(worst_tall, best[0] / q)
print(f"2s={S}: {n} tuples; best reading square in {sq} (ratio exactly 1 in {sq1}); tall in {tall}, worst ratio among tall {worst_tall:.4f}")

"""Survey for j1 >= 5/2: classify tuples (sorted legs 2..5 by S4 symmetry) and record
qmin, best-pairing lambda(W0) for W2/open types.  Usage: python survey_types.py SMIN SMAX LMAX"""
import sys, itertools, math, collections
from t1core import qmin, valid
from w0core import covered, pairings, W0_block, lam

S0, S1, L = map(int, sys.argv[1:4])
stats = collections.defaultdict(lambda: [0, 1e9, None, 1e9, None])
for S in range(S0, S1 + 1):
    for legs in itertools.combinations_with_replacement(range(1, L + 1), 4):
        if not valid(S, *legs):
            continue
        q, dk, n = qmin(S, *legs)
        if q is None:
            continue
        typ = covered(S, *legs)
        rec = stats[(S, typ)]
        rec[0] += 1
        if q < rec[1]:
            rec[1] = q; rec[2] = legs
        if typ in ('W2', 'open'):
            best = 0
            for p1, p2 in pairings(*legs):
                for col in (0, 1):
                    B0, c_, r_ = W0_block(S, p1, p2, col)
                    l = lam(B0)
                    if l is not None:
                        best = max(best, l)
            if best < rec[3]:
                rec[3] = best; rec[4] = legs
for k in sorted(stats):
    r = stats[k]
    print(f'2j1={k[0]:3d} {k[1]:10s} n={r[0]:6d} qmin={r[1]:9.4f} at {r[2]}  bestlamW0min={r[3]:.4f} at {r[4]}')

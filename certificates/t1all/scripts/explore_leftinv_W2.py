"""Layer left-inverse bound on integer-j1 W2 tuples (legs all integer or all half-integer >= 3/2).
Usage: python explore_leftinv_W2.py SLIST(comma, doubled) LMAX"""
import sys, itertools, numpy as np
from t1core import qmin, valid
from w0core import pairings
from explore_leftinv import leftinv

Ss = [int(x) for x in sys.argv[1].split(',')]; L = int(sys.argv[2])
for S in Ss:
    res = []; nol = []
    for legs in itertools.combinations_with_replacement(range(1, L + 1), 4):
        if not valid(S, *legs):
            continue
        par = {a % 2 for a in legs}
        if len(par) != 1 or (1 in par and min(legs) < 3):
            continue
        q, dk, n = qmin(S, *legs)
        if q is None:
            continue
        best = None
        for p1, p2 in pairings(*legs):
            for rc in (0, 1):
                r = leftinv(S, p1, p2, rc)
                if r is not None and (best is None or r['invL'] > best[0]['invL']):
                    best = (r, p1, p2, rc)
        if best is None:
            nol.append(legs); continue
        res.append((8 * best[0]['invL'], q, legs, best[1], best[2], best[3]))
    res.sort()
    print(f'2s={S}: W2 tuples {len(res)+len(nol)}, with left inverse {len(res)}, without {len(nol)} {nol[:6]}')
    for r in res[:8]:
        print('   8/|L|=%.4f q=%.3f legs=%s %s|%s rc=%d' % r)

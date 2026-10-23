"""Exploration: with legs 2..5 FIXED, is qmin(s+1) >= qmin(s)?  (s steps by 1, same lattice)
Usage: python explore_mono2.py LMAX"""
import sys, itertools
from t1core import qmin, valid
L = int(sys.argv[1])
viol = []; pairs = 0; worst = (1e9, None)
for legs in itertools.combinations_with_replacement(range(1, L + 1), 4):
    tot = sum(legs)
    prevq = {}
    for S in range(1, tot + 1):
        if not valid(S, *legs):
            continue
        q, dk, n = qmin(S, *legs)
        if q is not None and (S - 2) in prevq and prevq[S - 2] is not None:
            pairs += 1
            r = q / prevq[S - 2]
            if r < worst[0]:
                worst = (r, (S, legs))
            if r < 1 - 1e-9:
                viol.append(((S, legs), prevq[S - 2], q))
        prevq[S] = q
print('pairs', pairs, 'violations', len(viol), 'worst ratio', worst)
for v in viol[:30]:
    print(v)

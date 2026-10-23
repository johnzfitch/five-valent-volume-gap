"""Worst monotonicity violations restricted to tuples with no leg 1/2 and 2j1 >= 5 (both endpoints).
Form A: legs fixed, j1 -> j1+1.  Form B: (j2,j3,j4) and M = j1 - j5 fixed, j1 -> j1 + 1/2."""
import itertools
from t1core import qmin, valid
worstA = (9, None); worstB = (9, None); nA = vA = nB = vB = 0
for legs in itertools.combinations_with_replacement(range(2, 13), 4):
    prev = {}
    for S in range(3, sum(legs) + 1):
        if not valid(S, *legs): continue
        q = qmin(S, *legs)[0]
        if q is not None and S >= 7 and prev.get(S - 2) is not None:
            nA += 1; r = q / prev[S - 2]
            if r < 1 - 1e-9: vA += 1
            if r < worstA[0]: worstA = (r, (S - 2, S, legs, prev[S - 2], q))
        prev[S] = q
for a2, a3, a4 in itertools.combinations_with_replacement(range(2, 9), 3):
    for Mx in range(-(a2 + a3 + a4), a2 + a3 + a4 + 1):
        pq = None
        for S in range(5, 25):
            a5 = S - Mx
            if a5 < 2 or not valid(S, a2, a3, a4, a5): continue
            q = qmin(S, a2, a3, a4, a5)[0]
            if q is not None and pq is not None:
                nB += 1; r = q / pq[0]
                if r < 1 - 1e-9: vB += 1
                if r < worstB[0]: worstB = (r, (pq[1], (S, a2, a3, a4, a5), pq[0], q))
            pq = (q, (S, a2, a3, a4, a5)) if q is not None else None
print('Form A (legs fixed, no leg 1/2, legs<=6, 2j1>=5 at both ends): pairs', nA, 'violations', vA, 'worst', worstA)
print('Form B (fixed j2,j3,j4,M; no leg 1/2, legs<=4 (j5 any), 5<=2j1<=24): pairs', nB, 'violations', vB, 'worst', worstB)

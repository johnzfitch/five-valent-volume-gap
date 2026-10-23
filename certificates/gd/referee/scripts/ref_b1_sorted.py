"""Referee: Theorem B1's per-sigma constant in BR's sorted convention.
Orbit-1 sigma <-> distinguished triple T of sorted positions; operator 8 J_a.(J_b x J_c).
Sorted infimum over sorted tuples 2j_max <= M.  Usage: python ref_b1_sorted.py M OUTFILE"""
import sys, os, itertools, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_ops import bc_operator, qvals
M = int(sys.argv[1]); out = []
for T in itertools.combinations(range(5), 3):
    O = [i for i in range(5) if i not in T]
    best = (math.inf, None)
    for J in itertools.combinations_with_replacement(range(1, M + 1), 5):
        if sum(J) % 2: continue
        S, A = bc_operator((J[T[0]], J[T[1]], J[T[2]], J[O[0]], J[O[1]]), 2.0, 0.0)
        if len(S) < 2: continue
        q, k = qvals(A)
        if q.size and q[0] < best[0]: best = (q[0], J)
    out.append(f'triple (sorted positions) {tuple(t+1 for t in T)}: sorted inf over 2j_max<={M} = {best[0]:.10f} at {best[1]}'
               f'  [2sqrt3={2*math.sqrt(3):.6f}, 4sqrt2={4*math.sqrt(2):.6f}]')
print('\n'.join(out)); open(sys.argv[2], 'w').write('\n'.join(out) + '\n')

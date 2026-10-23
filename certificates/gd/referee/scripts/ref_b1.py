"""Referee: Theorem B1 (orbit 1, operator 8 J1.(J2xJ3)), all labelled spins 2j_i <= M,
with my signed (b,c) operator (alpha,beta) = (2,0).
Usage: python ref_b1.py M OUTFILE
"""
import sys, os, itertools, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ref_ops import bc_operator, qvals

M = int(sys.argv[1]); outf = sys.argv[2]
g = 2 * math.sqrt(3)
gmin = math.inf; argmin = None; viol = 0; eq = []; n = 0; nz = 0
for J in itertools.product(range(1, M + 1), repeat=5):
    if sum(J) % 2:
        continue
    # symmetry: legs 2,3 and 4,5 enter symmetrically up to sign; keep all for simplicity
    if J[1] > J[2] or J[3] > J[4]:
        continue
    S, A = bc_operator(J, 2.0, 0.0)
    if len(S) == 0:
        continue
    n += 1
    q, k = qvals(A)
    if q.size == 0:
        continue
    nz += 1
    if q[0] < gmin - 1e-12:
        gmin, argmin = q[0], J
    if q[0] < g - 1e-9:
        viol += 1
    if abs(q[0] - g) < 1e-9:
        eq.append(J)
pred = all(J[0] == J[1] == J[2] == 1 and abs(J[3] - J[4]) == 1 for J in eq)
# predicted equality set within the enumeration
predset = [J for J in itertools.product(range(1, M + 1), repeat=5) if sum(J) % 2 == 0 and J[1] <= J[2] and J[3] <= J[4]
           and J[0] == J[1] == J[2] == 1 and abs(J[3] - J[4]) == 1]
lines = [f'tuples with Inv != 0 (2j_i <= {M}, j2<=j3, j4<=j5): {n}; with nonzero spectrum: {nz}',
         f'min q_min = {gmin:.12f} at doubled spins {argmin}; 2*sqrt(3) = {g:.12f}',
         f'violations of q >= 2sqrt3: {viol}',
         f'equality cases: {len(eq)}; all of form (1/2,1/2,1/2; j4,j5) with |j4-j5|=1/2: {pred};'
         f' equals predicted set: {sorted(eq) == sorted(predset)}']
print('\n'.join(lines))
open(outf, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

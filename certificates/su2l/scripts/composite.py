"""Composite transfer norms of the zigzag reading in the T-picture, against the J_x(s) scalar products.
Reading 'down': X_r = R_{r-1}^+ R_r^T (V_{r+1}->V_{r-1}), r odd.  Product from the top: X_{r1} X_{r1+2} ... ."""
import numpy as np, sys
from tladder import tblocks
S = int(sys.argv[1]); legs = tuple(int(x) for x in sys.argv[2].split(',')); dirn = sys.argv[3]
sites, W, V, N = tblocks(S, legs)
s = S/2
R = [W[np.ix_(V[k+1], V[k])] for k in range(S)]
ell = lambda k: 0.5*np.sqrt((s-(k-s))*(s+(k-s)+1))
if dirn == 'down':
    steps = [r for r in range(S-2, 0, -2)]           # rows r with two columns r-1, r+1
    X = {r: np.linalg.pinv(R[r-1]) @ R[r].T for r in steps}
    jx = {r: ell(r)/ell(r-1) for r in steps}
else:
    steps = [r for r in range(2, S, 2)]
    X = {r: np.linalg.pinv(R[r].T) @ R[r-1] for r in steps}
    jx = {r: ell(r-1)/ell(r) for r in steps}
print("steps", steps)
worst = 0
for i in range(len(steps)):
    P = np.eye(X[steps[i]].shape[1]); pj = 1.0; pf = 1.0
    for j in range(i, len(steps)):
        P = X[steps[j]] @ P if j > i else X[steps[i]]
        pj *= jx[steps[j]]
        pf *= np.linalg.norm(X[steps[j]], 2) / jx[steps[j]]
        ratio = np.linalg.norm(P, 2) / pj
        worst = max(worst, ratio)
        print(f"from step {steps[i]} through {steps[j]}: ||prod||/J_x-prod = {ratio:.4f}   product of single-step flat factors = {pf:.3f}")
print("max composite/J_x ratio", worst)

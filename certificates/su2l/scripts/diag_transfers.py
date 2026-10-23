"""Per-step diagnostics of the d-direction reading in the T-picture.
For reading 'down' (rows odd k, solve x_{k-1} from row k given x_{k+1}):
   transfer  X_k = R_{k-1}^+ R_k^T  : V_{k+1} -> V_{k-1}  (R = l_s * N)
   flat factor: theta_k = sup{t : G^+_{k-1}... } computed as 1/||N_{k-1}^+ N_k^T||^2
For 'up' (rows even k, solve x_{k+1} from row k given x_{k-1}):  X = (R_k^T)^+ R_{k-1}."""
import numpy as np, sys
from tladder import tblocks

def diag(S, legs, direction):
    sites, W, V, N = tblocks(S, legs)
    s = S/2
    ell = lambda k: 0.5*np.sqrt((s-(k-s))*(s+(k-s)+1))   # link between k and k+1
    out = []
    if direction == 'down':
        for r in range(S, 0, -2):
            if r+1 > S: continue
            A = N[r-1]; Bm = N[r].T
            nrm = np.linalg.norm(np.linalg.pinv(A) @ Bm, 2)
            out.append((r, ell(r)/ell(r-1), nrm, np.linalg.svd(A, compute_uv=False).min()))
    else:
        for r in range(0, S+1, 2):
            if r-1 < 0 or r >= S: continue
            A = N[r].T; Bm = N[r-1]
            nrm = np.linalg.norm(np.linalg.pinv(A) @ Bm, 2)
            out.append((r, ell(r-1)/ell(r), nrm, np.linalg.svd(A, compute_uv=False).min()))
    return out

if __name__ == '__main__':
    S = int(sys.argv[1]); legs = tuple(int(x) for x in sys.argv[2].split(',')); dirn = sys.argv[3]
    for r, jx, nrm, smin in diag(S, legs, dirn):
        print(f"row k={r}: J_x ratio {jx:.4f}  flat transfer ||N^+N^T|| {nrm:.4f} (theta^-1/2)  smin(solving N) {smin:.3f}")

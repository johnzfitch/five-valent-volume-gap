"""Step-optimal oblique left inverses for the zigzag reading in the T-picture (reading 'down' or 'up').
At a step solving x_new from one row:  Rs (solving block, injective), Ro (other block into that row).
Left inverses Y = Rs^+ + Z P_K with K = ker Rs^T.  Choice Z = -A B^+ with A = Rs^+ Ro, B = P_K Ro
minimizes ||Y Ro|| (= ||A P_kerB||)."""
import numpy as np, sys
from tladder import tblocks

def steps(S, legs, dirn):
    sites, W, V, N = tblocks(S, legs)
    s = S/2
    R = [W[np.ix_(V[k+1], V[k])] for k in range(S)]
    ell = lambda k: 0.5*np.sqrt((s-(k-s))*(s+(k-s)+1))
    out = []
    if dirn == 'down':
        rows = list(range(S, 0, -2))
        for r in rows:
            Rs = R[r-1]; Ro = R[r].T if r < S else None
            out.append((r, Rs, Ro, (ell(r)/ell(r-1)) if r < S else None))
    else:
        rows = list(range(0, S, 2))
        for r in rows:
            Rs = R[r].T; Ro = R[r-1] if r > 0 else None
            out.append((r, Rs, Ro, (ell(r-1)/ell(r)) if r > 0 else None))
    return out, V

def opt_Y(Rs, Ro):
    Rp = np.linalg.pinv(Rs)
    U, sv, Vt = np.linalg.svd(Rs)          # Rs: m x n, m >= n
    rank = int((sv > 1e-10*sv.max()).sum())
    PK = U[:, rank:] @ U[:, rank:].T
    if Ro is None or PK.shape[0] == 0 or np.abs(PK).max() < 1e-14:
        return Rp
    A = Rp @ Ro; B = PK @ Ro
    Z = -A @ np.linalg.pinv(B, rcond=1e-10)
    return Rp + Z @ PK

if __name__ == '__main__':
    S = int(sys.argv[1]); legs = tuple(int(x) for x in sys.argv[2].split(',')); dirn = sys.argv[3]
    st, V = steps(S, legs, dirn)
    for r, Rs, Ro, jx in st:
        Y = opt_Y(Rs, Ro)
        assert np.allclose(Y @ Rs, np.eye(Rs.shape[1]), atol=1e-8)
        if Ro is None:
            print(f"row {r}: start  ||Y||={np.linalg.norm(Y,2):.4f}")
            continue
        n_pinv = np.linalg.norm(np.linalg.pinv(Rs) @ Ro, 2)
        n_opt = np.linalg.norm(Y @ Ro, 2)
        print(f"row {r}: J_x ratio {jx:.4f}  pinv step/Jx {n_pinv/jx:.4f}   optimal step/Jx {n_opt/jx:.4f}   ||Y|| {np.linalg.norm(Y,2):.4f} vs ||Rs^+|| {np.linalg.norm(np.linalg.pinv(Rs),2):.4f}")

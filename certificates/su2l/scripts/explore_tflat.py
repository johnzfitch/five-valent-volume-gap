import numpy as np, sys
from tladder import tblocks, flat_grams
S = int(sys.argv[1]); legs = tuple(int(x) for x in sys.argv[2].split(','))
sites, W, V, N = tblocks(S, legs)
Gp, Gm = flat_grams(S, V, N)
ev = np.abs(np.linalg.eigvalsh(W)); nz = ev[ev > 1e-9*ev.max()]
print("dims", [len(v) for v in V], " q =", 8*nz.min(), " kernel", len(ev)-len(nz))
for k in range(S + 1):
    L = Gp[k] - Gm[k]
    off = np.abs(L - np.diag(np.diag(L))).max()
    # relative defect against the larger of the two Grams' scale: max_i |L_ii| / G_ii
    gd = np.maximum(np.diag(Gp[k]), np.diag(Gm[k]))
    rel = np.abs(np.diag(L)) / gd
    sv = lambda G: np.sqrt(np.clip(np.linalg.eigvalsh(G), 0, None))
    print(f"k={k} |V|={len(V[k])} offdiag={off:.1e} rel defect max {rel.max():.3f} median {np.median(rel):.3f}; "
          f"sqrt eig Gp min {sv(Gp[k]).min():.3f} Gm min {sv(Gm[k]).min():.3f}")
for k in range(S):
    svn = np.linalg.svd(N[k], compute_uv=False)
    print(f"N_{k}: shape {N[k].shape} smin {svn.min():.4f}  4*smin {4*svn.min():.3f}")

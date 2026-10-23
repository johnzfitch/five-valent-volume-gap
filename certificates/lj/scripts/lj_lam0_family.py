"""lam(W0) (best of three pairings) and q along families (s; n-1/2, n, n, n) for growing n.
Sparse shift-invert on the Gram matrix of the smaller side; smallest nonzero singular value."""
import sys, os, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lj_core as L


def sparse_block(o, which):
    col = o['colour']; ev = np.where(col == 0)[0]; od = np.where(col == 1)[0]
    pos = np.empty(o['nS'], dtype=np.int64); pos[ev] = np.arange(len(ev)); pos[od] = np.arange(len(od))
    lo, up = o['lo'], o['up']
    r = np.where(col[lo] == 0, pos[lo], pos[up]); cc = np.where(col[lo] == 0, pos[up], pos[lo])
    return sp.csr_matrix((o[which], (r, cc)), shape=(len(ev), len(od)))


def smallest_nz(M):
    nr, nc = M.shape
    if max(nr, nc) <= 1500:
        return L.nzmin_sv(M.toarray())[0]
    A = ((M.T @ M) if nc <= nr else (M @ M.T)).tocsc()
    smax = spl.eigsh(A, k=1, which='LA', return_eigenvectors=False)[0]
    k = 16
    while True:
        vals = np.sort(np.abs(spl.eigsh(A, k=min(k, A.shape[0] - 2), sigma=-1e-7 * smax, which='LM',
                                        return_eigenvectors=False, tol=1e-12)))
        nz = vals[vals > 1e-11 * smax]
        if nz.size or k >= A.shape[0] - 2:
            return np.sqrt(nz[0])
        k *= 3


def run(tj):
    best = 0; q = None; bnd = 0
    for p in L.pairings(tj):
        o = L.build(p)
        if q is None:
            q = 8 * smallest_nz(sparse_block(o, 'w'))
        l0 = smallest_nz(sparse_block(o, 'w0'))
        col, sig2 = o['colour'], o['sig2']
        mu = sig2[col == 0].min() * sig2[col == 1].min()
        best = max(best, l0); bnd = max(bnd, 2 * np.sqrt(mu) * l0)
    return q, best, bnd


if __name__ == '__main__':
    for s2 in map(int, sys.argv[1].split(',')):
        for n2 in map(int, sys.argv[2].split(',')):
            tj = (s2, n2 - 1, n2, n2, n2)
            if sum(tj) % 2: tj = (s2, n2, n2, n2, n2)
            q, l0, bnd = run(tj)
            print(f"tj={tj}  q={q:9.3f}  best lam0={l0:.4f}  best 2sqrt(mu)lam0={bnd:9.3f}  q/(s+1)={q/(s2/2+1):.3f}", flush=True)

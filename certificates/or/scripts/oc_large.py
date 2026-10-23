"""Large-spin study of the open regions (float64; dense LAPACK when small, sparse shift-invert otherwise).

For each tuple: q = 8 * smallest nonzero |eig T| (pairing P1, the operator is S4-invariant), bG (phi-only bound,
best of three pairings) where affordable, and whether the tuple is open.
Families:  A_s: (2s; a, a+1, a+1, a+1)  (the per-level minimisers are A_{3/2});  B: (2s ~ 0.4*2j; j,j,j,j+1/2)
type fat family;  plus random open tuples with 2j_max <= Rmax.
Usage: python oc_large.py out.txt [Amax] [Bmax] [nrand] [Rmax]
"""
import sys, os, math, time, random
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_fast import region, is_open, pairings, build, smin_nz

np.seterr(divide='ignore', invalid='ignore')


def sparse_T(tj):
    s2, a, b_, c_, d_ = tj
    s = s2 / 2
    n1, m1 = (a + b_) / 2 + 1, abs(a - b_) / 2
    n2, m2 = (c_ + d_) / 2 + 1, abs(c_ - d_) / 2
    Bd, Cd = region(tj)
    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(Bd, Cd))}
    I, J, V = [], [], []
    for (x, y), i in idx.items():
        b, c = x / 2, y / 2; u, d = b + c, b - c
        G = (u + s + 1) * (u - s)
        j = idx.get((x - 2, y))
        if j is not None:
            h = (n1 ** 2 - b ** 2) * (b ** 2 - m1 ** 2) / (4 * b ** 2 - 1)
            w = math.sqrt(h * ((s + .5) ** 2 - (d - .5) ** 2) * G / 16)
            I += [i, j]; J += [j, i]; V += [w, w]
        j = idx.get((x, y - 2))
        if j is not None:
            h = (n2 ** 2 - c ** 2) * (c ** 2 - m2 ** 2) / (4 * c ** 2 - 1)
            w = math.sqrt(h * ((s + .5) ** 2 - (d + .5) ** 2) * G / 16)
            I += [i, j]; J += [j, i]; V += [w, w]
    n = len(idx)
    return sp.csr_matrix((V, (I, J)), shape=(n, n))


def qmin(tj):
    W = sparse_T(tj)
    n = W.shape[0]
    if n <= 3000:
        ev = np.linalg.eigvalsh(W.toarray())
        a = np.abs(ev); thr = 1e-10 * a.max() * n
        nz = a[a > thr]
        return 8 * nz.min(), n, 'dense'
    # shift-invert near zero (offset avoids the kernel); k eigenvalues nearest sigma0
    sigma0 = 1e-3
    amax = abs(spl.eigsh(W, k=1, which='LM', return_eigenvectors=False)[0])
    k = 24
    while True:
        vals = spl.eigsh(W, k=min(k, n - 2), sigma=sigma0, which='LM', return_eigenvectors=False, tol=1e-12)
        a = np.abs(vals)
        nz = a[a > 1e-9 * amax]
        if nz.size >= 2 or k >= n - 2:       # need the +-pair of the smallest nonzero eigenvalue
            return 8 * nz.min(), n, f'sparse(k={k},ker>={int((a <= 1e-9*amax).sum())})'
        k *= 3


def bG_best(tj):
    best = 0.0
    for P in pairings(tj):
        o = build(P, ('KG',))
        l, _ = smin_nz(o['KG'])
        if l:
            best = max(best, 2 * math.sqrt(o['Phi']) * l)
    return best


if __name__ == '__main__':
    out = sys.argv[1]
    Amax = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    Bmax = int(sys.argv[3]) if len(sys.argv) > 3 else 160
    nrand = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    Rmax = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    L = []; t0 = time.time()
    L.append("Family A_s = (2s; a, a+1, a+1, a+1)   [columns: 2jmax, q, dim, method, q/2jmax^(1/2), bG (if dim<=1500)]")
    for s2 in (3, 5, 7):
        for a in [8, 16, 32, 64, 128, 256, 512, Amax]:
            tj = (s2, a, a + 1, a + 1, a + 1)
            if sum(tj) % 2:
                tj = (s2, a, a + 1, a + 1, a + 2)
            if not is_open(tj):
                L.append(f"  {tj} not open"); continue
            q, n, meth = qmin(tj)
            bg = bG_best(tj) if n <= 1500 else float('nan')
            L.append(f"  {tj}: 2jmax={max(tj)} q={q:.4f} dim={n} {meth} q/sqrt(2jmax)={q/math.sqrt(max(tj)):.4f} bG={bg:.4f}")
    L.append("Family B = (2s; 2j, 2j, 2j, 2j+1) with 2s = odd nearest 0.8*2j  (fat strips)")
    for j2 in [8, 16, 32, 48, 64, 96, 128, Bmax]:
        s2 = int(round(0.8 * j2)) | 1
        tj = (s2, j2, j2, j2, j2 + 1)
        if sum(tj) % 2:
            tj = (s2, j2, j2, j2 + 1, j2 + 2)
        if not is_open(tj):
            L.append(f"  {tj} not open"); continue
        q, n, meth = qmin(tj)
        L.append(f"  {tj}: q={q:.4f} dim={n} {meth} q/2s={q/s2:.4f}")
    rng = random.Random(20261021)
    L.append(f"Random open tuples, 2j_i <= {Rmax} (seed 20261021): ")
    got = 0; mins = (9e9, None); tries = 0
    while got < nrand and tries < 50 * nrand:
        tries += 1
        s2 = rng.randrange(3, Rmax + 1, 2)
        tj = (s2,) + tuple(sorted(rng.randrange(1, Rmax + 1) for _ in range(4)))
        if sum(tj) % 2 or len(region(tj)[0]) < 2 or not is_open(tj):
            continue
        q, n, meth = qmin(tj)
        got += 1
        if q < mins[0]:
            mins = (q, tj, n)
    L.append(f"  {got} random open tuples evaluated; smallest q = {mins[0]:.4f} at {mins[1]} (dim {mins[2]})")
    L.append(f"total time {time.time()-t0:.1f}s")
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

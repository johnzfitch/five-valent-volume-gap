"""W2 setting. Doubled spins tj = (s, j2, j3, j4, j5); pairing (23|45).
Rectangle: b in [|j2-j3|, j2+j3], c in [|j4-j5|, j4+j5] (step 1 in undoubled units); K = H(n1,m1) (+) H(n2,m2).
Region R: |b-c| <= s <= b+c (and b+c+s integer). Colour = parity of u=b+c (integer s).
Closed class C: every rectangle-neighbour of every C-site in R lies in R.  Block = K restricted to l2(C cap R)."""
import numpy as np
from stark import stark
def setup(tj):
    s2, a2, b2, c2, d2 = tj
    m1, n1 = abs(a2 - b2), a2 + b2 + 2      # doubled m, n
    m2, n2 = abs(c2 - d2), c2 + d2 + 2
    N1, N2 = (n1 - m1) // 2, (n2 - m2) // 2
    H1, _ = stark(N1, m1 / 2); H2, _ = stark(N2, m2 / 2)
    s = s2 / 2
    def inR(i, k):
        b = m1 / 2 + i; c = m2 / 2 + k
        return 0 <= i < N1 and 0 <= k < N2 and abs(b - c) <= s + 1e-9 and s <= b + c + 1e-9
    def colour(i, k): return int(round(m1 / 2 + i + m2 / 2 + k)) % 2
    return dict(s=s, N1=N1, N2=N2, m1=m1/2, m2=m2/2, H1=H1, H2=H2, inR=inR, colour=colour)
def closed_blocks(tj):
    """returns list of (colour, block matrix B (rows: rectangle neighbours, cols: C cap R sites), sites) for closed classes."""
    D = setup(tj); N1, N2 = D['N1'], D['N2']; out = []
    if (tj[0] % 2): return out          # integer s only
    for col in (0, 1):
        C = [(i, k) for i in range(N1) for k in range(N2) if D['inR'](i, k) and D['colour'](i, k) == col]
        if not C: continue
        ok = all(D['inR'](i2, k2) for (i, k) in C for (i2, k2) in ((i+1, k), (i-1, k), (i, k+1), (i, k-1))
                 if 0 <= i2 < N1 and 0 <= k2 < N2)
        if not ok: continue
        rows = sorted({(i2, k2) for (i, k) in C for (i2, k2) in ((i+1, k), (i-1, k), (i, k+1), (i, k-1)) if 0 <= i2 < N1 and 0 <= k2 < N2})
        ri = {p: r for r, p in enumerate(rows)}
        B = np.zeros((len(rows), len(C)))
        for c_, (i, k) in enumerate(C):
            if i + 1 < N1: B[ri[(i+1, k)], c_] = D['H1'][i+1, i]
            if i - 1 >= 0: B[ri[(i-1, k)], c_] = D['H1'][i-1, i]
            if k + 1 < N2: B[ri[(i, k+1)], c_] = D['H2'][k+1, k]
            if k - 1 >= 0: B[ri[(i, k-1)], c_] = D['H2'][k-1, k]
        out.append((col, B, C, D))
    return out
def nzmin(B, tol=1e-9):
    sv = np.linalg.svd(B, compute_uv=False)
    nz = sv[sv > tol * max(1, sv.max())]
    return (nz.min() if nz.size else None), B.shape[1] - nz.size

"""Diagnostics for one ordered pairing: chain links, Lambda, pivots of the column-Gram matrices.
Usage: python z32_diag.py 3 a b c d [tau]
"""
import sys, os
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32


def gram_tridiag(rows, cols, L, weight=1):
    """Exact column Gram (sum over rows of weight*link products): returns diag list, offdiag-squared list,
    and offdiag sign info is irrelevant (all positive in the nonnegative gauge)."""
    ci = {c: j for j, c in enumerate(cols)}
    K = len(cols)
    dg = [Fr(0)] * K
    for (r, c), w in L.items():
        if r in rows and c in ci:
            dg[ci[c]] += weight * w
    return dg


def pivots(diag, off2):
    p = []
    d = None
    for i, a in enumerate(diag):
        d = a if i == 0 else a - off2[i - 1] / d
        p.append(d)
    return p


if __name__ == '__main__':
    tj = tuple(int(x) for x in sys.argv[1:6])
    tau = Fr(sys.argv[6]) if len(sys.argv) > 6 else Fr(1, 4)
    z = Z32(tj)
    print('P (n1,m1,n2,m2) =', [str(x) for x in z.P], 'imb', z.imb)
    print('R0', [str(b) for b in z.R0]); print('C0', [str(b) for b in z.C0])
    print('R1', [str(b) for b in z.R1]); print('C1', [str(b) for b in z.C1])
    # zz0 chain listing
    for name, R, C, L in (('zz0', z.R0, z.C0, z.D0), ('zz1', z.R1, z.C1, z.D1)):
        print(name)
        for b in C:
            q = L.get((b, b)); p = L.get((b - 1, b))
            print(f'  col {str(b):>5}: q(row b)={float(q) if q else 0:12.3f}  p(row b-1)={float(p) if p else 0:12.3f}')
    # Lambda on zz0 cols (full rows)
    kap = Fr(4, 3)
    C0 = z.C0
    for b in C0:
        nn = sum(w for (r, c), w in z.N.items() if c == b)
        dd = sum(w for (r, c), w in z.D0.items() if c == b)
        lam = nn - kap * dd
        print(f'  Lambda[{b}] = {float(lam):12.3f}   rel {float(lam/dd) if dd else float("nan"):8.4f}')
    # pivots of M0 = D0^T D0 - tau N^T N on all zz0 cols (no deletion)
    D0, N, D1 = z.mats()
    for t in (0.0, float(tau)):
        M0 = D0.T @ D0 - t * (N.T @ N)
        # pivots
        K = M0.shape[0]; d = []
        for i in range(K):
            d.append(M0[i, i] if i == 0 else M0[i, i] - M0[i, i - 1] ** 2 / d[-1])
        print(f' tau={t}: pivots M0', np.round(d, 3), 'lmin', np.linalg.eigvalsh(M0)[0] if K else None)

"""Thin strip s = 1: the closed class is the diagonal b = c (u >= 1). In the gauge phi_b = (-1)^b psi_b,
||K psi||^2 = sum_l [A_l (phi_l - phi_{l-1})^2 + M_l (phi_l + phi_{l-1})^2] + E_lo phi_lo^2 + E_hi phi_hi^2,
A = (h1+h2)^2/2, M = (h1-h2)^2/2 at link l, E_lo = h1(lo)^2+h2(lo)^2, E_hi = h1(hi+1)^2+h2(hi+1)^2.
Pairs given by (n_i, m_i) (undoubled, n = j+j'+1, m = |j-j'|); h_i(L)^2 = (n^2-L^2)(L^2-m^2)/(4L^2-1) on m < L < n, else 0."""
import numpy as np
def h2(n, m, L):
    if L <= m or L >= n: return 0.0
    return (n*n - L*L) * (L*L - m*m) / (4*L*L - 1)
def Qmat(n1, m1, n2, m2):
    lo = max(m1, m2); hi = min(n1, n2) - 1
    if lo == 0: lo = 1
    bs = np.arange(lo, hi + 1e-9, 1.0); n = len(bs)
    Q = np.zeros((n, n))
    for i in range(1, n):
        L = bs[i]; a, b = np.sqrt(h2(n1, m1, L)), np.sqrt(h2(n2, m2, L))
        A = (a + b)**2 / 2; M = (a - b)**2 / 2
        Q[i, i] += A + M; Q[i-1, i-1] += A + M; Q[i, i-1] += M - A; Q[i-1, i] += M - A
    Q[0, 0] += h2(n1, m1, bs[0]) + h2(n2, m2, bs[0])
    Q[-1, -1] += h2(n1, m1, bs[-1] + 1) + h2(n2, m2, bs[-1] + 1)
    return Q, bs
def lam(n1, m1, n2, m2):
    Q, bs = Qmat(n1, m1, n2, m2); w = np.linalg.eigvalsh(Q)
    nz = w[w > 1e-9 * max(1, w.max())]
    return nz.min(), len(w) - len(nz)
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from w2core import closed_blocks, nzmin
    # cross-check against the 2D block for all s=1 closed even-sum classes, 2j <= 14
    import itertools
    mx = 0; cnt = 0
    for tj in itertools.product(range(1, 15), repeat=4):
        a, b, c, d = tj; tjf = (2,) + tj
        if sum(tjf) % 2 or a > b or c > d: continue
        if (min(a, b) + min(c, d) + 2) % 2 or a == 1 or c == 1: continue
        for col, B, C, D in closed_blocks(tjf):
            v, k = nzmin(B)
            if v is None: continue
            l, k2 = lam((a+b)/2 + 1, abs(a-b)/2, (c+d)/2 + 1, abs(c-d)/2)
            mx = max(mx, abs(np.sqrt(l) - v)); cnt += 1
    print('s=1 formula vs 2D block: classes', cnt, 'max |sqrt(lam) - sigma|', mx)

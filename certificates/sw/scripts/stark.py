import numpy as np
def stark(N, m):
    """Stark matrix H(n=m+N, m) on L=m..n-1 (N sites); links g[k] between sites k-1,k (k=1..N-1)."""
    n = m + N
    g = np.zeros(N)
    for k in range(1, N):
        L = m + k
        g[k] = np.sqrt((n*n - L*L)*(L*L - m*m)/(4*L*L - 1))
    H = np.diag(g[1:], 1) + np.diag(g[1:], -1)
    return H, g
def Bwin(N, m, a, b=None):
    H, _ = stark(N, m)
    if b is None: b = N - 2
    A = H - np.eye(N)
    return A[a+1:b+2, a:b+1]
def smin(M): return np.linalg.svd(M, compute_uv=False).min()

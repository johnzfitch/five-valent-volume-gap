import numpy as np
from stark import stark, Bwin, smin
def V_seq(N, m, s):
    """V_k (2x2 forms in (psi_{k-1}, psi_k)) for k = N-2 .. 1; V_{N-1} scalar. returns dict k -> matrix."""
    H, g = stark(N, m)
    gg = np.append(g, 0.0)
    V = {}
    V[N-1] = np.array([[gg[N-1]**2]])
    if N >= 3:
        V[N-2] = np.array([[gg[N-2]**2, -gg[N-2]], [-gg[N-2], 1 + gg[N-1]**2]])
    for k in range(N-3, 0, -1):
        P = V[k+1]; p, q, r = P[0,0], P[0,1], P[1,1]
        kap = gg[k+1]**2 - s + r
        # V_k(x,y) = (g_k x - y)^2 + p y^2 - [g_{k+1}(g_k x - y) + q y]^2/kap
        a = np.array([gg[k], -1.0]); b = gg[k+1]*np.array([gg[k], -1.0]) + np.array([0, q])
        V[k] = np.outer(a, a) + np.diag([0, p]) - np.outer(b, b)/kap
    return V
if __name__ == '__main__':
    s = (15 - np.sqrt(97))/6
    for N, m in [(3,0),(5,0),(9,0),(15,0),(15,2),(16,0),(31,0)]:
        V = V_seq(N, m, s)
        mins = {k: np.linalg.eigvalsh(V[k]).min() - s for k in V}
        chk = [smin(Bwin(N, m, k-1))**2 - s for k in V]
        print(N, m, 'min_k lam(V_k - s):', round(min(mins.values()), 5), ' consistency', all((x >= -1e-9) == (y >= -1e-9) for x, y in zip(mins.values(), chk)))
        if N in (15, 16):
            for k in sorted(V)[:6]: print('   k', k, np.round(V[k], 4))

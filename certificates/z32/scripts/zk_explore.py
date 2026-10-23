"""Exploration for general half-integer j1 = s (K = s + 1/2 zigzags).

B (rows: d+s even, cols: d+s odd) is block lower-bidiagonal: diagonal blocks D_k (zigzag paths),
subdiagonal blocks N_k (rows of zigzag k, columns of zigzag k-1).  If every imbalance is in {0,1}
('all tall'), B is injective and lam(T) = sigma_min(B).  The chained Young inequality gives, for any
tau_1..tau_{K-1} > 0 (t_k = 1/(1+tau_k)):
  sigma_min(B)^2 >= min_k lam_min( (1-t_k) D_k^T D_k - tau_{k+1} N_{k+1}^T N_{k+1} )   (t_0 := 0, tau_K := 0).
This script (a) counts tuples having an all-tall ordered pairing, (b) evaluates the bound (coordinate search
over tau) and the true q.
Usage: python zk_explore.py 2s M
"""
import sys, os, itertools, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def region(tj):
    s2, a, b_, c_, d_ = tj
    out = []
    for B in range(abs(a - b_), a + b_ + 1, 2):
        for C in range(abs(c_ - d_), c_ + d_ + 1, 2):
            if abs(B - C) <= s2 <= B + C and (B + C + s2) % 2 == 0:
                out.append((B, C))
    return out


def blocks(tj):
    s2, a, b_, c_, d_ = tj
    s = s2 / 2
    n1, m1 = (a + b_) / 2 + 1, abs(a - b_) / 2
    n2, m2 = (c_ + d_) / 2 + 1, abs(c_ - d_) / 2
    S = region(tj)
    K = s2 // 2 + 1
    rows = [p for p in S if ((p[0] - p[1] + s2) // 2) % 2 == 0]
    cols = [p for p in S if ((p[0] - p[1] + s2) // 2) % 2 == 1]
    rk = {p: (p[0] - p[1] + s2) // 4 for p in rows}
    ck = {p: (p[0] - p[1] + s2 - 2) // 4 for p in cols}

    def w(up, isX):
        b, c = up[0] / 2, up[1] / 2
        u, d = b + c, b - c
        if isX:
            h2 = (n1 ** 2 - b ** 2) * (b ** 2 - m1 ** 2) / (4 * b ** 2 - 1); db = d - 0.5
        else:
            h2 = (n2 ** 2 - c ** 2) * (c ** 2 - m2 ** 2) / (4 * c ** 2 - 1); db = d + 0.5
        F = (s + 0.5) ** 2 - db ** 2; G = (u + s + 1) * (u - s)
        return math.sqrt(max(h2 * F * G / 16, 0.0))
    R = [[p for p in rows if rk[p] == k] for k in range(K)]
    C = [[p for p in cols if ck[p] == k] for k in range(K)]
    ri = [{p: i for i, p in enumerate(R[k])} for k in range(K)]
    ci = [{p: i for i, p in enumerate(C[k])} for k in range(K)]
    D = [np.zeros((len(R[k]), len(C[k]))) for k in range(K)]
    N = [None] + [np.zeros((len(R[k]), len(C[k - 1]))) for k in range(1, K)]
    for p in rows:
        k = rk[p]
        for q, isX, up in (((p[0] + 2, p[1]), True, (p[0] + 2, p[1])), ((p[0] - 2, p[1]), True, p),
                           ((p[0], p[1] + 2), False, (p[0], p[1] + 2)), ((p[0], p[1] - 2), False, p)):
            if q in ck:
                l = ck[q]
                if l == k:
                    D[k][ri[k][p], ci[k][q]] = w(up, isX)
                elif l == k - 1:
                    N[k][ri[k][p], ci[l][q]] = w(up, isX)
                else:
                    raise RuntimeError('not block bidiagonal')
    imb = [len(R[k]) - len(C[k]) for k in range(K)]
    return D, N, imb


def pairings6(tj):
    s, a, b, c, d = tj
    return [(s, a, b, c, d), (s, c, d, a, b), (s, a, c, b, d), (s, b, d, a, c), (s, a, d, b, c), (s, b, c, a, d)]


def lmin(M):
    return np.linalg.eigvalsh(M)[0] if M.size else math.inf


def chain_bound(D, N, taus):
    K = len(D)
    vals = []
    for k in range(K):
        tk = 1.0 if k == 0 else 1 / (1 + taus[k - 1])     # coefficient (1 - t_k) with t_0 = 0 -> 1
        coef = 1.0 if k == 0 else (1 - tk)
        M = coef * (D[k].T @ D[k])
        if k + 1 < K and N[k + 1].size:
            M = M - taus[k] * (N[k + 1].T @ N[k + 1])
        if M.size:
            vals.append(lmin(M))
    return min(vals) if vals else math.inf


def optimise(D, N, iters=4):
    """multi-start coordinate search over tau (bound is not concave in tau; starts from small values)."""
    K = len(D)
    grid = [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]
    best_all = (-math.inf, None)
    for start in (0.01, 0.05, 0.15, 0.3):
        taus = [start] * (K - 1)
        best = chain_bound(D, N, taus)
        for _ in range(iters):
            for j in range(K - 1):
                for g in grid:
                    tt = list(taus); tt[j] = g
                    v = chain_bound(D, N, tt)
                    if v > best:
                        best, taus = v, tt
        if best > best_all[0]:
            best_all = (best, taus)
    return best_all


if __name__ == '__main__':
    s2, M = int(sys.argv[1]), int(sys.argv[2])
    ntot = nall = 0; res = []
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    tj = (s2, a, b, c, d)
                    if sum(tj) % 2 or len(region(tj)) < 2:
                        continue
                    ntot += 1
                    cand = []
                    for P in pairings6(tj):
                        D, N, imb = blocks(P)
                        if min(imb) >= 0:
                            cand.append((P, D, N, imb))
                    if not cand:
                        continue
                    nall += 1
                    best = (-math.inf, None, None, None)
                    for P, D, N, imb in cand:
                        v, taus = optimise(D, N)
                        if v > best[0]:
                            best = (v, P, imb, taus)
                    P, D, N, imb = cand[0]
                    Bm = np.block([[D[k] if l == k else (N[k] if l == k - 1 else np.zeros((D[k].shape[0], D[l].shape[1])))
                                    for l in range(len(D))] for k in range(len(D))])
                    q = 8 * np.linalg.svd(Bm, compute_uv=False).min()
                    res.append((8 * math.sqrt(max(best[0], 0)), q, tj, best[1], best[2]))
    res.sort()
    print(f'2s={s2}, 2j<={M}: {ntot} tuples, {nall} with an all-tall ordered pairing')
    for r in res[:12]:
        print(f'  bound {r[0]:.4f}  q {r[1]:.4f}  {r[2]}  via {r[3]} imb {r[4]}')
    print('violations bound>q:', sum(1 for r in res if r[0] > r[1] * (1 + 1e-9)))

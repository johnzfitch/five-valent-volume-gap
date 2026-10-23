"""Diagonal (d-direction) block structure of W0 for one pairing.

Doubled-spin inputs: S = 2 j1, legs (a2, a3 | a4, a5) doubled; pairs (a2,a3), (a4,a5).
Sites of R are grouped by diagonal d = b - c (doubled: dd = b2 - c2, from -S to S in steps of 2).
A = raising part of W0 (d -> d+1): X-link (b-1,c)->(b,c) and D-link reversed (b,c)->(b,c-1).
Blocks M[k] : V_k -> V_{k+1}, k = 0..2s-1 (k = (dd + S)/2).

Exact objects (float64 here; exact rationals in check_ladder_exact.py):
  Gp[k] = M[k]^T M[k]        (up-Gram on V_k; zero matrix on the top wall)
  Gm[k] = M[k-1] M[k-1]^T    (down-Gram on V_k; zero on the bottom wall)
  Cp[k] = Gp[k] + d(d+1)/4,  Cm[k] = Gm[k] + d(d-1)/4   (d = k - s)
  M[k] Cp[k] = Cm[k+1] M[k]  holds identically (pure algebra).
"""
import numpy as np
from w0sparse import build, colour


def blocks(S, legs):
    sites, W = build(S, *legs)
    W = W.toarray()
    by = {}
    for i, (b, c) in enumerate(sites):
        k = (b - c + S) // 2
        by.setdefault(k, []).append(i)
    K = S + 1
    V = [sorted(by.get(k, []), key=lambda i: sites[i][0] + sites[i][1]) for k in range(K)]
    M = []
    for k in range(K - 1):
        M.append(W[np.ix_(V[k + 1], V[k])])     # rows: V_{k+1}, cols: V_k
    return sites, W, V, M


def grams(S, V, M):
    K = S + 1
    s = S / 2
    Gp, Gm, Cp, Cm = [], [], [], []
    for k in range(K):
        n = len(V[k])
        gp = M[k].T @ M[k] if k < K - 1 else np.zeros((n, n))
        gm = M[k - 1] @ M[k - 1].T if k > 0 else np.zeros((n, n))
        d = k - s
        Gp.append(gp); Gm.append(gm)
        Cp.append(gp + d * (d + 1) / 4 * np.eye(n)); Cm.append(gm + d * (d - 1) / 4 * np.eye(n))
    return Gp, Gm, Cp, Cm


def lam_min_nonzero(S, legs):
    sites, W = build(S, *legs)
    W = W.toarray()
    ev = np.abs(np.linalg.eigvalsh(W))
    nz = ev[ev > 1e-9 * max(1, ev.max())]
    return nz.min()

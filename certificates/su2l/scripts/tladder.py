"""T-picture ladder: T's d-raising blocks R_d = l_s(d) N_d, l_s(d) = 1/2 sqrt((s-d)(s+d+1)).
Flat-ladder defect Lt_d = N_d^T N_d - N_{d-1} N_{d-1}^T (diagonal by paper Lemma E.1 / Prop 6.13)."""
import numpy as np
from t1core import Tmatrix, colour

def tblocks(S, legs):
    sites, W = Tmatrix(S, *legs)
    s = S / 2
    by = {}
    for i, (b, c) in enumerate(sites):
        by.setdefault((b - c + S) // 2, []).append(i)
    V = [sorted(by.get(k, []), key=lambda i: sites[i][0] + sites[i][1]) for k in range(S + 1)]
    N = []
    for k in range(S):
        d = k - s
        l = 0.5 * np.sqrt((s - d) * (s + d + 1))
        N.append(W[np.ix_(V[k + 1], V[k])] / l)
    return sites, W, V, N

def flat_grams(S, V, N):
    Gp, Gm = [], []
    for k in range(S + 1):
        n = len(V[k])
        Gp.append(N[k].T @ N[k] if k < S else np.zeros((n, n)))
        Gm.append(N[k - 1] @ N[k - 1].T if k > 0 else np.zeros((n, n)))
    return Gp, Gm

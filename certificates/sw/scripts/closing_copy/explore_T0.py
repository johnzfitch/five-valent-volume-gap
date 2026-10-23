"""Exploration: Heron multiplier as site scaling; gap of the 'Stark compression' T0.
T links: x = h23(b) * sqrt(P)/4, P = F(dbar) G(u_up), F(t) = (s+1/2)^2 - t^2, G(u) = (u+s+1)(u-s).
T0 links: h23(b), h45(c) (hydrogen Stark weights), on the same region R.
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import sites, grid, bip


def stark2(n, m, l):  # undoubled n, m, l (floats)
    return (n * n - l * l) * (l * l - m * m) / (4 * l * l - 1)


def T0(tj):
    s, j2, j3, j4, j5 = [t / 2 for t in tj]
    S = sites(tj)
    idx = {p: i for i, p in enumerate(S)}
    W = np.zeros((len(S), len(S)))
    for (b, c), i in idx.items():
        k = idx.get((b + 2, c))
        if k is not None:
            W[i, k] = W[k, i] = np.sqrt(stark2(j2 + j3 + 1, abs(j2 - j3), (b + 2) / 2))
        k = idx.get((b, c + 2))
        if k is not None:
            W[i, k] = W[k, i] = np.sqrt(stark2(j4 + j5 + 1, abs(j4 - j5), (c + 2) / 2))
    return S, W


def scale(tj):
    """sigma^2 = phi(d) omega(u) with phi(d)phi(d+1)=F(d+1/2), omega(u)omega(u+1)=G(u+1); returns dict."""
    s = tj[0] / 2
    S = sites(tj)
    ds = sorted({(b - c) / 2 for b, c in S}); us = sorted({(b + c) / 2 for b, c in S})
    phi = {ds[0]: 1.0}
    for d in ds[1:]:
        phi[d] = ((s + .5) ** 2 - (d - .5) ** 2) / phi[d - 1]
    om = {us[0]: 1.0}
    for u in us[1:]:
        om[u] = (u + s + 1) * (u - s) / om[u - 1]
    return {(b, c): phi[(b - c) / 2] * om[(b + c) / 2] for b, c in S}


def nzmin(W, S, tol=1e-9):
    B, ev, od = bip(S, W)
    sv = np.linalg.svd(B, compute_uv=False)
    nz = sv[sv > tol * max(sv.max(), 1)]
    return nz.min() if nz.size else None, int(B.shape[0] + B.shape[1] - 2 * nz.size)


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    worst0 = (9e9, None); worstr = (9e9, None); maxerr = 0
    rows = []
    for tj in itertools.product(range(1, M + 1), repeat=5):
        if sum(tj) % 2 or tj[1] > tj[2] or tj[3] > tj[4]:
            continue
        S, W = grid(tj)
        if len(S) < 2:
            continue
        sig = scale(tj)
        # check scaling identity: W_link = 1/4 * h * sqrt(sig sig')
        S0, W0 = T0(tj)
        sq = np.sqrt(np.array([sig[p] for p in S]))
        Wrec = 0.25 * W0 * np.outer(sq, sq)
        maxerr = max(maxerr, np.abs(Wrec - W).max() / max(W.max(), 1))
        lt, k1 = nzmin(W, S); l0, k0 = nzmin(W0, S)
        if lt is None:
            continue
        ev = [sig[S[i]] for i in bip(S, W)[1]]; od = [sig[S[i]] for i in bip(S, W)[2]]
        bound = 0.25 * np.sqrt(min(ev) * min(od)) * l0 if od and ev else None
        if l0 < worst0[0]:
            worst0 = (l0, tj)
        r = lt / bound
        if r < worstr[0]:
            worstr = (r, tj)
        rows.append((tj, lt, l0, bound, k1, k0))
    print('max rel err of scaling identity', maxerr)
    print('min gap of T0', worst0)
    print('min ratio true/bound', worstr)
    rows.sort(key=lambda r: r[2])
    for r in rows[:25]:
        print(r)

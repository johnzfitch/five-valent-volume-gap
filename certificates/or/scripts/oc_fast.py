"""Fast float64 builders for the open-region study (no Fractions).  Spins doubled.

Builds, for pairing (23|45), the bipartite blocks of
  T   : links x = h1(b) sqrt(F G)/4 ...            (true operator, q = 8 * smallest nonzero sv)
  KG  : links h1(b) sqrt(G(u)), h2(c) sqrt(G(u))   (phi-only core:  T = 1/4 S_phi KG S_phi)
  K   : links h1(b), h2(c)                         (W0 of the closing note: T = 1/4 S W0 S)
and the site weights phi(d), omega(u) (free constants fixed at the first d / first u).
"""
import numpy as np
np.seterr(divide="ignore", invalid="ignore")


def region(tj):
    s2, a, b_, c_, d_ = tj
    B = np.arange(abs(a - b_), a + b_ + 1, 2)
    C = np.arange(abs(c_ - d_), c_ + d_ + 1, 2)
    BB, CC = np.meshgrid(B, C, indexing='ij')
    BB = BB.ravel(); CC = CC.ravel()
    ok = (np.abs(BB - CC) <= s2) & (s2 <= BB + CC) & ((BB + CC + s2) % 2 == 0)
    return BB[ok], CC[ok]


def build(tj, kinds=('T', 'KG', 'K')):
    """Return dict kind -> (Bmat even x odd), plus (phi, omega, colour) per site."""
    s2, a, b_, c_, d_ = tj
    s = s2 / 2
    n1, m1 = (a + b_) / 2 + 1, abs(a - b_) / 2
    n2, m2 = (c_ + d_) / 2 + 1, abs(c_ - d_) / 2
    Bd, Cd = region(tj)
    nS = len(Bd)
    if nS == 0:
        return None
    b = Bd / 2; c = Cd / 2; u = b + c; d = b - c
    col = ((Bd + Cd) // 2) % 2
    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(Bd, Cd))}
    ev = np.where(col == 0)[0]; od = np.where(col == 1)[0]
    pos = np.empty(nS, dtype=int); pos[ev] = np.arange(len(ev)); pos[od] = np.arange(len(od))
    # links: X with upper end (b,c) from (b-1,c); D with upper end (b,c) from (b,c-1)
    rows = []
    for i in range(nS):
        for kind, (dB, dC) in (('X', (2, 0)), ('D', (0, 2))):
            j = idx.get((int(Bd[i]) - dB, int(Cd[i]) - dC))
            if j is not None:
                rows.append((j, i, kind))
    lo = np.array([r[0] for r in rows], dtype=int); up = np.array([r[1] for r in rows], dtype=int)
    isX = np.array([r[2] == 'X' for r in rows])
    bu, cu, uu, du = b[up], c[up], u[up], d[up]
    hX2 = (n1 ** 2 - bu ** 2) * (bu ** 2 - m1 ** 2) / (4 * bu ** 2 - 1)
    hD2 = (n2 ** 2 - cu ** 2) * (cu ** 2 - m2 ** 2) / (4 * cu ** 2 - 1)
    h2 = np.where(isX, hX2, hD2)
    dbar = np.where(isX, du - 0.5, du + 0.5)
    F = (s + 0.5) ** 2 - dbar ** 2
    G = (uu + s + 1) * (uu - s)
    w2 = {'T': h2 * F * G / 16, 'KG': h2 * G, 'K': h2, 'KF': h2 * F}
    out = {}
    for k in kinds:
        M = np.zeros((len(ev), len(od)))
        w = np.sqrt(np.maximum(w2[k], 0))
        for l, q, x in zip(lo, up, w):
            if col[l] == 0:
                M[pos[l], pos[q]] = x
            else:
                M[pos[q], pos[l]] = x
        out[k] = M
    # site weights
    ds = np.unique(d); us = np.unique(u)
    phi = {ds[0]: 1.0}
    for dd in ds[1:]:
        phi[dd] = ((s + 0.5) ** 2 - (dd - 0.5) ** 2) / phi[dd - 1]
    om = {us[0]: 1.0}
    for uv in us[1:]:
        om[uv] = (uv + s + 1) * (uv - s) / om[uv - 1]
    P = np.array([phi[x] for x in d]); O = np.array([om[x] for x in u])
    out['Phi'] = P[ev].min() * P[od].min() if len(ev) and len(od) else np.nan
    out['mu'] = (P * O)[ev].min() * (P * O)[od].min() if len(ev) and len(od) else np.nan
    out['n'] = nS
    out['ev'] = len(ev); out['od'] = len(od)
    return out


def smin_nz(M, tol=1e-10):
    if M.size == 0:
        return None, 0
    sv = np.linalg.svd(M, compute_uv=False)
    thr = tol * max(sv.max(), 1.0) * sum(M.shape)
    nz = sv[sv > thr]
    return (float(nz.min()) if nz.size else None), int(min(M.shape) - nz.size)


def closed_any(tj):
    """True if some colour class of the (23|45) region is closed (all rectangle-neighbours in R)."""
    s2, a, b_, c_, d_ = tj
    Bd, Cd = region(tj)
    if len(Bd) == 0:
        return True
    S = set(zip(Bd.tolist(), Cd.tolist()))
    bmin, bmax = abs(a - b_), a + b_
    cmin, cmax = abs(c_ - d_), c_ + d_
    res = [True, True]
    for (x, y) in S:
        cl = ((x + y) // 2) % 2
        for (p, q) in ((x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)):
            if bmin <= p <= bmax and cmin <= q <= cmax and (p, q) not in S:
                res[cl] = False
    return res[0] or res[1]


def pairings(tj):
    s, a, b, c, d = tj
    return [(s, a, b, c, d), (s, a, c, b, d), (s, a, d, b, c)]


def is_open(tj):
    if tj[0] % 2 == 0 or tj[0] < 3:
        return False
    if len(region(tj)[0]) < 2:
        return False
    return not any(closed_any(P) for P in pairings(tj))

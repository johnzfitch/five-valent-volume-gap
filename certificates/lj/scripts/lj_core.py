"""Core tools for the large-j1 study.  Spins DOUBLED integers, tuple tj = (2s, 2j2, 2j3, 2j4, 2j5).
Pairing (23|45) unless the tuple is permuted by the caller.  Conventions follow
[DC] = research/al_dichotomy_closing_20261021/THEOREM.md (Theorem 3, Lemmas 4, 5):
  W   : nonnegative-gauge matrix of T (links x, d),  q = 8 * smallest nonzero |eig W|
  W0  : P_R K P_R, K = H(n1,m1) (+) H(n2,m2)  (Stark links h1(b), h2(c) restricted to R)
  sigma^2(b,c) = phi(d) omega(u),  W = (1/4) S W0 S.
Independent re-implementation (the W-link formulas are those of ls_core.py / [DC] Section 1)."""
import numpy as np


def region(tj):
    s2, a, b_, c_, d_ = tj
    B = np.arange(abs(a - b_), a + b_ + 1, 2)
    C = np.arange(abs(c_ - d_), c_ + d_ + 1, 2)
    BB, CC = np.meshgrid(B, C, indexing='ij')
    BB = BB.ravel(); CC = CC.ravel()
    inR = (np.abs(BB - CC) <= s2) & (s2 <= BB + CC) & ((BB + CC + s2) % 2 == 0)
    return BB, CC, inR          # whole rectangle (doubled) + mask of R


def build(tj):
    """Return dict with sites of R, link lists (lo, up), weights of W and of W0, colour, sigma^2."""
    s2, a, b_, c_, d_ = tj
    s = s2 / 2
    n1, m1 = (a + b_) / 2 + 1, abs(a - b_) / 2
    n2, m2 = (c_ + d_) / 2 + 1, abs(c_ - d_) / 2
    BB, CC, inR = region(tj)
    Bd, Cd = BB[inR], CC[inR]
    nS = len(Bd)
    out = dict(nS=nS, Bd=Bd, Cd=Cd, s=s, n1=n1, m1=m1, n2=n2, m2=m2)
    if nS == 0:
        return out
    b0, c0 = Bd.min(), Cd.min()
    grid = -np.ones(((Bd.max() - b0) // 2 + 2, (Cd.max() - c0) // 2 + 2), dtype=np.int64)
    ib, ic = (Bd - b0) // 2, (Cd - c0) // 2
    grid[ib, ic] = np.arange(nS)
    b = Bd / 2.0; c = Cd / 2.0; u = b + c; dd = b - c
    G = (u + s + 1) * (u - s)
    lo, up, w, w0 = [], [], [], []
    idx = np.arange(nS)
    # X links (b-1,c)-(b,c)
    ok = ib >= 1
    j = np.where(ok, grid[np.maximum(ib - 1, 0), ic], -1); ok &= j >= 0
    h2 = (n1 ** 2 - b ** 2) * (b ** 2 - m1 ** 2) / (4 * b ** 2 - 1)
    wx2 = h2 * ((s + .5) ** 2 - (dd - .5) ** 2) * G / 16
    lo.append(j[ok]); up.append(idx[ok]); w.append(np.sqrt(np.maximum(wx2[ok], 0))); w0.append(np.sqrt(np.maximum(h2[ok], 0)))
    ok = ic >= 1
    j = np.where(ok, grid[ib, np.maximum(ic - 1, 0)], -1); ok &= j >= 0
    h2 = (n2 ** 2 - c ** 2) * (c ** 2 - m2 ** 2) / (4 * c ** 2 - 1)
    wd2 = h2 * ((s + .5) ** 2 - (dd + .5) ** 2) * G / 16
    lo.append(j[ok]); up.append(idx[ok]); w.append(np.sqrt(np.maximum(wd2[ok], 0))); w0.append(np.sqrt(np.maximum(h2[ok], 0)))
    out.update(lo=np.concatenate(lo), up=np.concatenate(up), w=np.concatenate(w), w0=np.concatenate(w0))
    colour = np.rint(u - s).astype(int) % 2
    out['colour'] = colour
    # sigma^2 = phi(d) omega(u), recursions phi(d)phi(d+1)=F(d+1/2), omega(u-1)omega(u)=G(u)
    dvals = np.arange(dd.min(), dd.max() + 1)
    phi = np.empty(len(dvals)); phi[0] = 1.0
    for k in range(1, len(dvals)):
        phi[k] = ((s + .5) ** 2 - (dvals[k - 1] + .5) ** 2) / phi[k - 1]
    uvals = np.arange(u.min(), u.max() + 1)
    om = np.empty(len(uvals)); om[0] = 1.0
    for k in range(1, len(uvals)):
        om[k] = (uvals[k] + s + 1) * (uvals[k] - s) / om[k - 1]
    sig2 = phi[np.rint(dd - dvals[0]).astype(int)] * om[np.rint(u - uvals[0]).astype(int)]
    out['sig2'] = sig2
    # cut sites: in R with a rectangle neighbour outside R
    rect = set(zip(BB.tolist(), CC.tolist())); Rset = set(zip(Bd.tolist(), Cd.tolist()))
    cut = np.zeros(nS, bool)
    for i in range(nS):
        for (x, y) in ((Bd[i] + 2, Cd[i]), (Bd[i] - 2, Cd[i]), (Bd[i], Cd[i] + 2), (Bd[i], Cd[i] - 2)):
            if (x, y) in rect and (x, y) not in Rset:
                cut[i] = True; break
    out['cut'] = cut
    return out


def block(o, which='w'):
    """Dense bipartite block (colour 0 rows x colour 1 cols)."""
    col = o['colour']; ev = np.where(col == 0)[0]; od = np.where(col == 1)[0]
    pos = np.empty(o['nS'], dtype=np.int64); pos[ev] = np.arange(len(ev)); pos[od] = np.arange(len(od))
    lo, up = o['lo'], o['up']
    r = np.where(col[lo] == 0, pos[lo], pos[up]); cc = np.where(col[lo] == 0, pos[up], pos[lo])
    M = np.zeros((len(ev), len(od)))
    np.add.at(M, (r, cc), o[which])
    return M, ev, od


def nzmin_sv(M, rel=1e-9):
    if M.size == 0:
        return None, 0
    sv = np.linalg.svd(M, compute_uv=False)
    thr = rel * max(sv.max(), 1.0)
    nz = sv[sv > thr]
    return (nz.min() if nz.size else None), int(sv.size - nz.size)


def pairings(tj):
    s2, a, b, c, d = tj
    return [(s2, a, b, c, d), (s2, a, c, b, d), (s2, a, d, b, c)]


def analyse(tj):
    """q (true), and per pairing: lam(W0), mu, covered flag (closed class & (odd N-sum or spin-1/2 pair))."""
    res = []
    q = None
    for p in pairings(tj):
        o = build(p)
        if o['nS'] == 0:
            return None
        if q is None:
            Mw, _, _ = block(o, 'w'); l, _ = nzmin_sv(Mw); q = None if l is None else 8 * l
        M0, _, _ = block(o, 'w0'); l0, _ = nzmin_sv(M0)
        col, sig2 = o['colour'], o['sig2']
        mu = (sig2[col == 0].min() if (col == 0).any() else np.inf) * (sig2[col == 1].min() if (col == 1).any() else np.inf)
        closed = [not (o['cut'] & (col == k)).any() and (col == k).any() for k in (0, 1)]
        N1 = min(p[1], p[2]) + 1; N2 = min(p[3], p[4]) + 1
        half = 1 in (p[1], p[2], p[3], p[4])
        cov = any(closed) and ((N1 + N2) % 2 == 1 or half)
        res.append(dict(l0=l0, mu=mu, closed=closed, cov=cov, Nodd=(N1 + N2) % 2 == 1))
    return q, res

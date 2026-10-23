"""Exact (rational) certificate of the chained inequality (Proposition 11) for half-integer j1 with K zigzags,
on all tuples with 2j <= M that have an all-tall ordered pairing.

For an all-tall pairing, sigma_min(B)^2 >= min_k lam_min(M_k), with
  M_k = c_k D_k^T D_k - tau_{k+1} N_{k+1}^T N_{k+1},  c_0 = 1, c_k = tau_k/(1+tau_k), tau_K = 0.
Each M_k is tridiagonal in the zigzag-k column index with rational diagonal; its (c,c+1) entry is
  c_k sqrt(a) - tau_{k+1} sqrt(b),  a = product of the two D-links at the shared zigzag-k row (squared),
  b = product of the two N-links at the shared zigzag-(k+1) row (squared), and b = kappa_{k+1}^2 a exactly
  ([OR] Prop. 6; asserted), so the squared off-diagonal is rational.  PSD of M_k - s_k I is certified by exact
  LDL^T pivots for rational tau's and s_k.
Usage: python zk_exact_cert.py 2s M out.txt
"""
import sys, os, time, math, itertools
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zk_core import ZK, pairings6, tuplesK


def tridiag(z, k, ck, tnext):
    """exact (diag, offdiag^2) of M_k on zigzag-k columns."""
    cols = z.C[k]; idx = {c: i for i, c in enumerate(cols)}; Kc = len(cols)
    dg = [Fr(0)] * Kc
    for b, w in z.q[k].items():
        if b in idx: dg[idx[b]] += ck * w
    for b, w in z.p[k].items():
        if b + 1 in idx: dg[idx[b + 1]] += ck * w
    a = {}
    for b in z.R[k]:
        if b in z.q[k] and b in z.p[k] and b in idx and b + 1 in idx:
            a[idx[b]] = z.q[k][b] * z.p[k][b]          # row R^k_b joins columns b, b+1
    bb = {}
    if tnext and k + 1 < z.K:
        npk, nmk = z.np_[k + 1], z.nm[k + 1]
        for b, w in npk.items():
            if b in idx: dg[idx[b]] -= tnext * w
        for b, w in nmk.items():
            if b - 1 in idx: dg[idx[b - 1]] -= tnext * w
        for b in z.R[k + 1]:
            if b in npk and b in nmk and b in idx and b - 1 in idx:
                bb[idx[b - 1]] = npk[b] * nmk[b]       # row R^{k+1}_b joins columns b-1, b
        kap = z.kappa[k + 1]
    off2 = []
    for i in range(Kc - 1):
        A = a.get(i); B = bb.get(i)
        if B is None:
            off2.append(ck * ck * A if A is not None else Fr(0))
        elif A is None:
            off2.append(tnext * tnext * B)
        else:
            assert B == kap * kap * A, ('coupling identity failed', z.tj, k)
            off2.append(A * (ck - tnext * kap) ** 2)
    return dg, off2


def flmin(dg, off2):
    if not dg:
        return math.inf
    M = np.diag([float(x) for x in dg])
    for i in range(len(dg) - 1):
        v = math.sqrt(float(off2[i])); M[i, i + 1] = M[i + 1, i] = v
    return np.linalg.eigvalsh(M)[0]


def psd(dg, off2, s):
    d = None
    for i in range(len(dg)):
        d = dg[i] - s if i == 0 else dg[i] - s - off2[i - 1] / d
        if d < 0 or (d == 0 and i < len(dg) - 1):
            return False
    return True


def bound(z, taus):
    K = z.K; vals = []
    for k in range(K):
        ck = Fr(1) if k == 0 else taus[k - 1] / (1 + taus[k - 1])
        tn = taus[k] if k < K - 1 else Fr(0)
        dg, o2 = tridiag(z, k, ck, tn)
        vals.append((flmin(dg, o2), dg, o2))
    return vals


def search(z):
    """float search over the taus (multi-start coordinate search); exact certification follows."""
    K = z.K
    D, N = z.mats()
    G = [d.T @ d for d in D]
    NN = [None] + [n.T @ n for n in N[1:]]

    def fb(taus):
        vals = []
        for k in range(K):
            ck = 1.0 if k == 0 else taus[k - 1] / (1 + taus[k - 1])
            Mk = ck * G[k]
            if k + 1 < K and NN[k + 1].size and Mk.size:
                Mk = Mk - taus[k] * NN[k + 1]
            if Mk.size:
                vals.append(np.linalg.eigvalsh(Mk)[0])
        return min(vals) if vals else math.inf
    grid = [1, 2, 4, 7, 10, 15, 20, 30, 40, 60, 80, 100, 120, 160, 200, 300]
    best = (-math.inf, None)
    for start in (2, 10, 30, 60):
        taus = [start] * (K - 1)
        cur = fb([t / 200 for t in taus])
        for _ in range(4):
            for j in range(K - 1):
                for g in grid:
                    tt = list(taus); tt[j] = g
                    v = fb([t / 200 for t in tt])
                    if v > cur:
                        cur, taus = v, tt
        if cur > best[0]:
            best = (cur, [Fr(t, 200) for t in taus])
    return best


def work(tj):
    z0 = ZK(tj)
    if sum(len(r) + len(c) for r, c in zip(z0.R, z0.C)) < 2:
        return None
    cands = [z for z in (ZK(P) for P in pairings6(tj)) if min(z.imb) >= 0]
    if not cands:
        return ('noall', tj)
    best = (-math.inf, None, None)
    for z in cands:
        v, taus = search(z)
        if v > best[0]:
            best = (v, taus, z)
    v, taus, z = best
    sig2 = None
    for (lv, dg, o2) in bound(z, taus):
        if lv == math.inf:
            continue
        s = Fr(lv * (1 - 1e-6)).limit_denominator(10 ** 9) if lv > 0 else None
        if s is None or not psd(dg, o2, s):
            return ('fail', tj, float(v))
        sig2 = s if sig2 is None else min(sig2, s)
    if sig2 is None:
        return ('fail', tj, float(v))
    return ('ok', 8 * math.sqrt(float(sig2)), tj, z.tj, [str(t) for t in taus])


if __name__ == '__main__':
    import multiprocessing as mp
    s2, M, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        rs = [r for r in pool.imap_unordered(work, list(tuplesK(s2, M)), chunksize=16) if r is not None]
    res = sorted([r[1:] for r in rs if r[0] == 'ok'])
    fails = [r[1:] for r in rs if r[0] == 'fail']
    noall = sorted([r[1] for r in rs if r[0] == 'noall'])
    L = [f'2j1={s2}: exact chained-inequality certificate, all tuples 2j<={M} (dim Inv>=2): {len(res)} certified, '
         f'{len(fails)} not certified, {len(noall)} without an all-tall pairing, {time.time()-t0:.0f}s on {nproc} processes',
         'smallest certified q lower bounds (q_cert, tuple, ordered pairing, taus):']
    L += [f'  {r[0]:.4f}  {r[1]}  {r[2]}  {r[3]}' for r in res[:12]]
    L += ['  NOT CERTIFIED ' + str(f) for f in fails[:10]]
    L += ['  no all-tall pairing: ' + ' '.join(map(str, noall[:40]))] if noall else []
    print(chr(10).join(L))
    if out != '-':
        open(out, 'w').write(chr(10).join(L) + chr(10))

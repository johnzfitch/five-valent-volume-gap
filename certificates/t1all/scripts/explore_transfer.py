"""Layer transfer test.  For each tuple, pairing and choice of row colour:
  - is the lowest layer a row layer?
  - are all column layers no larger than the row layer below (D_k injective)?
  - theta*_v = sup{theta : Gup_v - theta Gdn_v >= 0} with Gup_v := Gamma_{v+1}^T Gamma_{v+1},
    Gdn_v := Gamma_v Gamma_v^T, at every row layer v with Gdn_v != 0 (min over v reported).
  - sigma_min(B) vs 1/||L|| with L the forward-substitution left inverse.
Usage: python explore_transfer.py SMIN SMAX LMAX"""
import sys, itertools, numpy as np
from t1core import Tmatrix, valid, colour
from w0core import pairings


def analyse(S, p1, p2, rowcol):
    a2, a3 = p1; a4, a5 = p2
    st, W = Tmatrix(S, a2, a3, a4, a5)
    if len(st) < 2:
        return None
    lay = {}
    for i, (b, c) in enumerate(st):
        lay.setdefault(b + c, []).append(i)
    Us = sorted(lay)
    G = {U: W[np.ix_(lay[U], lay[U - 2])] for U in Us if U - 2 in lay}
    isrow = lambda U: colour(S, st[lay[U][0]]) == rowcol
    res = {'bottom_row': isrow(Us[0]), 'inj': True, 'theta': np.inf, 'theta_at': None}
    for U in Us:
        if not isrow(U):
            # column layer: need rank Gamma_U == |layer U| (Gamma_U : layer U-2 -> U)
            if U - 2 in lay:
                g = G[U]
                if np.linalg.matrix_rank(g, tol=1e-9 * max(1, np.abs(g).max())) < len(lay[U]):
                    res['inj'] = False
            else:
                res['inj'] = False
            continue
        if U - 2 not in lay or U + 2 not in lay:
            continue
        Gd = G[U] @ G[U].T
        Gu = G[U + 2].T @ G[U + 2]
        if np.abs(Gd).max() < 1e-12:
            continue
        # theta* = sup{theta : Gu - theta Gd >= 0 on the whole layer}
        w, Q = np.linalg.eigh(Gu); tol = 1e-10 * max(1, w.max())
        ker = Q[:, w <= tol]; rng = Q[:, w > tol]
        if ker.shape[1] and np.abs(ker.T @ Gd @ ker).max() > 1e-8 * max(1, np.abs(Gd).max()):
            th = 0.0
        else:
            Rh = rng / np.sqrt(w[w > tol])
            th = 1 / np.linalg.eigvalsh(Rh.T @ Gd @ Rh).max()
        if th < res['theta']:
            res['theta'] = th; res['theta_at'] = (U / 2, len(lay[U]))
    rows = [i for i, x in enumerate(st) if colour(S, x) == rowcol]
    cols = [i for i, x in enumerate(st) if colour(S, x) != rowcol]
    B = W[np.ix_(rows, cols)]
    sv = np.linalg.svd(B, compute_uv=False) if B.size else np.array([0.0])
    res['smin'] = sv.min() if len(cols) <= len(rows) else 0.0
    res['ncols'] = len(cols); res['nrows'] = len(rows)
    return res


if __name__ == '__main__':
    S0, S1, L = map(int, sys.argv[1:4])
    tot = 0; good_any = 0; fails = []
    thmins = []
    for S in range(S0, S1 + 1):
        for legs in itertools.combinations_with_replacement(range(1, L + 1), 4):
            if not valid(S, *legs):
                continue
            best = None
            for p1, p2 in pairings(*legs):
                for rc in (0, 1):
                    r = analyse(S, p1, p2, rc)
                    if r is None:
                        continue
                    ok = r['bottom_row'] and r['inj']
                    key = (ok, r['theta'] if ok else -1)
                    if best is None or key > best[0]:
                        best = (key, p1, p2, rc, r)
            if best is None:
                continue
            tot += 1
            (ok, th), p1, p2, rc, r = best
            if ok:
                good_any += 1; thmins.append((th, S, legs, p1, p2, rc, r['theta_at']))
            else:
                fails.append((S, legs))
    thmins.sort(key=lambda t: t[0])
    print(f'tuples {tot}; with a pairing/colour having bottom row + injective column steps: {good_any}')
    print('no such choice:', len(fails), fails[:15])
    print('theta (best choice) distribution: frac>=1:', np.mean([t[0] >= 1 for t in thmins]))
    for t in thmins[:25]:
        print('  theta=%.4f 2s=%d legs=%s pair=%s|%s rowcol=%d at(u,size)=%s' % t)

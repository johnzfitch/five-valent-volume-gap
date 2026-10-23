"""Forward-substitution left inverse along layers (bottom-up).
Row layers = colour rowcol; requires: lowest layer is a row layer; each column layer U has
Gamma_U (layer U-2 -> U) of full row rank.  Then for (Bx)_{V} = Gamma_V x_{V-2}... we solve
x_U = Gamma_U^{T+} (y_{U-2} - Gamma_{U-2} x_{U-4}) -- i.e. L with blocks built from the
pseudo-inverse of D_k = Gamma_U^T.  Report 1/||L|| vs sigma_min(B) and transfer norms.
Usage: python explore_leftinv.py SMIN SMAX LMAX"""
import sys, itertools, numpy as np
from t1core import Tmatrix, valid, colour
from w0core import pairings


def leftinv(S, p1, p2, rowcol):
    st, W = Tmatrix(S, *p1, *p2)
    if len(st) < 2:
        return None
    lay = {}
    for i, (b, c) in enumerate(st):
        lay.setdefault(b + c, []).append(i)
    Us = sorted(lay)
    isrow = {U: colour(S, st[lay[U][0]]) == rowcol for U in Us}
    if not isrow[Us[0]]:
        return None
    rows = [i for U in Us if isrow[U] for i in lay[U]]
    cols = [i for U in Us if not isrow[U] for i in lay[U]]
    if not cols:
        return None
    ri = {i: k for k, i in enumerate(rows)}; ci = {i: k for k, i in enumerate(cols)}
    B = W[np.ix_(rows, cols)]
    L = np.zeros((len(cols), len(rows)))
    taus = []
    # process column layers bottom-up; column layer U solved from row layer U-2
    prevU = None
    for U in Us:
        if isrow[U]:
            continue
        if U - 2 not in lay:
            return None
        D = W[np.ix_(lay[U - 2], lay[U])]          # rows (layer U-2) x cols (layer U): D = Gamma_U^T
        if np.linalg.matrix_rank(D, tol=1e-9 * max(1, np.abs(D).max())) < D.shape[1]:
            return None
        Dp = np.linalg.pinv(D)
        # (Bx)_{U-2} = D x_U + E x_{U-4}, E = Gamma_{U-2} block rows layer U-2, cols layer U-4
        rr = [ri[i] for i in lay[U - 2]]
        cc = [ci[i] for i in lay[U]]
        # x_U = Dp (y_{U-2} - E x_{U-4}); in terms of L: L[cc,:] = Dp (I_{rr} - E L[c_prev,:])
        Lrow = np.zeros((len(cc), len(rows)))
        Lrow[:, rr] = Dp
        if U - 4 in lay and not isrow[U - 4]:
            E = W[np.ix_(lay[U - 2], lay[U - 4])]
            cp = [ci[i] for i in lay[U - 4]]
            Lrow -= Dp @ E @ L[cp, :]
            taus.append(np.linalg.norm(Dp @ E, 2))
        L[cc, :] = Lrow
    # sanity: L B = I
    err = np.abs(L @ B - np.eye(len(cols))).max()
    sv = np.linalg.svd(B, compute_uv=False)
    return dict(smin=sv.min(), invL=1 / np.linalg.norm(L, 2), err=err,
                taumax=max(taus) if taus else 0, ncols=len(cols), nrows=len(rows))


if __name__ == '__main__':
    S0, S1, Lm = map(int, sys.argv[1:4])
    tot = 0; ok = 0; worst = []; notok = []
    for S in range(S0, S1 + 1):
        for legs in itertools.combinations_with_replacement(range(1, Lm + 1), 4):
            if not valid(S, *legs):
                continue
            st, W = Tmatrix(S, *legs)
            if len(st) < 2:
                continue
            tot += 1
            best = None
            for p1, p2 in pairings(*legs):
                for rc in (0, 1):
                    r = leftinv(S, p1, p2, rc)
                    if r is None:
                        continue
                    if best is None or r['invL'] > best[0]['invL']:
                        best = (r, p1, p2, rc)
            if best is None:
                notok.append((S, legs)); continue
            ok += 1
            r = best[0]
            worst.append((8 * r['invL'], 8 * r['smin'], r['invL'] / r['smin'], r['taumax'], S, legs, best[1], best[2], best[3], r['err']))
    print(f'tuples {tot}, with a bottom-up left inverse: {ok}, without: {len(notok)}')
    worst.sort()
    print('lowest 8/||L|| (q-scale bound) cases:')
    for w in worst[:20]:
        print('  8/|L|=%.3f  q_B=%.3f ratio=%.3f taumax=%.3f 2s=%d legs=%s %s|%s rc=%d err=%.1e' % w)
    rat = sorted(worst, key=lambda w: w[2])
    print('worst ratio (1/|L|)/smin:')
    for w in rat[:10]:
        print('  8/|L|=%.3f  q_B=%.3f ratio=%.3f taumax=%.3f 2s=%d legs=%s %s|%s rc=%d err=%.1e' % w)
    print('sample of tuples without:', notok[:20])

"""Numerical test of Proposition L (layer left inverse) and its transfer bound:
  for each column layer U whose row layer V=U-2 has a down block Gamma_V != 0 and up block Gamma_U of
  full row rank:  tau^2 := ||Gamma_U^{T+} Gamma_V||^2 = 1/theta*_V (identity), where
  theta*_V = sup{theta : Gamma_U^T Gamma_U - theta Gamma_V Gamma_V^T >= 0 on the whole layer V}
  (theta* := 0 if ker(Gamma_U^T Gamma_U) is not contained in ker(Gamma_V Gamma_V^T)).
  Also checks L B = I and 1/||L|| <= sigma_min(B).
Usage: python check_propL.py SMIN SMAX LMAX"""
import sys, itertools, numpy as np
from t1core import Tmatrix, valid, colour
from w0core import pairings
from explore_leftinv import leftinv

S0, S1, L = map(int, sys.argv[1:4])
nsteps = 0; badstep = 0; ntup = 0; badinv = 0; worst = 0
for S in range(S0, S1 + 1):
    for legs in itertools.combinations_with_replacement(range(1, L + 1), 4):
        if not valid(S, *legs):
            continue
        for p1, p2 in pairings(*legs):
            st, W = Tmatrix(S, *p1, *p2)
            if len(st) < 2:
                continue
            lay = {}
            for i, (b, c) in enumerate(st):
                lay.setdefault(b + c, []).append(i)
            for rc in (0, 1):
                for U in sorted(lay):
                    if colour(S, st[lay[U][0]]) == rc:      # U must be a column layer
                        continue
                    V = U - 2
                    if V not in lay or V - 2 not in lay:
                        continue
                    GU = W[np.ix_(lay[U], lay[V])]        # Gamma_U : layer V -> U
                    GV = W[np.ix_(lay[V], lay[V - 2])]    # Gamma_V : layer V-2 -> V
                    if np.linalg.matrix_rank(GU, 1e-9 * max(1, np.abs(GU).max())) < GU.shape[0]:
                        continue
                    tau2 = np.linalg.norm(np.linalg.pinv(GU.T) @ GV, 2) ** 2
                    Gd = GV @ GV.T; Gu = GU.T @ GU
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
                    nsteps += 1
                    if th > 0:
                        dev = abs(tau2 * th - 1)
                        worst = max(worst, dev)
                        if dev > 1e-6:
                            badstep += 1
                r = leftinv(S, p1, p2, rc)
                if r is not None:
                    ntup += 1
                    if r['err'] > 1e-8 or r['invL'] > r['smin'] * (1 + 1e-8):
                        badinv += 1
print(f'transfer steps tested {nsteps}, steps with theta*>0 violating tau^2 = 1/theta*: {badstep}, max |tau^2 theta* - 1| = {worst:.2e}')
print(f'left inverses built {ntup}, failures of (LB=I and 1/||L|| <= sigma_min): {badinv}')

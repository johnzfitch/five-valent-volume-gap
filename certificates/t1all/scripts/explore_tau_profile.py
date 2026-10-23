"""Per-layer transfer norms and up/down Gram comparison for one tuple, all pairings and row colours.
Usage: python explore_tau_profile.py S a2 a3 a4 a5"""
import sys, numpy as np
from t1core import Tmatrix, colour
from w0core import pairings

S, *legs = map(int, sys.argv[1:6])
for p1, p2 in pairings(*legs):
    st, W = Tmatrix(S, *p1, *p2)
    lay = {}
    for i, (b, c) in enumerate(st):
        lay.setdefault(b + c, []).append(i)
    Us = sorted(lay)
    for rc in (0, 1):
        isrow = {U: colour(S, st[lay[U][0]]) == rc for U in Us}
        line = []
        for U in Us:
            tag = 'R' if isrow[U] else 'C'
            info = f'{tag}{U/2:g}[{len(lay[U])}]'
            if not isrow[U] and U - 2 in lay and U - 4 in lay:
                D = W[np.ix_(lay[U - 2], lay[U])]
                E = W[np.ix_(lay[U - 2], lay[U - 4])]
                if np.linalg.matrix_rank(D, 1e-9 * max(1, np.abs(D).max())) == D.shape[1]:
                    tau = np.linalg.norm(np.linalg.pinv(D) @ E, 2)
                    info += f't{tau:.2f}'
                else:
                    info += 'x'
            line.append(info)
        print(f'{p1}|{p2} rows=colour{rc}: ' + ' '.join(line))

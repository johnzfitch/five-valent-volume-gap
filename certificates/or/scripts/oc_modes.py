"""Location of the lowest nonzero mode of T, W0 (core K) and K_G: mean u, weight within 3 of the corner u=s,
weight on the walls |d|=s.  Usage: python oc_modes.py out.txt"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_core import matrix
out = sys.argv[1]; L = []
for tj in [(3, 31, 32, 32, 32), (5, 16, 17, 17, 17)]:
    for kind, name in (('T', 'T'), ('K', 'W0'), ('KG', 'K_G')):
        S, W = matrix(tj, kind); w, v = np.linalg.eigh(W); a = np.abs(w); nz = a > 1e-9 * a.max()
        i = np.argmin(np.where(nz, a, 1e99)); psi = v[:, i] ** 2
        u = np.array([(B + C) / 2 for B, C in S]); d = np.array([(B - C) / 2 for B, C in S]); s = tj[0] / 2
        L.append(f"{tj} {name:4s} mean u={float((psi*u).sum()):.1f} (u in [{u.min()},{u.max()}])  "
                 f"weight u<=s+3: {psi[u <= s + 3].sum():.2f}  weight on walls |d|=s: {psi[np.abs(np.abs(d)-s) < 1e-9].sum():.2f}")
open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

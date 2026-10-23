"""The [DC] bound 2 sqrt(mu) lam(W0) (best of three pairings) and q along its minimising family
(2s=3; a, a+1, a+1, a+1), to large a.  Usage: python oc_b0_family.py out.txt"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_fast import build, smin_nz, pairings
out = sys.argv[1]; L = ["a, q, best 2sqrt(mu)lam(W0), lam(W0) at the best pairing, mu"]
for a in [9, 19, 31, 63, 127, 255, 511]:
    tj = (3, a, a + 1, a + 1, a + 1)
    best = (0, None, None)
    for k, P in enumerate(pairings(tj)):
        o = build(P, ('T', 'K') if k == 0 else ('K',))
        if k == 0:
            q = 8 * smin_nz(o['T'])[0]
        l0 = smin_nz(o['K'])[0]
        b0 = 2 * math.sqrt(o['mu']) * l0
        if b0 > best[0]:
            best = (b0, l0, o['mu'])
    L.append(f"  {tj}: q={q:.4f}  b0={best[0]:.4f}  lam(W0)={best[1]:.4f}  mu={best[2]:.4f}")
    print(L[-1], flush=True)
open(out, 'w').write('\n'.join(L) + '\n')

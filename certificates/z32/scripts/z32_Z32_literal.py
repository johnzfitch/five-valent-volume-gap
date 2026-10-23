"""Literal Conjecture Z_{3/2} of [OR] along large-spin families: using [OR]'s own code (oc_s32_plan, imported
read-only), the best over admissible pairings and uniform end deletions of  8*min_k sigma_min(D_k)  and the
coupling norm used (min(||N D0^-1||, ||D1^-1 N||) at the configuration maximising the plan bound), plus the
coupling norms minimised over all configurations.
Usage: python z32_Z32_literal.py out.txt"""
import sys, os
import numpy as np
OR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'or', 'scripts'))
sys.path.insert(0, OR)
from oc_s32_plan import plan_bound
from oc_fast import pairings
out = sys.argv[1]
fams = {'thin (3/2; a/2,(a+1)/2,(a+1)/2,(a+1)/2)': lambda a: (3, a, a + 1, a + 1, a + 1),
        'equal (3/2; a/2,a/2,a/2,(a+1)/2)': lambda a: (3, a, a, a, a + 1)}
L = []
for name, f in fams.items():
    L.append(name)
    for a in [3, 7, 15, 31, 63, 127]:
        tj = f(a)
        best = None; cmin = np.inf
        for P in pairings(tj):
            for end in ('low', 'high'):
                r = plan_bound(P, end)
                if r is None:
                    continue
                sB, plan, th, imb, sD = r
                cmin = min(cmin, th[0] if th[0] > 0 else np.inf)
                if best is None or plan > best[0]:
                    best = (plan, sB, th, sD, P, end, imb)
        plan, sB, th, sD, P, end, imb = best
        L.append(f'  2j={a+1:4d}: plan 8*bound={8*plan:8.3f}  8*sigma(B\')={8*sB:8.3f}  8*min sigma(D_k)={8*sD:8.3f}  '
                 f'coupling used={th[0]:.4f}  min coupling over configs={cmin:.4f}  {P} {end} {imb}')
        print(L[-1], flush=True)
open(out, 'w').write('\n'.join(L) + '\n')

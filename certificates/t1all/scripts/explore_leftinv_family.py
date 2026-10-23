"""Bottom-up left-inverse bound 8/||L|| (best pairing/colour) vs true q along growing families."""
import numpy as np
from t1core import qmin, valid
from w0core import pairings
from explore_leftinv import leftinv

fams = {
    'unbal (k,k,2k-1,2k)': lambda k: (k, k, 2 * k - 1, 2 * k),
    'bal (k,k,k+1,k+1)': lambda k: (k, k, k + 1, k + 1),
    'eq (k,k,k,k)': lambda k: (k, k, k, k),
    'G1-like (k-1,k,k,k)': lambda k: (k - 1, k, k, k),
}
for S in (5, 6, 9):
    for name, f in fams.items():
        for k in (5, 9, 17, 33, 65):
            legs = f(k)
            if not valid(S, *legs):
                legs = (legs[0], legs[1], legs[2] + 1, legs[3] + 1) if name.startswith('unbal') else (legs[0] + 1,) + legs[1:]
                if not valid(S, *legs):
                    continue
            q, dk, n = qmin(S, *legs)
            best = None
            for p1, p2 in pairings(*legs):
                for rc in (0, 1):
                    r = leftinv(S, p1, p2, rc)
                    if r is not None and (best is None or r['invL'] > best['invL']):
                        best = r
            lb = 8 * best['invL'] if best else float('nan')
            print(f'2s={S} {name:22s} legs={legs} dim={n:5d} q={q:9.3f} 8/|L|best={lb:10.3e}', flush=True)

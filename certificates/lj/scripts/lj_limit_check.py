"""Check lj_limit_core link formulas and site sets against lj_core at several large s."""
import sys, os, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import lj_core as L, lj_limit_core as LC
from fractions import Fraction as Fr
mx = 0; n = 0; nerr = 0
from lj_limit_cert import s_min_struct
for t2 in itertools.product(range(1, 7), repeat=3):
    for M2 in range(-sum(t2), sum(t2) + 1, 2):
        S, Ls = LC.links(t2, M2)
        if not S: continue
        s2lo = int(2 * s_min_struct(t2, M2))
        for s2 in list(range(s2lo, s2lo + 6)) + [2 * 60 + (M2 % 2)]:
            j52 = s2 - M2
            if j52 < t2[2] or (s2 + sum(t2) + j52) % 2: continue
            tj = (s2, t2[0], t2[1], t2[2], j52)
            o = L.build(tj)
            # map lj_core sites to (b, gamma)
            mp = {(Fr(int(b), 2), Fr(int(c), 2) - Fr(s2, 2)): i for i, (b, c) in enumerate(zip(o['Bd'], o['Cd']))}
            assert set(mp) == set(S), (tj, sorted(mp), S)
            W = {}
            for lo, up, w in zip(o['lo'], o['up'], o['w']):
                if w > 0: W[frozenset((int(lo), int(up)))] = w
            Wf = {}
            for lk in Ls:
                a = mp[S[lk[0]]]; bb = mp[S[lk[1]]]
                Wf[frozenset((a, bb))] = LC.w_exact(lk, s2 / 2, t2, M2)
                # error bound sanity: actual |w - s w_inf| <= bound(s0 = s)
                winf = 0.5 * float(lk[3]['c2']) ** .5
                act = abs(Wf[frozenset((a, bb))] - s2 / 2 * winf)
                bnd = 0.25 * float(lk[3]['c2']) ** .5 * float(LC.err_bound(lk, Fr(s2, 2)))
                if act > bnd * (1 + 1e-12) + 1e-12: nerr += 1
            assert set(W) == set(Wf), tj
            for k in W:
                mx = max(mx, abs(W[k] - Wf[k]) / W[k])
            n += 1
print(f"cases {n}: site sets equal, link sets equal, max rel link diff {mx:.2e}, error-bound violations {nerr}")

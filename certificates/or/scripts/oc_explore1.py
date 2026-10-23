"""Exploration 1: open-region tuples.  For each: q = 8 lam(T); closing-note bound 2 sqrt(mu) lam(W0);
splits: omega-only scaling with core KF (Stark x sqrt F), phi-only scaling with core KG (Stark x sqrt G).
Bound from a split with core W_a and site weights sig_a^2:  q >= 2 sqrt(min_E sig_a^2 * min_O sig_a^2) lam(W_a).
Usage: python oc_explore1.py M
"""
import sys, os, math
import numpy as np
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_core import *


def weights(tj):
    s = Fraction(tj[0], 2)
    S = sites(tj)
    ds = sorted({Fraction(B - C, 2) for B, C in S}); us = sorted({Fraction(B + C, 2) for B, C in S})
    phi = {ds[0]: Fraction(1)}
    for d in ds[1:]:
        phi[d] = F(s, d - Fraction(1, 2)) / phi[d - 1]
    om = {us[0]: Fraction(1)}
    for u in us[1:]:
        om[u] = G(s, u) / om[u - 1]
    P = [float(phi[Fraction(B - C, 2)]) for B, C in S]
    O = [float(om[Fraction(B + C, 2)]) for B, C in S]
    return S, np.array(P), np.array(O)


def mm(S, w, s2):
    e = [w[i] for i, p in enumerate(S) if colour(p, s2) == 0]
    o = [w[i] for i, p in enumerate(S) if colour(p, s2) == 1]
    return min(e) * min(o)


def analyse(tj):
    S, P, O = weights(tj)
    lT = lam(tj, 'T'); l0 = lam(tj, 'K'); lF = lam(tj, 'KF'); lG = lam(tj, 'KG')
    if lT is None:
        return None
    r = dict(tj=tj, q=8 * lT, l0=l0, lF=lF, lG=lG)
    r['b0'] = 2 * math.sqrt(mm(S, P * O, tj[0])) * l0 if l0 else 0
    r['bF'] = 2 * math.sqrt(mm(S, O, tj[0])) * lF if lF else 0
    r['bG'] = 2 * math.sqrt(mm(S, P, tj[0])) * lG if lG else 0
    return r


if __name__ == '__main__':
    M = int(sys.argv[1])
    rows = []
    for tj in tuples_sorted(M, 3):
        if not is_open(tj):
            continue
        best = None
        for P in pairings(tj):
            r = analyse(P)
            if r is None:
                continue
            if best is None:
                best = r; best['all'] = []
            best['all'].append((r['b0'], r['bF'], r['bG'], r['l0'], r['lF'], r['lG']))
        if best:
            rows.append(best)
    print(f"open tuples with 2jmax<={M}: {len(rows)}")
    for key in ('q',):
        rows.sort(key=lambda r: r[key])
        print('smallest q:', [(r['tj'], round(r['q'], 4)) for r in rows[:6]])
    for k, name in ((0, 'b0 (closing note)'), (1, 'bF (omega only)'), (2, 'bG (phi only)')):
        vals = [(max(a[k] for a in r['all']), r['tj'], r['q']) for r in rows]
        vals.sort()
        print(name, 'min over tuples of best-pairing bound:', [(round(v, 4), t, round(q, 3)) for v, t, q in vals[:5]])
        print('   min ratio q/bound:', min(q / v if v else 1e9 for v, t, q in vals))
    for k, name in ((3, 'lam W0'), (4, 'lam KF'), (5, 'lam KG')):
        vals = sorted((max(a[k] or 0 for a in r['all']), r['tj']) for r in rows)
        print(name, 'min over tuples of best-pairing:', vals[:5])

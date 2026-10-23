"""Remaining-obligation survey.  For every tuple (j1 >= 5/2, sorted legs 2..5 by S4 symmetry) that is
NOT covered by Thm 6.5 (closed class + odd sum / spin-1/2 pair), the ladder corollary (a leg 1/2), or
C_6 (three legs <= 3), record the best (over 3 pairings x 2 colour classes) lambda(W0) and
lambda(W0)*(j1+1).  Also check q >= 2 sqrt(mu) lambda(W0) with the actual mu of that pairing.
Usage: python obligation_survey.py SMIN SMAX LMAX"""
import sys, itertools, math, numpy as np
from fractions import Fraction as Fr
from t1core import qmin, valid, sites, colour
from w0core import covered, pairings, W0_block, lam


def mu_actual(S, p1, p2):
    """mu = min_colour0 sigma^2 * min_colour1 sigma^2 with sigma^2 = phi(d) omega(u)."""
    st = sites(S, *p1, *p2)
    if not st:
        return None
    s = Fr(S, 2)
    F = lambda t: (s + Fr(1, 2)) ** 2 - t * t
    G = lambda u: (u + s + 1) * (u - s)
    ds = sorted({Fr(b - c, 2) for b, c in st}); us = sorted({Fr(b + c, 2) for b, c in st})
    phi = {ds[0]: 1.0}
    for d in ds[1:]:
        phi[d] = float(F(d - Fr(1, 2))) / phi[d - 1]
    om = {us[0]: 1.0}
    for u in us[1:]:
        om[u] = float(G(u)) / om[u - 1]
    mins = [math.inf, math.inf]
    for b, c in st:
        v = phi[Fr(b - c, 2)] * om[Fr(b + c, 2)]
        k = colour(S, (b, c))
        mins[k] = min(mins[k], v)
    return mins[0] * mins[1]


S0, S1, L = map(int, sys.argv[1:4])
rows = []; viol = 0; ncount = 0
for S in range(S0, S1 + 1):
    for legs in itertools.combinations_with_replacement(range(1, L + 1), 4):
        if not valid(S, *legs):
            continue
        if sum(1 for a in legs if a <= 6) >= 3:      # C_6
            continue
        typ = covered(S, *legs)
        if typ not in ('W2', 'open'):
            continue
        q, dk, n = qmin(S, *legs)
        if q is None:
            continue
        ncount += 1
        best = (0, None)
        for p1, p2 in pairings(*legs):
            mu = mu_actual(S, p1, p2)
            for col in (0, 1):
                B0, c_, r_ = W0_block(S, p1, p2, col)
                l = lam(B0)
                if l is None:
                    continue
                if q < 2 * math.sqrt(mu) * l * (1 - 1e-9):
                    viol += 1
                if l > best[0]:
                    best = (l, (p1, p2, col))
        rows.append((best[0] * (S / 2 + 1), best[0], q, S, legs, typ))
rows.sort()
print(f'uncovered tuples (W2/open, not C6, no leg 1/2) with 5<=2j1<={S1}, legs<={L}/2: {ncount}')
print(f'violations of q >= 2 sqrt(mu) lambda(W0) (any pairing/class): {viol}')
print('smallest lambda(W0)*(j1+1):')
for r in rows[:15]:
    print('  %.4f  lambda=%.4f q=%.3f 2j1=%d legs=%s %s' % r)
byS = {}
for r in rows:
    byS.setdefault(r[3], []).append(r)
for S in sorted(byS):
    rs = byS[S]
    print(f'2j1={S:3d}: n={len(rs):6d} min lambda={min(r[1] for r in rs):.4f}  min lambda*(j1+1)={min(r[0] for r in rs):.4f}  min q={min(r[2] for r in rs):.3f}')

"""Exact check: on interior diagonals of W0, the defect Delta(x) := Up(x) - Dn(x) + d/2 equals
(eps1(b) - eps2(c))/2 + [u = s](h1(b)^2 - h2(c)^2), with eps(L) := 2 delta(L) + L + 1/2, delta(L) = h(L+1)^2 - h(L)^2
(h := 0 outside the chain), i.e. Lemma L of the previous round plus the corner term.  Exact rationals."""
from fractions import Fraction as Fr
from t1core import pair_params, h2stark, sites
def run(S, legs):
    a2, a3, a4, a5 = legs
    n1, m1 = pair_params(a2, a3); n2, m2 = pair_params(a4, a5)
    st = set(sites(S, *legs)); s = Fr(S, 2)
    H1 = lambda L2: h2stark(n1, m1, L2) if m1 < L2 < n1 else Fr(0)
    H2 = lambda L2: h2stark(n2, m2, L2) if m2 < L2 < n2 else Fr(0)
    bad = n = ncorner = 0
    for (b, c) in st:
        d = Fr(b - c, 2); u = Fr(b + c, 2)
        if not (-s < d < s): continue
        up = (H1(b + 2) if (b + 2, c) in st else 0) + (H2(c) if (b, c - 2) in st else 0)
        dn = (H1(b) if (b - 2, c) in st else 0) + (H2(c + 2) if (b, c + 2) in st else 0)
        e1 = 2 * (H1(b + 2) - H1(b)) + Fr(b, 2) + Fr(1, 2)
        e2 = 2 * (H2(c + 2) - H2(c)) + Fr(c, 2) + Fr(1, 2)
        pred = (e1 - e2) / 2 + ((H1(b) - H2(c)) if u == s else 0)
        n += 1; ncorner += (u == s)
        if up - dn + d / 2 != pred: bad += 1
    return n, ncorner, bad
tot = [0, 0, 0]
for S in range(1, 12):
    for a2 in range(1, 9):
        for a3 in range(1, 9):
            for a4 in range(1, 9):
                for a5 in range(1, 9):
                    if (S + a2 + a3 + a4 + a5) % 2: continue
                    tot = [x + y for x, y in zip(tot, run(S, (a2, a3, a4, a5)))]
print(f"interior-diagonal sites {tot[0]} (corner sites {tot[1]}), 2j1<=11, doubled legs 1..8: mismatches {tot[2]}")

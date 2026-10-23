"""Validation against brute force: T = (1/4) S W0 S (paper Theorem 6.2, S^2 = phi(d) omega(u)) built from
w0sparse.build must have the spectrum of T = J1.(J2xJ3 + J4xJ5) computed by the full tensor product
(t1core.brute, no recoupling theory).  Also checks q >= 2 sqrt(mu) lambda(W0) (Lemmas 6.3-6.4) per tuple.
All tuples with every doubled spin <= 5 (2j1 = 1..5).  Float64."""
import itertools, math
import numpy as np
from fractions import Fraction as Fr
from t1core import brute
from w0sparse import build, colour

worst = 0; cnt = 0; viol = 0
for S in range(1, 6):
    for legs in itertools.product(range(1, 6), repeat=4):
        if (S + sum(legs)) % 2:
            continue
        sites, W = build(S, *legs)
        if not sites:
            continue
        s = Fr(S, 2)
        F = lambda t: (s + Fr(1, 2)) ** 2 - t * t
        G = lambda u: (u + s + 1) * (u - s)
        ds = sorted({Fr(b - c, 2) for b, c in sites}); us = sorted({Fr(b + c, 2) for b, c in sites})
        phi = {ds[0]: 1.0}
        for d in ds[1:]:
            phi[d] = float(F(d - Fr(1, 2))) / phi[d - 1]
        om = {us[0]: 1.0}
        for u in us[1:]:
            om[u] = float(G(u)) / om[u - 1]
        sig = np.array([math.sqrt(phi[Fr(b - c, 2)] * om[Fr(b + c, 2)]) for b, c in sites])
        T = 0.25 * (sig[:, None] * W.toarray() * sig[None, :])
        e1 = np.sort(np.linalg.eigvalsh(T))
        e2 = brute(S, *legs)
        if len(e1) != len(e2):
            print('dimension mismatch', S, legs, len(e1), len(e2)); worst = 1; continue
        worst = max(worst, np.abs(e1 - e2).max() / max(1, np.abs(e2).max())); cnt += 1
        # Lemma 6.3-6.4 check
        nz = np.abs(e2)[np.abs(e2) > 1e-9]
        if len(nz) == 0:
            continue
        q = 8 * nz.min()
        ev = np.abs(np.linalg.eigvalsh(W.toarray())); lam = ev[ev > 1e-9 * max(1, ev.max())].min()
        mins = [math.inf, math.inf]
        for x, v in zip(sites, sig):
            k = colour(S, x); mins[k] = min(mins[k], v * v)
        mu = mins[0] * mins[1]
        if q < 2 * math.sqrt(mu) * lam * (1 - 1e-9):
            viol += 1
print(f'tuples {cnt}: max relative spectral difference T(from W0) vs brute force {worst:.1e}; '
      f'violations of q >= 2 sqrt(mu) lambda(W0): {viol}')

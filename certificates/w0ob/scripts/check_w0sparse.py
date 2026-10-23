"""Validate w0sparse against the prior-round builder w0core (dense) on all tuples with small spins."""
import itertools, numpy as np
from w0core import W0_block, lam
from w0sparse import build, colour, lam_sparse

worst = 0; cnt = 0
for S in range(3, 10, 2):
    for legs in itertools.product(range(1, 8), repeat=4):
        if (S + sum(legs)) % 2:
            continue
        a2, a3, a4, a5 = legs
        B0, c_, r_ = W0_block(S, (a2, a3), (a4, a5), 0)
        l1 = lam(B0)
        sites, W = build(S, *legs)
        if l1 is None:
            continue
        ev = np.abs(np.linalg.eigvalsh(W.toarray()))
        nz = ev[ev > 1e-9 * max(1, ev.max())]
        l2 = nz.min()
        worst = max(worst, abs(l1 - l2) / l1); cnt += 1
print('tuples compared', cnt, 'max rel diff', worst)

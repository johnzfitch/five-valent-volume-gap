"""Counterexample search for (RO): best-pairing lambda(W0) on uncovered tuples beyond the prior survey box.
Tuples: 2j1 = S, sorted doubled legs a2<=a3<=a4<=a5, no leg 1/2 (a>=2), fewer than three legs with a<=6,
not covered by closed-class+odd-sum (w0core.covered gives 'W2' or 'open').
Families sampled (doubled): random structured multisets with legs up to LMAX, several regimes.
Usage: python search_RO.py SEED NSAMPLES LMAX OUT"""
import sys, random, math, time
import numpy as np
from w0core import covered
from w0sparse import lam_sparse, pairings

seed, NS, LMAX, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = random.Random(seed)
Ss = list(range(5, 42))


def sample():
    S = rng.choice(Ss)
    mode = rng.randrange(6)
    if mode == 0:   # all four large, small gaps (bulk)
        base = rng.randrange(20, LMAX - 8)
        legs = [base + rng.randrange(0, 5) for _ in range(4)]
    elif mode == 1:  # two large, two moderate
        a = rng.randrange(7, 40); b = rng.randrange(7, 40)
        c = rng.randrange(40, LMAX); d = c + rng.randrange(-6, 7)
        legs = [a, b, c, max(7, d)]
    elif mode == 2:  # two small legs (2..6) and two large
        a = rng.randrange(2, 7); b = rng.randrange(7, 14)
        c = rng.randrange(20, LMAX); d = c + rng.randrange(-S, S + 1)
        legs = [a, b, c, max(2, d)]
    elif mode == 3:  # unbalanced pairs: (x,x+e) and (y,y+f) with y >> x
        x = rng.randrange(7, 30); y = rng.randrange(60, LMAX)
        legs = [x, x + rng.randrange(0, 4), y, y + rng.randrange(0, 4)]
    elif mode == 4:  # large gaps
        legs = sorted(rng.randrange(7, LMAX) for _ in range(4))
    else:            # near S scale
        legs = [max(2, S // 2 + rng.randrange(-3, 12)) for _ in range(3)] + [rng.randrange(20, LMAX)]
    legs = sorted(min(LMAX, max(2, l)) for l in legs)
    return S, tuple(legs)


rows = []
t0 = time.time()
seen = set()
tries = 0
while len(rows) < NS and tries < 50 * NS:
    tries += 1
    S, legs = sample()
    if (S + sum(legs)) % 2:
        legs = (legs[0], legs[1], legs[2], legs[3] + 1)
    if (S, legs) in seen:
        continue
    seen.add((S, legs))
    if 2 in legs and False:
        pass
    if min(legs) < 2 or sum(1 for a in legs if a <= 6) >= 3:
        continue
    typ = covered(S, *legs)
    if typ not in ('W2', 'open'):
        continue
    best = 0.0; per = []
    for p in pairings(*legs):
        l, N = lam_sparse(S, p)
        per.append(l if l is not None else float('nan'))
        if l is not None and l > best:
            best = l
    if best == 0.0:
        continue
    rows.append((best * (S / 2 + 1), best, S, legs, typ, per))
rows.sort()
with open(out, 'w') as f:
    f.write(f'seed={seed} samples={len(rows)} LMAX={LMAX} time={time.time()-t0:.0f}s\n')
    for S in sorted(set(r[2] for r in rows)):
        rs = [r for r in rows if r[2] == S]
        m = min(rs, key=lambda r: r[1])
        f.write(f'2j1={S:3d} n={len(rs):5d} min best-lambda={m[1]:.5f} at {m[3]} {m[4]}\n')
    f.write('smallest lambda*(j1+1):\n')
    for r in rows[:30]:
        f.write('  %.4f lam=%.5f 2j1=%d legs=%s %s per-pairing=%s\n' % (r[0], r[1], r[2], r[3], r[4], ['%.4f' % x for x in r[5]]))
    rl = sorted(rows, key=lambda r: r[1])
    f.write('smallest lambda:\n')
    for r in rl[:30]:
        f.write('  lam=%.5f 2j1=%d legs=%s %s per-pairing=%s\n' % (r[1], r[2], r[3], r[4], ['%.4f' % x for x in r[5]]))
print('done', len(rows), time.time() - t0)

"""Survey: best zigzag left-inverse bound 8/||L|| over pairings and readings, against q.
Tuples: random uncovered-type (W2/open by the paper's rule, implemented in w0core.covered), no leg 1/2,
fewer than three legs <= 3, doubled legs in [lo, hi].  Usage: survey_zigzag.py S count lo hi seed"""
import numpy as np, sys, time
from zigzag import readings, leftinv, tdiag
from t1core import qmin
from w0core import pairings, covered
S, count, lo, hi, seed = (int(x) for x in sys.argv[1:6])
rng = np.random.default_rng(seed)
t0 = time.time(); done = 0; noadm = 0; ratios = []; worst = None
while done < count:
    legs = tuple(sorted(int(x) for x in rng.integers(lo, hi + 1, 4)))
    if (S + sum(legs)) % 2: continue
    if 1 in legs or sum(1 for a in legs if a <= 6) >= 3: continue
    cv = covered(S, *legs)
    if cv not in ('W2', 'open'): continue
    q, dk, n = qmin(S, *legs)
    if q is None: continue
    best = None; bestlab = None
    for (p1, p2) in pairings(*legs):
        lg = (p1[0], p1[1], p2[0], p2[1])
        data = tdiag(S, lg)
        for lab, st in readings(S).items():
            b = leftinv(S, lg, st, data)
            if b is not None and (best is None or b > best):
                best, bestlab = b, (p1, p2, lab)
    done += 1
    if best is None:
        noadm += 1
        print("NO ADMISSIBLE READING", S, legs, cv, f"q={q:.3f}")
        continue
    ratios.append(best / q)
    if worst is None or best / q < worst[0]:
        worst = (best / q, legs, cv, q, best, bestlab)
print(f"2s={S} tuples={done} no-admissible={noadm} min ratio bound/q={min(ratios):.4f} median={np.median(ratios):.4f} "
      f"worst={worst}  time {time.time()-t0:.1f}s")

"""Validate lj_core: q against ls_core (large-spin folder, read-only import) on all tuples with 2j<=7,
and the congruence W = (1/4) S W0 S link by link."""
import sys, os, itertools, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(1, os.path.join(here, '..', '..', 'ls', 'scripts'))
import lj_core as L
import ls_core as R
mx = 0; mxc = 0; n = 0
for tj in itertools.product(range(1, 8), repeat=5):
    if sum(tj) % 2: continue
    o = L.build(tj)
    if o['nS'] < 2: continue
    M, _, _ = L.block(o); l, _ = L.nzmin_sv(M)
    qr, _ = R.qmin(tj)
    if l is None or qr is None:
        assert (l is None) == (qr is None), tj
        continue
    n += 1
    mx = max(mx, abs(8 * l - qr[0]) / qr[0])
    sg = np.sqrt(o['sig2'])
    pred = 0.25 * sg[o['lo']] * sg[o['up']] * o['w0']
    mxc = max(mxc, np.max(np.abs(pred - o['w']) / np.maximum(o['w'], 1e-300)))
print(f"tuples compared {n}; max rel diff q vs ls_core {mx:.2e}; max rel diff of W vs S W0 S/4 {mxc:.2e}")

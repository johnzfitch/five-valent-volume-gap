"""Per-j1 minimum of q over the finite certificate's tuples (2j_max <= 38), read-only from
research/dichotomy_finite_cert_20261021/outputs/cert_M*.jsonl.  Note t[0] = 2 j1 (doubled)."""
import json, glob, os, sys
src = os.path.join(os.path.dirname(__file__), '..', '..', 'fc', 'outputs')
best = {}
for f in sorted(glob.glob(os.path.join(src, 'cert_M*.jsonl'))):
    for line in open(f):
        r = json.loads(line)
        if r.get('skip') or r.get('qmin') is None:
            continue
        s2 = r['t'][0]
        if s2 not in best or r['qmin'] < best[s2][0]:
            best[s2] = (r['qmin'], tuple(r['t']))
print("2j1  min q (2j_max<=38)  q/j1   argmin (doubled spins)")
for s2 in sorted(best):
    q, t = best[s2]
    print(f"{s2:3d}  {q:12.6f}  {q/(s2/2):7.4f}  {t}")

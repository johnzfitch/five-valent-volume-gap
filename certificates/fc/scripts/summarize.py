"""Machine check of the certificate outputs: for each level file cert_MXXX.jsonl, recompute the tuple list
(run_cert.tuples), check that every tuple appears exactly once and that every nonempty tuple has cert == true.
Prints the largest 2j_max such that ALL levels 1..M are complete and certified, and per-level minima of q."""
import json, glob, os, sys
from run_cert import tuples
lines = []; good_upto = 0; ok_chain = True; glob_min = (9e9, None)
for M in range(1, 200):
    fn = f"../outputs/cert_M{M:03d}.jsonl"
    if not os.path.exists(fn): break
    recs = [json.loads(l) for l in open(fn)]
    got = sorted(tuple(r['t']) for r in recs); want = sorted(tuples(M))
    complete = got == want
    bad = [r['t'] for r in recs if not r.get('skip') and not r['cert']]
    qs = [(r['qmin'], tuple(r['t'])) for r in recs if not r.get('skip') and r.get('qmin') is not None]
    qm = min(qs) if qs else (None, None)
    if qm[0] is not None and qm[0] < glob_min[0]: glob_min = qm
    kers = sum(1 for r in recs if not r.get('skip') and r['kerT'] > 0)
    lines.append(f"2j_max={M:3d}: tuples {len(recs):8d} complete={complete} uncertified={len(bad)} tuples_with_kernel={kers} min q={qm[0]} at {qm[1]}")
    if ok_chain and complete and not bad: good_upto = M
    else: ok_chain = False
lines.append(f"ALL levels 1..{good_upto} complete and certified: every nonzero eigenvalue of 8T has modulus >= 4.5579607")
lines.append(f"global minimum of q found: {glob_min[0]} at {glob_min[1]}")
print("\n".join(lines)); open("../outputs/summary.txt", "w").write("\n".join(lines) + "\n")

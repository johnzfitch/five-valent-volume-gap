"""Machine check of the rfloat_sqrt outputs, as fc/scripts/summarize.py: every tuple of every level appears exactly
once, every nonempty tuple is certified; and, for comparison, the per-tuple results agree with the original runs
(fc/outputs for levels <= 20, whose per-tuple files are shipped; for other levels only the level minima are compared,
against fc/outputs/summary.txt and rr/outputs/summary52.txt).
Usage (from any directory): python summarize_sqrt.py all 38   |   python summarize_sqrt.py j52 39 70"""
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_sqrt import tuples
OUT = os.path.join(HERE, '..', 'outputs')
CERT = os.path.join(HERE, '..', '..')
mode = sys.argv[1]
lo, hi = (1, int(sys.argv[2])) if mode == 'all' else (int(sys.argv[2]), int(sys.argv[3]))
orig_min = {}
src = os.path.join(CERT, 'fc', 'outputs', 'summary.txt') if mode == 'all' else os.path.join(CERT, 'rr', 'outputs', 'summary52.txt')
for line in open(src):
    m = re.match(r"2j_max=\s*(\d+):.*min q=([0-9.]+)", line)
    if m: orig_min[int(m.group(1))] = float(m.group(2))
lines = []; allok = True; gmin = (9e9, None); total = 0; perrec_cmp = 0; perrec_diff = 0
for M in range(lo, hi + 1):
    fn = os.path.join(OUT, f"{mode}_M{M:03d}.jsonl")
    if not os.path.exists(fn):
        lines.append(f"2j_max={M}: MISSING"); allok = False; continue
    recs = [json.loads(l) for l in open(fn)]
    complete = sorted(tuple(r['t']) for r in recs) == sorted(tuples(M, mode))
    bad = [r['t'] for r in recs if not r.get('skip') and not r['cert']]
    ne = sum(1 for r in recs if not r.get('skip')); total += ne
    qs = [(r['qmin'], tuple(r['t'])) for r in recs if not r.get('skip') and r.get('qmin') is not None]
    qm = min(qs) if qs else (None, None)
    if qm[0] is not None and qm[0] < gmin[0]: gmin = qm
    om = orig_min.get(M)
    agree = (qm[0] is None and om is None) or (qm[0] is not None and om is not None and abs(qm[0] - om) <= (1e-9 * max(1, om) if mode == 'all' else 6e-7))  # summary52 prints 6 decimals
    ofn = os.path.join(CERT, 'fc', 'outputs', f"cert_M{M:03d}.jsonl")
    if mode == 'all' and os.path.exists(ofn):
        old = {tuple(r['t']): r for r in map(json.loads, open(ofn))}
        for r in recs:
            o = old.get(tuple(r['t'])); perrec_cmp += 1
            if o is None or o.get('skip') != r.get('skip') or o.get('cert') != r.get('cert') or o.get('kerT') != r.get('kerT'):
                perrec_diff += 1
    ok = complete and not bad and agree
    allok &= ok
    lines.append(f"2j_max={M:3d}: nonempty {ne:7d} complete={complete} uncertified={len(bad)} min q={qm[0]} at {qm[1]} "
                 f"(original {om}; agree={agree})")
lines.append(f"levels {lo}..{hi} ({mode}): {total} nonempty tuples; all complete, certified and matching the original minima: {allok}")
if perrec_cmp:
    lines.append(f"per-tuple comparison with the original fc records (levels with shipped files): {perrec_cmp} records, "
                 f"{perrec_diff} differing in certified/kernel/skip")
lines.append(f"minimum q over the range: {gmin[0]} at {gmin[1]}")
print("\n".join(lines))
open(os.path.join(OUT, f"summary_{mode}.txt"), "w").write("\n".join(lines) + "\n")

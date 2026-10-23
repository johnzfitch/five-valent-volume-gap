"""Digest of the labelled surveys (orbit 1: M<=16, orbit 2: M<=34, orbit 3: M<=22) -> outputs/numerics_digest.txt"""
import json, os, sys
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
def load(*names):
    rows = []
    for n in names:
        rows += [json.loads(l) for l in open(os.path.join(here, n))]
    return sorted(rows, key=lambda r: r["m"])
out = []
for orb, files in [(1, ["survey_orbit1_M16.jsonl"]), (2, ["survey_orbit2_M26.jsonl", "survey_orbit2_M27to34.jsonl"]),
                   (3, ["survey_orbit3_M22.jsonl"])]:
    rows = load(*files)
    out.append(f"orbit {orb}: labelled q_min at exact 2j_max = m (min over all labelled spins), argmin (j1 = distinguished edge)")
    amb = 0
    for r in rows:
        if r["q_min"] is None: continue
        out.append(f"   m={r['m']:2d} tuples={r['ntuples']:7d} q_min={r['q_min']:.9e} argmin={tuple(r['argmin'])} ({r['seconds']}s)")
        amb = max(amb, r["max_zero_over_q"])
    out.append(f"   largest 'zero' singular value / q_min over all records: {amb:.1e}")
    if orb == 2:
        best = {}
        for r in rows:
            for j1, (q, J) in r["per_j1"].items():
                if int(j1) not in best or q < best[int(j1)][0]: best[int(j1)] = (q, tuple(J))
        out.append("   orbit 2, min over all spins (2j_max<=34) at fixed 2j1: " +
                   "; ".join(f"2j1={k}: {v[0]:.4f}" for k, v in sorted(best.items())))
open(os.path.join(here, "numerics_digest.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

"""Digest of the closed survey (../volume_gap_higher_valence_20261021/survey_M22, sorted spins, 2j_max<=22)
regrouped by permutation orbit (outputs/orbits.json). Read-only use of the closed folder.
For each orbit and 2j_max: min over sigma in the orbit of q_min at exactly that 2j_max (= labelled q_min over
all spin assignments whose largest spin is j_max). Also prints the per-sigma sequences of the sorted classes
with +-4 that the closed survey called 'constant'."""
import json, sys
surv, orb = sys.argv[1], sys.argv[2]
O = json.load(open(orb))
owner = {}
for o in O["orbits"]:
    for s in o["sigma"]:
        owner[tuple(s)] = o["orbit"]
rows = [json.loads(l) for l in open(surv)]
by = {}
for r in rows:
    s = tuple(r["sigma"]); k = owner[s]
    if r["q_min"] is None: continue
    key = (k, r["2jmax"])
    if key not in by or r["q_min"] < by[key][0]:
        by[key] = (r["q_min"], s, tuple(r["argmin"]))
for k in (1, 2, 3):
    print(f"orbit {k}: labelled q_min at 2j_max = M (min over sigma in orbit, sorted tuples)")
    for M in range(1, 23):
        if (k, M) in by:
            q, s, a = by[(k, M)]
            print(f"   M={M:2d}  q={q:.6e}  sigma={s}  spins2j={a}")
seq = {}
for r in rows:
    seq.setdefault(tuple(r["sigma"]), {})[r["2jmax"]] = r["q_min"]
const4 = [(0,2,-4,-4),(0,2,4,4),(2,-4,0,4),(2,0,-4,-4),(2,0,4,4),(2,2,-4,-4),(2,2,4,4),(2,4,0,-4),(4,-2,-2,4),(4,0,-4,-2),(4,0,-4,2),(4,0,-2,4),(4,0,2,4),(4,2,2,4)]
print("\nsorted 'constant' classes containing +-4: q_min at 2j_max = 8, 12, 16, 20, 22 (sorted spins, exact 2j_max)")
for s in const4:
    d = seq[s]
    print("  ", s, " ".join(f"{d.get(M):.4e}" if d.get(M) else "None" for M in (8, 12, 16, 20, 22)))

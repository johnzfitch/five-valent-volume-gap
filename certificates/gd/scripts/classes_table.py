"""Table of all 85 nonzero valence-5 sigma classes (BR sorted labelling, up to overall sign):
orbit, +-4, structure in sorted positions, trend in the closed survey (2j_max<=22), sorted-convention running
minimum at 2j_max<=22 and <=34 (orbit 3, survey_sorted), and status of the dichotomy statement in the sorted
convention.  Inputs: outputs/orbits.json, outputs/survey_sorted_orbit3_M34.jsonl, closed survey summary."""
import json, itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orbits import act, sigma
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
O = json.load(open(os.path.join(here, "outputs", "orbits.json")))
allc = [tuple(int(x) for x in k.strip("()").split(",")) for k in O["witnesses"]]
owner = {tuple(s): o["orbit"] for o in O["orbits"] for s in o["sigma"]}
closed = [json.loads(l) for l in open(os.path.join(here, "..", "hv", "survey_M22", "summary.jsonl"))]
trend = {}
for line in open(os.path.join(here, "..", "hv", "survey_summary.txt")):
    for lab in ("decreasing", "constant", "increasing"):
        if line.strip().startswith(lab + ":"):
            for tok in line.split(":", 1)[1].split(") ("):
                t = tuple(int(x) for x in tok.strip(" ()\n").split(","))
                trend[t] = lab
srt = [json.loads(l) for l in open(os.path.join(here, "outputs", "survey_sorted_orbit3_M34.jsonl"))]
runmin = {}
for r in sorted(srt, key=lambda r: r["m"]):
    p = (r["pattern"][0], tuple(r["pattern"][1]))
    cur = runmin.setdefault(p, {})
    prev = cur.get(r["m"] - 1, 1e99)
    cur[r["m"]] = min(prev, r["q_min"] if r["q_min"] else 1e99)
def structure(e, orb):
    found = set()
    for p in itertools.permutations(range(5)):
        t = sigma(act(e, p))
        if t[3] == 0 and t[1] == t[2] and t[0] != 0 and abs(t[0]) >= abs(t[1]):
            if orb == 1:
                found.add(("triple", tuple(sorted((p[0] + 1, p[1] + 1, p[2] + 1)))))
            elif orb == 2:
                found.add((p[0] + 1, tuple(sorted([tuple(sorted((p[1] + 1, p[2] + 1))), tuple(sorted((p[3] + 1, p[4] + 1)))]))))
            else:
                found.add((p[0] + 1, tuple(sorted((p[1] + 1, p[2] + 1)))))
    return sorted(found)
rows = []
seen = set()
for e in sorted(allc):
    s = sigma(e)
    if not any(s): continue
    key = max(s, tuple(-x for x in s))
    if key in seen: continue
    seen.add(key)
    orb = owner[s]
    st = structure(e, orb)
    if orb == 1:
        status = "gap >= 2*sqrt(3), sharp (proved: Thm B1)"
        extra = ""
    elif orb == 2:
        k = st[0][0]
        status = "gap conjectured; proved when the distinguished edge has spin 1/2 (Thm B2)" + (" [possible in sorted order: k=1]" if k == 1 else " [sorted order forces j_k >= others before it]")
        extra = ""
    else:
        k, ap = st[0]
        rm = runmin[(k, ap)]
        extra = f"runmin22={rm[22]:.2e} runmin34={rm[34]:.2e}"
        if k == 1 and 2 in ap:
            status = "inf=0 proved (Thm A, family (1/2, j-1/2, j, j, j))"
        else:
            status = "inf=0 numerical only (decreasing to 2j_max=34)"
    rows.append((orb, key, "yes" if 4 in map(abs, key) else "no", st, trend.get(key, "?"), extra, status))
rows.sort()
with open(sys.argv[1], "w") as f:
    for r in rows:
        line = f"orbit {r[0]} sigma={r[1]} +-4:{r[2]} structure(sorted positions)={r[3]} closed-survey trend={r[4]} {r[5]} | {r[6]}"
        print(line); f.write(line + "\n")
    from collections import Counter
    c = Counter((r[0], r[6].split(" (")[0].split(";")[0]) for r in rows)
    f.write("\nsummary: " + str(dict(c)) + "\n"); print("\nsummary:", dict(c))

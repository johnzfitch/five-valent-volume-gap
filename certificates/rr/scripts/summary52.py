"""Machine check of the j1 = 5/2 recheck: per level 39..70, every tuple present once, every nonempty tuple certified,
nonempty counts equal to the per-level counts of dichotomy_fixed_j1_20261021/outputs/stagecert_s52_39_70.txt,
plus an exact rational LDL^T inertia spot check (certcore.inertia_ldl) on a random sample of tuples."""
import json, re, random
from fractions import Fraction as Fr
from run52 import tuples
from certcore import matrices, rank_exact, Cmat, inertia_ldl
ref = {}
for line in open("../../fj/outputs/stagecert_s52_39_70.txt"):
    m = re.match(r"2j1=5 2j_max=(\d+): (\d+) tuples", line)
    if m: ref[int(m.group(1))] = int(m.group(2))
lines = []; tot = 0; allok = True; gmin = (9e9, None); recs_all = []
for M in range(39, 71):
    recs = [json.loads(l) for l in open(f"../outputs/cert52_M{M:03d}.jsonl")]
    complete = sorted(tuple(r['t']) for r in recs) == sorted(tuples(M))
    ne = [r for r in recs if not r.get('skip')]
    bad = [r for r in ne if not r['cert']]
    qs = [(r['qmin'], tuple(r['t'])) for r in ne if r.get('qmin') is not None]
    qm = min(qs); gmin = min(gmin, qm)
    ok = complete and not bad and len(ne) == ref.get(M)
    allok &= ok; tot += len(ne); recs_all += ne
    lines.append(f"2j_max={M}: nonempty {len(ne)} (fixed_j1 file: {ref.get(M)}), complete={complete}, uncertified={len(bad)}, min q={qm[0]:.6f} at {qm[1]}")
lines.append(f"TOTAL levels 39..70: {tot} nonempty tuples (fixed_j1: 430,481); all complete, certified and count-matched: {allok}")
lines.append(f"minimum q over the range: {gmin[0]:.6f} at {gmin[1]}")
random.seed(20261021); smp = random.sample(recs_all, 400)
eps = Fr(45579607, 80000000) ** 2; agree = 0; zp = 0
for r in smp:
    D = matrices(tuple(r['t'])); z = D['nO'] - rank_exact(D)
    inn = inertia_ldl(D['nO'], Cmat(D), D['thO'], eps)
    if inn is None: zp += 1; continue
    agree += int((inn[0] == z and inn[1] == 0) == r['cert'])
lines.append(f"exact rational LDL spot check at lam0, 400 random tuples: agree {agree}, zero pivots {zp}")
print("\n".join(lines)); open("../outputs/summary52.txt", "w").write("\n".join(lines) + "\n")

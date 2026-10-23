"""Census: every j1=3/2 tuple with 2j <= M (dim Inv >= 2) is assigned exactly one regime; chain types,
imbalances in {0,1}, and pointwise ratio hypotheses hold on the actual links (exact rationals).
Usage: python z32_regime_census.py M out.txt"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, tuples32
from z32_regime_plan import regime, EXPECTED
from z32_regimes import ends
from z32_validate_bounds import hyp_ok
M, out = int(sys.argv[1]), sys.argv[2]
cnt = collections.Counter(); bad = []
for tj in tuples32(M):
    name, P5 = regime(tj)
    z = Z32(P5)
    if sum(len(x) for x in (z.R0, z.C0, z.R1, z.C1)) < 2:
        cnt['dim<2'] += 1; continue
    t0, t1 = ends(z.R0, z.C0), ends(z.R1, z.C1)
    e = EXPECTED[name]
    ok = (not z.C0 or t0 == e[0]) and (not z.C1 or t1 == e[1]) and min(z.imb) >= 0 \
        and hyp_ok(t0, z.R0, z.D0) and hyp_ok(t1, z.R1, z.D1)
    cnt[name] += 1
    if not ok:
        bad.append((tj, name, t0, t1, z.imb))
L = [f'regime census, all j1=3/2 multiset tuples 2j<={M}: ' + ', '.join(f'{k}: {v}' for k, v in sorted(cnt.items())),
     f'type/imbalance/ratio-hypothesis failures: {len(bad)}'] + ['  ' + str(b) for b in bad[:10]]
print('\n'.join(L)); open(out, 'w').write('\n'.join(L) + '\n')

"""Exact check: does every tuple with 2j1 = S2 (default 5) have a monotone pairing?  Multisets 2j<=M.
Usage: python oc_mono_s52.py M S2 out.txt"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_lemma_s32 import imb_pattern, monotone
from oc_core import sites, pairings
M, S2, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
n = 0; bad = []
for a in range(1, M + 1):
    for b in range(a, M + 1):
        for c in range(b, M + 1):
            for d in range(c, M + 1):
                tj = (S2, a, b, c, d)
                if sum(tj) % 2 or len(sites(tj)) < 2:
                    continue
                n += 1
                if not any(monotone(imb_pattern(P)) for P in pairings(tj)):
                    bad.append(tj)
L = [f"2j1={S2}, multiset tuples 2j<={M}: {n}; without a monotone pairing: {len(bad)} {bad[:10]}"]
open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

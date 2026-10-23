"""Which shifted windows of a Stark matrix actually occur in the ladder case (spin 1/2 on leg 2), j1 half-integer >= 3/2.
Uses copies of the closing folder's grid code (closing_copy/, read-only copies)."""
import sys, os, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closing_copy'))
from explore_T0 import T0
from ladder_windows import stark_full
from explore_T0 import nzmin
noncontig = 0; blockmin = (9e9, None)
M = int(sys.argv[1])
occ = collections.Counter(); ex = {}
for s2 in range(3, M + 1, 2):
    for j3 in range(1, M + 1):
        for j4 in range(1, M + 1):
            for j5 in range(j4, M + 1):
                tj = (s2, 1, j3, j4, j5)
                if sum(tj) % 2: continue
                S, W0 = T0(tj)
                if len(S) < 2: continue
                cs, H = stark_full(j4 + j5 + 2, abs(j4 - j5)); N = len(cs)
                pos = {c: i for i, c in enumerate(cs)}
                E = sorted(pos[c] for (b, c) in S if ((b + c) // 2) % 2 == 0)
                O = sorted(pos[c] for (b, c) in S if ((b + c) // 2) % 2 == 1)
                if not E or not O: continue
                def nb(X): return {y for x in X for y in (x-1, x, x+1) if 0 <= y < N}
                if nb(O) <= set(E) or nb(E) <= set(O): continue   # nested: covered
                # shifted: express as rows x cols window
                r0, r1, c0, c1 = min(E), max(E), min(O), max(O)
                if len(E) != r1 - r0 + 1 or len(O) != c1 - c0 + 1: noncontig += 1
                l0, _ = nzmin(W0, S)
                if l0 is not None and l0 < blockmin[0]: blockmin = (l0, tj)
                if r0 == c0 + 1 and r1 == c1 + 1: key = ('up', c0, N - 2 - c1)
                elif c0 == r0 + 1 and c1 == r1 + 1: key = ('down', r0, N - 2 - r1)
                else: key = ('other', r0, r1, c0, c1, N)
                k2 = (key[0], 'a=%d' % key[1], 'top gap=%s' % (key[2] if len(key) == 3 else '?'), 'N%%2=%d' % (N % 2))
                occ[k2] += 1; ex.setdefault(k2, (tj, N, abs(j4 - j5) / 2))
for k, v in sorted(occ.items(), key=str): print(k, v, ex[k])
print('shifted ladder blocks (j2=1/2, half-integer j1>=3/2, 2j<=%d): %d; non-contiguous row/col sets: %d' % (M, sum(occ.values()), noncontig))
print('smallest nonzero singular value of the T0 bipartite block over them: %.12f at %s (SW constant sqrt((15-sqrt97)/6) = 0.926565539885)' % blockmin)

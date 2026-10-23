"""Data for Conjecture K: for half-integer s, is dim ker T = min over pairings of sum_k |imb_k|, and does
some pairing attain it?  (float rank with the survey threshold).  All multiset tuples 2j<=M, 2s<=M.
Usage: python oc_kernel_conj.py M out.txt"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_reduce import zig_paths, ker_dim, monotone_ok
from oc_fast import region, pairings, is_open
M, out = int(sys.argv[1]), sys.argv[2]
n = 0; eq_min = 0; nomono = 0; nomono_eq = 0; nomono_open = 0; bad = []
for s2 in range(1, M + 1, 2):
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    tj = (s2, a, b, c, d)
                    if sum(tj) % 2 or len(region(tj)[0]) < 2:
                        continue
                    n += 1
                    Ss = []; mono = False; kd = None
                    for P in pairings(tj):
                        B, rk, ck, rows, cols, Z = zig_paths(P)
                        imb = [len(R) - len(C) for R, C in Z]
                        Ss.append(sum(abs(x) for x in imb)); mono |= monotone_ok(imb)
                        if kd is None:
                            kd = ker_dim(B)
                    ok = (kd == min(Ss)); eq_min += ok
                    if not mono:
                        nomono += 1; nomono_eq += ok; nomono_open += is_open(tj)
                    if not ok and len(bad) < 8:
                        bad.append((tj, kd, Ss))
L = [f"half-integer s, multiset tuples 2s,2j<={M}: {n}",
     f"  dim ker T == min_P sum|imb^P|: {eq_min} ({n-eq_min} exceptions) {bad}",
     f"  tuples with no monotone pairing: {nomono} (open: {nomono_open}); of these, dim ker T == min_P sum|imb^P|: {nomono_eq}"]
open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

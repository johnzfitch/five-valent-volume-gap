"""Exploration: monotonicity of qmin along s -> s+1 with j5 -> j5+1 (fixed j2,j3,j4, M=s-j5).
Usage: python explore_mono.py KMAX SMAX
Prints violations of q(s+1) >= q(s) and of q(s+1)/(s+2) >= q(s)/(s+1)."""
import sys, itertools
from t1core import qmin, valid

K = int(sys.argv[1]); SM = int(sys.argv[2])
viol = []; viol2 = []; npairs = 0
for a2, a3, a4 in itertools.combinations_with_replacement(range(1, K + 1), 3):
    for Mx in range(-(a2 + a3 + a4), a2 + a3 + a4 + 1):      # doubled M
        prev = None
        for S in range(1, SM + 1):
            a5 = S - Mx
            if a5 < 1 or not valid(S, a2, a3, a4, a5):
                continue
            q, dk, n = qmin(S, a2, a3, a4, a5)
            if q is not None and prev is not None and prev[0] is not None:
                npairs += 1
                if q < prev[0] * (1 - 1e-9):
                    viol.append((prev[1], (S, a2, a3, a4, a5), prev[0], q))
                if q / (S / 2 + 1) < prev[0] / (prev[1][0] / 2 + 1) * (1 - 1e-9):
                    viol2.append((prev[1], (S, a2, a3, a4, a5), prev[0], q))
            prev = (q, (S, a2, a3, a4, a5))
print('pairs', npairs, 'violations of q monotone', len(viol), 'of q/(s+1) monotone', len(viol2))
for v in viol[:40]:
    print('Q', v)
for v in viol2[:20]:
    print('R', v)

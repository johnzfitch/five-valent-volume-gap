"""Test of the literal Conjecture Z_{3/2} of [OR] (section 5.1):
  'there is an admissible pairing and deletion such that 8 sigma_min(D0), 8 sigma_min(D1) >= 7.7 and the coupling
   norm is <= 1.9',
where B' = [[D0,0],[N,D1]] is the square reduction ([OR] Theorem 3: delete one row from each zigzag with surplus rows,
one column from each with surplus columns, in ANY position), and the coupling norm is ||N D0^-1|| or the variant
||D1^-1 N|| ([OR] section 5.1).  For a tuple we enumerate every ordered pairing, keep the admissible ones
(S_P = sum|imb| minimal among monotone pairings, as in [OR]), every deletion position, and report the smallest
coupling norm (either variant) over configurations whose chains satisfy 8 sigma_min >= 7.7, and over all.
Usage: python z32_Z32_refute.py out.txt
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, pairings6, monotone


def configs(z):
    for keep in z.deletions():
        yield keep


def analyse(tj):
    zs = [Z32(P) for P in pairings6(tj)]
    S = [abs(z.imb[0]) + abs(z.imb[1]) for z in zs]
    Smin = min(S[k] for k in range(6) if monotone(zs[k].imb))
    best_ok = np.inf; best_any = np.inf; info = None
    for z, s in zip(zs, S):
        if s != Smin:
            continue
        for keep in configs(z):
            D0, N, D1 = z.mats(*keep)
            if D0.size == 0 or D1.size == 0 or N.size == 0:
                continue
            s0 = np.linalg.svd(D0, compute_uv=False).min(); s1 = np.linalg.svd(D1, compute_uv=False).min()
            E0 = np.linalg.inv(D0); E1 = np.linalg.inv(D1)
            c = min(np.linalg.norm(N @ E0, 2), np.linalg.norm(E1 @ N, 2))
            best_any = min(best_any, c)
            if 8 * min(s0, s1) >= 7.7 and c < best_ok:
                best_ok = c; info = (z.tj, z.imb, [len(x) for x in keep], 8 * s0, 8 * s1)
    return best_ok, best_any, info


if __name__ == '__main__':
    out = sys.argv[1]
    L = []
    for a in [7, 15, 31, 63, 127, 255]:
        tj = (3, a, a, a, a + 1)
        ok, anyc, info = analyse(tj)
        L.append(f'(3/2; {a}/2,{a}/2,{a}/2,{a+1}/2): min coupling over admissible configs with chains>=7.7/8: {ok:.4f}; '
                 f'over all admissible configs: {anyc:.4f}; at {info}')
        print(L[-1], flush=True)
        open(out, 'w').write('\n'.join(L) + '\n')

"""j1 = 3/2: Frobenius-norm version of the two-chain bound (the version an explicit B2-style proof would bound):
   8 / min( max(F0,F1) + F1*C1 , max(F0,F1) + C2*F0 ),
   F_k = ||D_k^-1||_F,  C1 = ||N D0^-1||_F,  C2 = ||D1^-1 N||_F,
maximised over admissible pairings (S_P = S of a monotone pairing) and uniform end choices.
Usage: python oc_s32_frob.py M out.txt
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_s32_plan import blocks
from oc_reduce import monotone_ok
from oc_fast import region, pairings, smin_nz

if __name__ == '__main__':
    M, out = int(sys.argv[1]), sys.argv[2]
    t0 = time.time(); rows = []
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    tj = (3, a, b, c, d)
                    if sum(tj) % 2 or len(region(tj)[0]) < 2:
                        continue
                    infos = []
                    for P in pairings(tj):
                        B, kept, imb = blocks(P, 'low')
                        infos.append((P, imb, sum(map(abs, imb)), monotone_ok(imb)))
                    S0 = [S for P, imb, S, m in infos if m][0]
                    best = None
                    for P, imb, S, m in infos:
                        if S != S0:
                            continue
                        for end in ('low', 'high'):
                            B, kept, imb = blocks(P, end)
                            (R0, C0), (R1, C1) = kept
                            D0 = B[np.ix_(R0, C0)]; D1 = B[np.ix_(R1, C1)]; N = B[np.ix_(R1, C0)]
                            f0 = np.linalg.norm(np.linalg.inv(D0), 'fro') if D0.size else 0
                            f1 = np.linalg.norm(np.linalg.inv(D1), 'fro') if D1.size else 0
                            if D0.size and D1.size and N.size:
                                c1 = np.linalg.norm(N @ np.linalg.inv(D0), 'fro'); c2 = np.linalg.norm(np.linalg.inv(D1) @ N, 'fro')
                                inv = min(max(f0, f1) + f1 * c1, max(f0, f1) + c2 * f0)
                            else:
                                inv = max(f0, f1)
                            v = 8 / inv
                            if best is None or v > best[0]:
                                best = (v, P, end)
                    q = 8 * smin_nz(blocks(tj, 'low')[0])[0]
                    rows.append((best[0], q, tj, best[1], best[2]))
    rows.sort()
    L = [f"j1=3/2 Frobenius two-chain bound, all tuples 2j<={M}: {len(rows)} tuples, {time.time()-t0:.1f}s",
         "smallest (8*bound, q, tuple, pairing, end):"]
    L += [f"  {r[0]:.4f} {r[1]:.4f} {r[2]} {r[3]} {r[4]}" for r in rows[:10]]
    L.append(f"min ratio bound/q = {min(r[0]/r[1] for r in rows):.4f}; violations bound > q: {sum(1 for r in rows if r[0] > r[1]*(1+1e-9))}")
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

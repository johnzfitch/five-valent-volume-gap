"""Brute-force minima of q = 8|eig T| (smallest nonzero) with the independent Racah operator rb_core.

Only the elementary symmetries are used: j2<->j3, j4<->j5 (diagonal gauge (-1)^b / (-1)^c) and exchange of
the two pairs (X<->D).  The full S4 symmetry of [DC] Prop. 1(d) is NOT assumed; it is tested separately on
random tuples (all 24 permutations of legs 2-5).
Usage: python rb_minima.py MAX2J 2J1 [2J1 ...] > ../outputs/rb_minima_out.txt
"""
import sys, os, itertools, time, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C


def tuples(s2, M):
    out = []
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(1, M + 1):
                for d in range(c, M + 1):
                    if (a, b) > (c, d):
                        continue
                    if (s2 + a + b + c + d) % 2:
                        continue
                    out.append((s2, a, b, c, d))
    return out


def work(tj):
    if C.dim_inv(tj) < 2:
        return tj, None
    q, k = C.qmin(tj)
    return tj, q


def w2(tj):
    a = tj[1:]
    return all(x % 2 == 0 for x in a) or all(x % 2 == 1 and x >= 3 for x in a)


if __name__ == '__main__':
    M = int(sys.argv[1]); s2s = [int(x) for x in sys.argv[2:]]
    t0 = time.time()
    with Pool(8) as pool:
        for s2 in s2s:
            T = tuples(s2, M)
            res = [r for r in pool.imap_unordered(work, T, chunksize=50) if r[1] is not None]
            res.sort(key=lambda r: r[1])
            mn = res[0]
            print(f"2j1={s2}: {len(res)} tuples with dim Inv>=2 and nonzero spectrum, legs 2j<={M}; "
                  f"min q = {mn[1]:.10f} at {mn[0]}  ({time.time()-t0:.0f}s)", flush=True)
            print("   five smallest:", [(r[0], round(r[1], 6)) for r in res[:5]])
            # per-level minima (2j_max of legs)
            lev = {}
            for tj, q in res:
                L = max(tj[1:])
                if L not in lev or q < lev[L][0]:
                    lev[L] = (q, tj)
            print("   per-level min (2j_max of legs 2-5):", ", ".join(f"{L}:{lev[L][0]:.4f}" for L in sorted(lev)))
            if s2 % 2 == 0:
                rw = [r for r in res if w2(r[0])]; rn = [r for r in res if not w2(r[0])]
                if rw: print(f"   W2-class tuples: {len(rw)}, min q = {rw[0][1]:.6f} at {rw[0][0]}")
                if rn: print(f"   non-W2 tuples:   {len(rn)}, min q = {rn[0][1]:.6f} at {rn[0][0]}  "
                             f"(DC Cor. 10 bound 4*sqrt(2s(s+1)/pi) = {4*np.sqrt(s2/2*(s2/2+1)*2/np.pi):.4f})")
        # S4 test
        random.seed(1)
        worst = 0.0; nt = 0
        for _ in range(60):
            s2 = random.choice(s2s)
            while True:
                legs = [random.randint(1, 9) for _ in range(4)]
                if (s2 + sum(legs)) % 2 == 0 and C.dim_inv((s2, *legs)) >= 2:
                    break
            base = None
            for p in itertools.permutations(legs):
                v, k = C.q_values((s2, *p))
                v = np.array(v)
                if base is None:
                    base = (v, k)
                else:
                    if len(v) != len(base[0]) or k != base[1]:
                        worst = float('inf')
                    else:
                        worst = max(worst, np.abs(v - base[0]).max() / max(1, v.max()))
            nt += 1
        print(f"S4 symmetry test (DC Prop 1(d)): {nt} random tuples x 24 permutations, max rel diff of |spec| {worst:.2e}")
    print(f"total {time.time()-t0:.0f}s")

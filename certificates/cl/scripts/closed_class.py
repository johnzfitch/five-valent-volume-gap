"""General (2D) 'closed colour class' criterion for T0 = P_R (H23 (+) H45) P_R.
A colour class C of R is closed if every rectangle-neighbour of every site of C lies in R.
If C is closed, then for psi supported on C:  T0 psi = K psi  (K = Kronecker sum on the rectangle),
so the bipartite block restricted to C equals K restricted to C.
Records, per category (closed?, N23+N45 parity), the minimum of lam(T0).
Usage: python closed_class.py M
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import sites
from explore_T0 import T0, nzmin


def rect_sites(tj):
    s, j2, j3, j4, j5 = tj
    out = set()
    for b in range(abs(j2 - j3), j2 + j3 + 1, 2):
        for c in range(abs(j4 - j5), j4 + j5 + 1, 2):
            out.add((b, c))
    return out


def closed_classes(tj):
    S = sites(tj); Sset = set(S); rect = rect_sites(tj)
    res = []
    for colour in (0, 1):
        C = [p for p in S if ((p[0] + p[1]) // 2) % 2 == colour]
        ok = True
        for (b, c) in C:
            for q in ((b + 2, c), (b - 2, c), (b, c + 2), (b, c - 2)):
                if q in rect and q not in Sset:
                    ok = False; break
            if not ok: break
        res.append(ok and len(C) > 0)
    return res


if __name__ == '__main__':
    M = int(sys.argv[1])
    stats = {}
    for s in range(1, M + 1):
        for j2 in range(1, M + 1):
            for j3 in range(j2, M + 1):
                for j4 in range(1, M + 1):
                    for j5 in range(j4, M + 1):
                        if (j2, j3) > (j4, j5): continue
                        tj = (s, j2, j3, j4, j5)
                        if sum(tj) % 2: continue
                        S, W0 = T0(tj)
                        if len(S) < 2: continue
                        l0, _ = nzmin(W0, S)
                        if l0 is None: continue
                        cl = closed_classes(tj)
                        N23 = min(j2, j3) + 1; N45 = min(j4, j5) + 1   # doubled min -> 2min+1
                        key = ('closed' if any(cl) else 'open', 'Nsum odd' if (N23 + N45) % 2 else 'Nsum even',
                               's int' if s % 2 == 0 else 's half')
                        st = stats.setdefault(key, [0, 9e9, None])
                        st[0] += 1
                        if l0 < st[1]: st[1], st[2] = l0, tj
    for k, v in sorted(stats.items()):
        print(k, 'count', v[0], 'min lam0 %.6f' % v[1], v[2])

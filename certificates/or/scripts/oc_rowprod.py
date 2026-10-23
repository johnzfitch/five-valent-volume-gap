"""Exact check of the row-product identity (Proposition 6), all half-integer s.

Plaquette: rows r1=(b,c) [diagonal d] and r2=(b+1,c-1) [diagonal d+2], columns c_up=(b+1,c), c_lo=(b,c-1) [diagonal d+1].
Row products:  P(r1) = x(r1--c_up) * y(c_lo--r1),  P(r2) = y(r2--c_up) * x(c_lo--r2).
Claim: P(r2)/P(r1) = F(d+3/2)/F(d+1/2) exactly (independent of h, G), and if both columns lie in the region then
both rows do.  Consequence: for adjacent zigzags, N_k^T N_k - kappa_k D_{k-1}^T D_{k-1} is diagonal,
kappa_k = F(d_k - 1/2)/F(d_k - 3/2)  (here d = d_k - 2 is the row diagonal of zigzag k-1).
Checked on squares in exact rationals.  Also: column-pair version (columns on d, d+2 sharing rows on d+1):
ratio of the two row products = G(u+1)/G(u).  Usage: python oc_rowprod.py M out.txt
"""
import sys, os, itertools
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_core import links, sites, F, G

if __name__ == '__main__':
    M, out = int(sys.argv[1]), sys.argv[2]
    h = Fr(1, 2)
    n = 0; bad = 0; cut = 0; n2 = 0; bad2 = 0
    for tj in itertools.product(range(1, M + 1), repeat=5):
        if tj[0] % 2 == 0 or sum(tj) % 2:
            continue
        S, L = links(tj, 'T')
        if len(S) < 4:
            continue
        s = Fr(tj[0], 2)
        idx = {(Fr(B, 2), Fr(C, 2)): i for i, (B, C) in enumerate(S)}
        W = {frozenset(k): v for k, v in L.items()}

        def w2(p, q):
            return W.get(frozenset((idx[p], idx[q])))
        for (b, c), i in idx.items():
            d = b - c
            if (d + s) % 2 != 0 or d + 2 > s - 1:
                continue          # r1 must be a row (d+s even) with a row diagonal d+2 inside the strip
            r1 = (b, c); r2 = (b + 1, c - 1); cu = (b + 1, c); cl = (b, c - 1)
            if cu in idx and cl in idx:
                if r2 not in idx:
                    cut += 1; continue
                n += 1
                P1sq = w2(r1, cu) * w2(cl, r1); P2sq = w2(r2, cu) * w2(cl, r2)
                if P2sq * F(s, d + h) ** 2 != P1sq * F(s, d + 3 * h) ** 2:
                    bad += 1
        # column-pair version: columns c1=(b-1,c), c2=(b,c-1) on diagonals d-1? use rows (b,c) and (b-1,c-1)
        for (b, c), i in idx.items():
            d = b - c
            if (d + s) % 2 != 1:
                continue          # c1 column
            c1 = (b, c); c2 = (b + 1, c - 1); ru = (b + 1, c); rl = (b, c - 1)
            if c2 in idx and ru in idx and rl in idx:
                n2 += 1
                u = b + c
                Pu = w2(ru, c1) * w2(ru, c2); Pl = w2(rl, c1) * w2(rl, c2)
                if Pu * G(s, u) ** 2 != Pl * G(s, u + 1) ** 2:
                    bad2 += 1
    L = [f"row-product identity, all labelled tuples 2s odd, 2j_i<={M}: {n} plaquettes, {bad} failures; "
         f"plaquettes (both row diagonals inside the strip) with both columns but a missing row: {cut}",
         f"column-pair identity (ratio G(u+1)/G(u)): {n2} plaquettes, {bad2} failures"]
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

"""Exact check of the layer Gram identity (Lemma U) on all valid tuples with doubled spins <= MAX.
  Gamma_{u+1}^T Gamma_{u+1} - (G(u+1)/G(u)) Gamma_u Gamma_u^T = diag(Lambda^u),  u >= s+1,
with entries of Gram matrices exact (products of square roots squared => rational check of
off-diagonal products via squares).  Off-diagonal entries are products of two links; we compare
squares exactly (signs are all + in the nonnegative gauge)."""
import sys, itertools
from fractions import Fraction as Fr
from t1core import grid, valid
from layers import Lambda_closed

MX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SMX = int(sys.argv[2]) if len(sys.argv) > 2 else MX
ntup = 0; nchk = 0; bad = 0
for S in range(1, SMX + 1):
    for legs in itertools.product(range(1, MX + 1), repeat=4):
        if not valid(S, *legs):
            continue
        st, links = grid(S, *legs, exact=True)
        if len(st) < 2:
            continue
        ntup += 1
        idx = {x: i for i, x in enumerate(st)}
        w2 = {}
        for i, k, w in links:
            w2[(i, k)] = w; w2[(k, i)] = w
        s = Fr(S, 2)
        G = lambda u: (u + s + 1) * (u - s)
        up = {i: [] for i in range(len(st))}; dn = {i: [] for i in range(len(st))}
        for i, k, w in links:   # i lower, k upper
            up[i].append(k); dn[k].append(i)
        layers = {}
        for i, (b, c) in enumerate(st):
            layers.setdefault(b + c, []).append(i)
        for U, L in layers.items():
            u = Fr(U, 2)
            if u < s + 1:
                continue
            r = G(u + 1) / G(u)
            for x in L:
                for y in L:
                    upxy = sum(w2.get((z, x), 0) ** 1 * 0 for z in []) # placeholder
                    # Gram entries: (Gup)_{xy} = sum_z T_zx T_zy over z in layer u+1 ; squared-products exact:
                    zs_up = set(up[x]) & set(up[y]); zs_dn = set(dn[x]) & set(dn[y])
                    if x == y:
                        gu = sum(w2[(z, x)] for z in up[x]); gd = sum(w2[(z, x)] for z in dn[x])
                        lam = gu - r * gd
                        lc = Lambda_closed(S, *legs, *st[x])
                        nchk += 1
                        if lam != lc:
                            bad += 1
                            if bad < 10: print('diag mismatch', S, legs, st[x], lam, lc)
                    else:
                        # each shared neighbour set has at most one element; compare squares
                        pu = [w2[(z, x)] * w2[(z, y)] for z in zs_up]
                        pd = [w2[(z, x)] * w2[(z, y)] for z in zs_dn]
                        su = pu[0] if pu else Fr(0); sd = pd[0] if pd else Fr(0)
                        nchk += 1
                        if len(pu) > 1 or len(pd) > 1 or su != r * r * sd:
                            bad += 1
                            if bad < 10: print('offdiag mismatch', S, legs, st[x], st[y], su, sd, r)
print(f'tuples {ntup}, entries checked {nchk}, mismatches {bad}')

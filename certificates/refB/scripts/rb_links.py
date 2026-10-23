"""Exact check of the closed-form link moduli and of the nonnegative gauge, against rb_core (Racah).

For every tuple in range and every link of T (pairing 23|45):
  (1) |T_link|^2 from rb_core equals, exactly in rationals, the closed form used by all documents refereed here
        X (b-1,c)-(b,c):  x^2 = Q1(b) ((s+c+1)^2-b^2)(b^2-(s-c)^2) / (16 (4b^2-1)),  Q1(b) = (n1^2-b^2)(b^2-m1^2)
        D (b,c-1)-(b,c):  d^2 = Q2(c) ((s+b+1)^2-c^2)(c^2-(s-b)^2) / (16 (4c^2-1))
      and every closed-form link between two sites of the region that is nonzero is present in rb_core;
  (2) flux: there is a +-1 diagonal gauge making every nonzero entry of the bipartite block positive
      (checked exactly by BFS sign propagation over the actual signs of R).
Usage: python rb_links.py MAX2J MAX2J1 > ../outputs/rb_links_out.txt
"""
import sys, os, itertools, time
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C


def closed_links(tj):
    s2, a, b, c, d = tj
    s = Fr(s2, 2)
    n1 = Fr(a + b, 2) + 1; m1 = Fr(abs(a - b), 2); n2 = Fr(c + d, 2) + 1; m2 = Fr(abs(c - d), 2)
    S = C.sites_of(tj); idx = {p: i for i, p in enumerate(S)}
    out = {}
    for (B, Cc), i in idx.items():
        bb = Fr(B, 2); cc = Fr(Cc, 2)
        if (B - 2, Cc) in idx:
            x2 = (n1 * n1 - bb * bb) * (bb * bb - m1 * m1) * ((s + cc + 1) ** 2 - bb * bb) * (bb * bb - (s - cc) ** 2) / (16 * (4 * bb * bb - 1))
            out[(idx[(B - 2, Cc)], i)] = x2
        if (B, Cc - 2) in idx:
            d2 = (n2 * n2 - cc * cc) * (cc * cc - m2 * m2) * ((s + bb + 1) ** 2 - cc * cc) * (cc * cc - (s - bb) ** 2) / (16 * (4 * cc * cc - 1))
            out[(idx[(B, Cc - 2)], i)] = d2
    return out


def flux_free(tj):
    S, ent, col = C.colour_classes(tj)
    n = len(S)
    adj = [[] for _ in range(n)]
    for (i, j), (sg, sq) in ent.items():
        adj[i].append((j, sg)); adj[j].append((i, sg))
    eps = [0] * n
    for r in range(n):
        if eps[r]:
            continue
        eps[r] = 1; st = [r]
        while st:
            v = st.pop()
            for w, sg in adj[v]:
                want = eps[v] * sg      # require eps_v * eps_w * sg = +1
                if eps[w] == 0:
                    eps[w] = want; st.append(w)
                elif eps[w] != want:
                    return False
    return True


if __name__ == '__main__':
    M = int(sys.argv[1]); M1 = int(sys.argv[2])
    t0 = time.time()
    cnt = nl = bad_val = bad_set = bad_flux = 0
    for tj in itertools.product(range(1, M1 + 1), *[range(1, M + 1)] * 4):
        if sum(tj) % 2:
            continue
        S, ent = C.R_entries(tj)
        if len(S) < 2:
            continue
        sc = C.scale_sq(tj[0])
        cl = closed_links(tj)
        mine = {}
        for (i, j), (sg, sq) in ent.items():
            key = (min(i, j), max(i, j))
            mine[key] = sc * sq
        cln = {}
        for (i, j), v in cl.items():
            if v != 0:
                cln[(min(i, j), max(i, j))] = v
        if set(mine) != set(cln):
            bad_set += 1
            if bad_set < 5: print('LINK SET MISMATCH', tj, set(mine) ^ set(cln))
        for k in mine:
            nl += 1
            if cln.get(k) != mine[k]:
                bad_val += 1
                if bad_val < 5: print('VALUE MISMATCH', tj, k, mine[k], cln.get(k))
        if not flux_free(tj):
            bad_flux += 1
            if bad_flux < 5: print('FLUX', tj)
        cnt += 1
    print(f"exact closed-form links vs Racah: {cnt} tuples (2j1<={M1}, 2j_i<={M}), {nl} links; value mismatches {bad_val}, "
          f"link-set mismatches {bad_set}, tuples not flux-free {bad_flux}  ({time.time()-t0:.0f}s)")

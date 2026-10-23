"""Exact-rational check of the T-picture ladder (Proposition T) on real tuples, pairing (23|45).
For each link of T between diagonals d and d+1:  w^2 = l_s(d)^2 * nu^2  with l_s(d)^2 = (s-d)(s+d+1)/4 and
nu^2 = h^2 G(u_upper) / 4  (h = h1(b) for an X-link with upper label b, h2(c) for a D-link with upper c).
Flat defect on an interior diagonal (-s < d < s), site x = (b,c):
   G+(x) - G-(x) = (1/4)[G(u+1) P(b+1,c+1) - G(u) P(b,c)],  P(b,c) = h1(b)^2 - h2(c)^2,
with h(L) := 0 outside [m+1, n-1] (natural vanishing), and the off-diagonal entries of G+ and G- agree
(compared through squares).  G+ = N_d^T N_d, G- = N_{d-1} N_{d-1}^T, N = (T raising block)/l_s."""
import sys
from fractions import Fraction as Fr
from t1core import grid, pair_params, h2stark


def run(S, legs):
    a2, a3, a4, a5 = legs
    st, links = grid(S, *legs, exact=True)
    s = Fr(S, 2)
    n1, m1 = pair_params(a2, a3); n2, m2 = pair_params(a4, a5)
    pos = {i: (Fr(b, 2), Fr(c, 2)) for i, (b, c) in enumerate(st)}
    l2 = lambda d: (s - d) * (s + d + 1) / 4
    G = lambda u: (u + s + 1) * (u - s)

    def h1sq(L):
        L2 = int(2 * L)
        return h2stark(n1, m1, L2) if m1 < L2 < n1 else Fr(0)

    def h2sq(L):
        L2 = int(2 * L)
        return h2stark(n2, m2, L2) if m2 < L2 < n2 else Fr(0)
    bad = 0; nlinks = 0; nsites = 0; noff = 0
    nu2 = {}
    for i, k, w2 in links:
        (bi, ci), (bk, ck) = pos[i], pos[k]
        lo, hi = (i, k) if bi + ci < bk + ck else (k, i)
        bl, cl = pos[lo]; bh, ch = pos[hi]
        dlo, dhi = bl - cl, bh - ch
        dmin = min(dlo, dhi)
        hsq = h1sq(bh) if bh != bl else h2sq(ch)
        nlinks += 1
        if w2 != l2(dmin) * hsq * G(bh + ch) / 4:
            bad += 1
        nu2[(i, k)] = nu2[(k, i)] = w2 / l2(dmin)
    nbr = {}
    for (i, k), v in nu2.items():
        nbr.setdefault(i, []).append((k, v))
    for i, (b, c) in pos.items():
        d, u = b - c, b + c
        if not (-s < d < s):
            continue
        nsites += 1
        up = sum(v for k, v in nbr.get(i, []) if pos[k][0] - pos[k][1] == d + 1)
        dn = sum(v for k, v in nbr.get(i, []) if pos[k][0] - pos[k][1] == d - 1)
        P = lambda bb, cc: h1sq(bb) - h2sq(cc)
        if up - dn != (G(u + 1) * P(b + 1, c + 1) - G(u) * P(b, c)) / 4:
            bad += 1
        # off-diagonal: partner (b+1, c+1); shared up (b+1, c) [d+1], shared down (b, c+1) [d-1]
        j = next((k for k, (bb, cc) in pos.items() if bb == b + 1 and cc == c + 1), None)
        if j is not None:
            noff += 1
            iu = next((k for k, (bb, cc) in pos.items() if bb == b + 1 and cc == c), None)
            idn = next((k for k, (bb, cc) in pos.items() if bb == b and cc == c + 1), None)
            pu = nu2[(i, iu)] * nu2[(j, iu)] if iu is not None else Fr(0)
            pd = nu2[(i, idn)] * nu2[(j, idn)] if idn is not None else Fr(0)
            if pu != pd:
                bad += 1
    return nlinks, nsites, noff, bad


if __name__ == '__main__':
    tot = [0, 0, 0, 0]; ntup = 0
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    for S in range(1, 12):
        for a2 in range(1, M + 1):
            for a3 in range(1, M + 1):
                for a4 in range(1, M + 1):
                    for a5 in range(1, M + 1):
                        if (S + a2 + a3 + a4 + a5) % 2:
                            continue
                        r = run(S, (a2, a3, a4, a5))
                        ntup += 1
                        tot = [x + y for x, y in zip(tot, r)]
    print(f"tuples {ntup} (2j1 <= 11, doubled legs 1..{M}): links {tot[0]}, interior sites {tot[1]}, "
          f"off-diagonal pairs {tot[2]}; mismatches {tot[3]}")

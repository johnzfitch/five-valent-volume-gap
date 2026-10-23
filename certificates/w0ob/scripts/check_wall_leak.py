"""Lemma W checks (exact rationals), half-integer s, every ordered pairing given as (a2 a3 | a4 a5):
 (W1) exactly one wall has the corner's colour 0;  the other ('odd') wall has colour 1;
 (W2) every cut site of colour 1 lies on the odd wall (so the colour-1 class is closed once that wall's
      cut is removed);
 (W3) on the odd-wall sites, Leak^T Leak = Zin^T Zin - diag(Lambda_w) with Lambda_w(b,c) = s/2 + sigma (eps1(b) - eps2(c))/2
      for the wall d = -s (sigma = +1) and Leak^T Leak = Zin^T Zin + diag(-s/2 + (eps1(b)-eps2(c))/2) for d = +s,
      i.e. Leak^T Leak = Zin^T Zin - s/2 - sigma'(eps1 - eps2)/2; diagonal entries exact, off-diagonals as
      exact products.  Here Leak: wall -> rectangle neighbours outside R, Zin: wall -> neighbours in R.
All tuples with doubled legs <= 9 and 2s in {3,5,7,9}."""
import itertools
from fractions import Fraction as Fr


def h2(n, m, L):
    if 4 * L * L - 1 == 0:
        return 0 * L
    return (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)


def eps(n, m, L):
    """eps(L) = 2 delta(L) + L + 1/2 from the actual links (h(bottom) = 0)."""
    hl = h2(n, m, L) if L > m else 0 * L
    hu = h2(n, m, L + 1) if L + 1 <= n - 1 else 0 * L
    return 2 * (hu - hl) + L + Fr(1, 2)


n_tup = 0; badW1 = 0; badW2 = 0; diag_checks = 0; diag_bad = 0; off_checks = 0; off_bad = 0
for S in (3, 5, 7, 9):
    s = Fr(S, 2)
    for legs in itertools.product(range(1, 10), repeat=4):
        if (S + sum(legs)) % 2:
            continue
        a2, a3, a4, a5 = legs
        n1x, m1x, n2x, m2x = a2 + a3 + 2, abs(a2 - a3), a4 + a5 + 2, abs(a4 - a5)
        rect = {(b, c) for b in range(m1x, n1x - 1, 2) for c in range(m2x, n2x - 1, 2)}
        R = {(b, c) for (b, c) in rect if abs(b - c) <= S <= b + c}
        if len(R) < 2:
            continue
        n_tup += 1
        col = lambda x: ((x[0] + x[1] - S) // 2) % 2
        kp, km = (a4 + a5) % 2, (a2 + a3) % 2
        if sorted((kp, km)) != [0, 1]:
            badW1 += 1
        odd_d = S if kp == 1 else -S          # doubled d of the odd wall
        for x in R:
            b, c = x
            for y in [(b - 2, c), (b + 2, c), (b, c - 2), (b, c + 2)]:
                if y in rect and y not in R and col(x) == 1 and (b - c) != odd_d:
                    badW2 += 1
        n1, m1, n2, m2 = Fr(n1x, 2), Fr(m1x, 2), Fr(n2x, 2), Fr(m2x, 2)
        H1 = lambda bb: h2(n1, m1, Fr(bb, 2)) if Fr(bb, 2) > m1 else Fr(0)
        H2 = lambda cc: h2(n2, m2, Fr(cc, 2)) if Fr(cc, 2) > m2 else Fr(0)
        wall = sorted(x for x in R if x[0] - x[1] == odd_d)
        out_dir = -1 if odd_d < 0 else +1     # outward d-direction
        def nbrs(x, direction):
            b, c = x
            if direction > 0:   # d+1: (b+1,c) via h1(b+1), (b,c-1) via h2(c)
                return [((b + 2, c), H1(b + 2)), ((b, c - 2), H2(c))]
            return [((b - 2, c), H1(b)), ((b, c + 2), H2(c + 2))]
        for x in wall:
            leak = sum(w for y, w in nbrs(x, out_dir) if y in rect and y not in R)
            zin = sum(w for y, w in nbrs(x, -out_dir) if y in R)
            b, c = x
            e = eps(n1, m1, Fr(b, 2)) - eps(n2, m2, Fr(c, 2))
            pred = zin - s / 2 - (e / 2 if odd_d < 0 else -e / 2)
            diag_checks += 1
            if leak != pred:
                diag_bad += 1
            y = (b + 2, c + 2)          # next site on the same wall
            if y in R:
                off_checks += 1
                # shared outward neighbour and shared inward neighbour, with the two link products
                if out_dir < 0:
                    so, si = (b, c + 2), (b + 2, c)
                    po = H2(c + 2) * H1(b + 2) if (so in rect and so not in R) else 0
                    pi = H1(b + 2) * H2(c + 2) if si in R else 0
                else:
                    so, si = (b + 2, c), (b, c + 2)
                    po = H1(b + 2) * H2(c + 2) if (so in rect and so not in R) else 0
                    pi = H2(c + 2) * H1(b + 2) if si in R else 0
                if po != pi:
                    off_bad += 1
print(f'tuples {n_tup}: (W1) violations {badW1}; (W2) colour-1 cut sites off the odd wall {badW2}; '
      f'(W3) diagonal {diag_checks} checks, {diag_bad} mismatches; off-diagonal {off_checks} checks, {off_bad} mismatches')

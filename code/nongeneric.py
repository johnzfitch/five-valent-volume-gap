"""Five-valent sign vectors off general position (the remark "Tangents not in general position", Section 4).

Tangent directions with some three coplanar but not all five give sign vectors in
{0,+-1}^10 with a zero entry. Two enumerations, compared:
  (A) abstract: every eps in {0,+-1}^10, not identically zero, satisfying the chirotope
      axioms of rank 3 (basis exchange for the support, three-term Grassmann-Pluecker
      relations), and simple (no two elements parallel);
  (B) realised: random integer tangents with entries in [-k,k], k = 2,3,4, pairwise
      non-parallel, with some zero triple determinant.
Then relabelling orbits, the reduced signs sigma of BR (4.12), and the presentations of
the operator as lambda J_a.(J_b x J_c) + mu J_a.(J_d x J_e) (the form of eq. (1.1)), or
as a combination of any two triple products. Exact integer and rational arithmetic.

Usage: python nongeneric.py [output file]   (default outputs/nongeneric_out.txt)
"""
import itertools, os, random, sys
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orbits import TRI, chirotope, sigma_of, sigma_via_T, relabel, normal_forms, T_coords


def chi(E, i, j, k):
    lst = [i, j, k]
    if len(set(lst)) < 3:
        return 0
    s = 1
    for x in range(3):
        for y in range(x + 1, 3):
            if lst[x] > lst[y]:
                s = -s
    return s * E[tuple(sorted(lst))]


def is_chirotope(eps):
    E = dict(zip(TRI, eps))
    B = {frozenset(t) for t, e in zip(TRI, eps) if e}
    if not B:
        return False
    for b1 in B:
        for b2 in B:
            for x in b1 - b2:
                if not any((b1 - {x}) | {y} in B for y in b2 - b1):
                    return False
    for a in range(5):
        rest = [x for x in range(5) if x != a]
        for (b1, b2, b3, b4) in itertools.permutations(rest):
            S = {chi(E, a, b1, b2) * chi(E, a, b3, b4), -chi(E, a, b1, b3) * chi(E, a, b2, b4),
                 chi(E, a, b1, b4) * chi(E, a, b2, b3)}
            if S != {0} and not ({1, -1} <= S):
                return False
    return True


def simple(eps):
    E = dict(zip(TRI, eps))
    return all(any(E[t] for t in TRI if a in t and b in t) for a, b in itertools.combinations(range(5), 2))


def parallel(vs):
    for a, b in itertools.combinations(vs, 2):
        if (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]) == (0, 0, 0):
            return True
    return False


def two_products(eps):
    """Presentations by any two distinct triple products, coordinates as in orbits.py."""
    sv = sigma_via_T(eps)
    out = []
    for t1, t2 in itertools.combinations(TRI, 2):
        u, w = T_coords(*t1), T_coords(*t2)
        for (p, q) in itertools.combinations(range(4), 2):
            det = u[p] * w[q] - u[q] * w[p]
            if det:
                lam = Fr(sv[p] * w[q] - sv[q] * w[p], det)
                mu = Fr(u[p] * sv[q] - u[q] * sv[p], det)
                if all(lam * u[r] + mu * w[r] == sv[r] for r in range(4)):
                    out.append((t1, t2, lam, mu))
                break
    return out


def main(out):
    P = lambda *a: print(*a, file=out)
    abstract = [e for e in itertools.product([-1, 0, 1], repeat=10)
                if 0 in e and any(e) and is_chirotope(e) and simple(e)]
    rnd = random.Random(20261022)
    real = {}
    for k in (2, 3, 4):
        for _ in range(600000):
            vs = [tuple(rnd.randint(-k, k) for _ in range(3)) for _ in range(5)]
            if any(v == (0, 0, 0) for v in vs) or parallel(vs):
                continue
            e = chirotope(vs)
            if any(e) and 0 in e:
                real.setdefault(e, vs)
    P(f"simple non-uniform rank-3 chirotopes (abstract): {len(abstract)}")
    P(f"realised by pairwise non-parallel integer tangents in [-4,4]: {len(real)}; same set: {set(real) == set(abstract)}")
    perms = list(itertools.permutations(range(5)))
    seen, orbits = set(), []
    for e in abstract:
        if e in seen:
            continue
        orb = {relabel(e, p) for p in perms}
        seen |= orb
        orbits.append(sorted(orb))
    orbits.sort(key=lambda o: (len(o), o[0]))
    P(f"relabelling orbits: {len(orbits)}, sizes {[len(o) for o in orbits]}")
    entries, ratios, weights, nf_orbits, other = set(), set(), set(), 0, []
    for k, orb in enumerate(orbits):
        sigs = {sigma_of(e) for e in orb}
        entries |= {x for s in sigs for x in s}
        nfs = normal_forms(orb[0])
        closed = all(tuple(-x for x in e) in set(orb) for e in orb)
        if nfs:
            pres = {(abs(l), abs(m)) for (_, _, _, (l, m)) in nfs}
            ts = sorted({min(a, b) / max(a, b) for a, b in pres})
            ws = sorted({max(a, b) for a, b in pres})
            ratios |= set(ts); weights |= set(ws); nf_orbits += 1
            desc = f"form (1.1): |t| in {[str(x) for x in ts]}, larger weight {[str(x) for x in ws]}"
        else:
            tp = two_products(orb[0])
            other.append(len(orb))
            desc = "not of the form (1.1); " + ("two triple products sharing two edges" if tp else "no two triple products suffice")
        P(f"orbit {k}: size {len(orb)}, {'closed' if closed else 'not closed'} under eps -> -eps, "
          f"{len(sigs)} distinct sigma, e.g. {max(sigs)}: {desc}")
    P(f"sigma entries: {sorted(entries)}")
    P(f"orbits of the form (1.1): {nf_orbits}, |t| in {[str(x) for x in sorted(ratios)]}, |alpha| in {[str(x) for x in sorted(weights)]}")
    P(f"other orbits: {len(other)} ({sum(other)} sign vectors)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "outputs", "nongeneric_out.txt")
    with open(path, "w") as f:
        main(f)
    print(open(path).read())

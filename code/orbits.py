"""Chirotopes of five tangent directions, their relabelling orbits, the reduced sign
factors sigma of BR II (4.12), and the normal form of every orbit.

Exact integer arithmetic throughout. Output: outputs/orbits_out.txt
"""
import itertools, random, sys
from fractions import Fraction as Fr

TRI = list(itertools.combinations(range(5), 3))          # all 10 triples, 0-based
RED = list(itertools.combinations(range(4), 3))          # 123,124,134,234 (I<J<K<5)


def det3(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def sgn(x):
    return (x > 0) - (x < 0)


def chirotope(vs):
    return tuple(sgn(det3(*(vs[i] for i in t))) for t in TRI)


def gp_ok(eps):
    """Three-term Grassmann-Pluecker sign condition for a uniform rank-3 chirotope on 5
    elements: for a in E and distinct b,c,d,e not a, the set
    {chi(a,b,c)chi(a,d,e), -chi(a,b,d)chi(a,c,e), chi(a,b,e)chi(a,c,d)} contains +1."""
    E = dict(zip(TRI, eps))

    def chi(i, j, k):
        if len({i, j, k}) < 3:
            return 0
        perm = sorted((i, j, k))
        s = 1
        lst = [i, j, k]
        for x in range(3):
            for y in range(x + 1, 3):
                if lst[x] > lst[y]:
                    s = -s
        return s * E[tuple(perm)]

    for a in range(5):
        rest = [x for x in range(5) if x != a]
        b, c, d, e = rest
        for (b, c, d, e) in itertools.permutations(rest):
            terms = {chi(a, b, c) * chi(a, d, e), -chi(a, b, d) * chi(a, c, e), chi(a, b, e) * chi(a, c, d)}
            if 1 not in terms:
                return False
    return True


def chi_of(eps, i, j, k):
    E = dict(zip(TRI, eps))
    lst = [i, j, k]
    if len(set(lst)) < 3:
        return 0
    s = 1
    for x in range(3):
        for y in range(x + 1, 3):
            if lst[x] > lst[y]:
                s = -s
    return s * E[tuple(sorted(lst))]


def T_coords(i, j, k):
    """Coordinates of J_i.(J_j x J_k) on Inv in the basis T123,T124,T134,T234
    (0-based legs 0..4, leg 4 eliminated). Alternating in (i,j,k)."""
    lst = [i, j, k]
    if len(set(lst)) < 3:
        return [0, 0, 0, 0]
    s = 1
    for x in range(3):
        for y in range(x + 1, 3):
            if lst[x] > lst[y]:
                s = -s
    I, J, K = sorted(lst)
    if K < 4:
        v = [0, 0, 0, 0]
        v[RED.index((I, J, K))] = s
        return v
    # T_{I J 4} = -sum_{L not in {I,J,4}} T_{I J L}
    v = [0, 0, 0, 0]
    for L in range(4):
        if L in (I, J):
            continue
        w = T_coords(I, J, L)
        v = [a - b for a, b in zip(v, w)]
    return [s * x for x in v]


def sigma_of(eps):
    """BR II (4.12): sigma(IJK) = eps(IJK) - eps(IJN) + eps(IKN) - eps(JKN), N = 5."""
    E = dict(zip(TRI, eps))
    out = []
    for (I, J, K) in RED:
        out.append(E[(I, J, K)] - E[(I, J, 4)] + E[(I, K, 4)] - E[(J, K, 4)])
    return tuple(out)


def sigma_via_T(eps):
    v = [0, 0, 0, 0]
    for t, e in zip(TRI, eps):
        w = T_coords(*t)
        v = [a + e * b for a, b in zip(v, w)]
    return tuple(v)


def relabel(eps, pi):
    """Chirotope of the relabelled tangent set: new leg i carries old leg pi[i]."""
    return tuple(chi_of(eps, pi[t[0]], pi[t[1]], pi[t[2]]) for t in TRI)


def normal_forms(eps):
    """All presentations sigma = lam*T(a;b,c) + mu*T(a;d,e) (coords on Inv)."""
    sv = sigma_via_T(eps)
    res = []
    for a in range(5):
        rest = [x for x in range(5) if x != a]
        for b, c in itertools.combinations(rest, 2):
            d, e = [x for x in rest if x not in (b, c)]
            u = T_coords(a, b, c); w = T_coords(a, d, e)
            # solve sv = lam u + mu w exactly (overdetermined, 4 eqs)
            sol = None
            for (p, q) in itertools.combinations(range(4), 2):
                det = u[p] * w[q] - u[q] * w[p]
                if det != 0:
                    lam = Fr(sv[p] * w[q] - sv[q] * w[p], det)
                    mu = Fr(u[p] * sv[q] - u[q] * sv[p], det)
                    if all(lam * u[r] + mu * w[r] == sv[r] for r in range(4)):
                        sol = (lam, mu)
                    break
            if sol is not None:
                res.append((a, (b, c), (d, e), sol))
    return res


def main(out=sys.stdout):
    allsigns = list(itertools.product([1, -1], repeat=10))
    chis = [e for e in allsigns if gp_ok(e)]
    print(f"sign vectors satisfying 3-term GP: {len(chis)}", file=out)
    # realisation by integer tangents
    rnd = random.Random(20261021)
    real = {}
    for _ in range(400000):
        vs = [tuple(rnd.randint(-4, 4) for _ in range(3)) for _ in range(5)]
        e = chirotope(vs)
        if 0 in e:
            continue
        real.setdefault(e, vs)
        if len(real) == len(chis):
            break
    print(f"realised by integer tangents in [-4,4]: {len(real)} (all GP vectors realised: {set(real)==set(chis)})", file=out)
    # sigma consistency
    assert all(sigma_of(e) == sigma_via_T(e) for e in chis)
    print("BR (4.12) agrees with the elimination identity on all chirotopes", file=out)
    # orbits
    perms = list(itertools.permutations(range(5)))
    seen, orbits = set(), []
    for e in chis:
        if e in seen:
            continue
        orb = {relabel(e, p) for p in perms}
        seen |= orb
        orbits.append(sorted(orb))
    orbits.sort(key=len)
    sig_all = set()
    for k, orb in enumerate(orbits):
        sigs = {sigma_of(e) for e in orb}
        sig_all |= sigs
        has4 = [any(abs(x) == 4 for x in s) for s in sigs]
        nfs = normal_forms(orb[0])
        ratios = sorted({abs(m / l) if l != 0 else None for (_, _, _, (l, m)) in nfs if l != 0}, key=lambda x: (x is None, x))
        mags = sorted({(abs(l), abs(m)) for (_, _, _, (l, m)) in nfs})
        dist = sorted({a for (a, _, _, (l, m)) in nfs if l != 0 and m != 0})
        print(f"orbit {k}: size {len(orb)}, closed under -eps: {all(tuple(-x for x in e) in set(orb) for e in orb)}, "
              f"distinct sigma {len(sigs)}, sigma with +-4: {sum(has4)}/{len(sigs)}", file=out)
        print(f"   normal forms of a representative: |lam|,|mu| in {[(str(a),str(b)) for a,b in mags]}, "
              f"|mu/lam| in {[str(r) for r in ratios]}, legs carrying both terms: {dist}", file=out)
        rep = min(sigs, key=lambda s: (-sum(abs(x) for x in s), s))
        print(f"   example sigma (BR order 123,124,134,234): {rep}", file=out)
    print(f"total distinct sigma: {len(sig_all)}; nonzero up to sign: {len({max(s, tuple(-x for x in s)) for s in sig_all if any(s)})}", file=out)
    return orbits, real


if __name__ == "__main__":
    with open(sys.argv[1] if len(sys.argv) > 1 else "outputs/orbits_out.txt", "w") as f:
        main(f)

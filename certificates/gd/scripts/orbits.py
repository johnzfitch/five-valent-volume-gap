"""Exact combinatorics of 5-valent sign configurations.

1. Enumerate all eps in {+-1}^10 (triples of 5 labelled tangents) satisfying the 3-term
   Grassmann-Pluecker relations of a uniform rank-3 chirotope; find an integer realisation for each.
2. Orbits under relabelling (S5) and under the global sign eps -> -eps (tangent parity).
3. sigma(IJK) = eps(IJK) - eps(IJ5) + eps(IK5) - eps(JK5) (BR II (4.12), leg 5 eliminated),
   in BR order (123,124,134,234). For each orbit: the set of sigma, which contain +-4,
   and which have the 'pair' form (a,a,0,0) [one triple product with a composite leg]
   or the 'chain' form (alpha,beta,beta,0).
Writes outputs/orbits.json and prints a report.
"""
import itertools
import json
import sys
import numpy as np

T5 = list(itertools.combinations(range(5), 3))
POS = {t: i for i, t in enumerate(T5)}
T4 = list(itertools.combinations(range(4), 3))


def chi(eps, a, b, c):
    """alternating extension of eps to ordered triples."""
    if len({a, b, c}) < 3:
        return 0
    lst = [a, b, c]
    s = 1
    for i in range(3):
        for j in range(2 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                s = -s
    return s * eps[POS[tuple(lst)]]


def is_chirotope(eps):
    for a in range(5):
        others = [x for x in range(5) if x != a]
        for x1, x2, x3, x4 in itertools.permutations(others, 4):
            t = {chi(eps, a, x1, x2) * chi(eps, a, x3, x4),
                 -chi(eps, a, x1, x3) * chi(eps, a, x2, x4),
                 chi(eps, a, x1, x4) * chi(eps, a, x2, x3)}
            if not ({1, -1} <= t):
                return False
    return True


def act(eps, p):
    """relabel: new eps'(I,J,K) = chi(eps, p[I], p[J], p[K])."""
    return tuple(chi(eps, p[I], p[J], p[K]) for (I, J, K) in T5)


def sigma(eps):
    return tuple(eps[POS[(I, J, K)]] - eps[POS[(I, J, 4)]] + eps[POS[(I, K, 4)]] - eps[POS[(J, K, 4)]]
                 for (I, J, K) in T4)


def eps_of_vectors(V):
    return tuple(int(np.sign(round(np.linalg.det(np.array([V[i], V[j], V[k]]))))) for (i, j, k) in T5)


def main(out):
    allc = [e for e in itertools.product((-1, 1), repeat=10) if is_chirotope(e)]
    print(f"uniform rank-3 chirotopes on 5 labelled elements (3-term GP): {len(allc)}")
    # realisations with small integer vectors
    rng = np.random.default_rng(7)
    wit = {}
    tries = 0
    while len(wit) < len(allc) and tries < 2_000_000:
        V = rng.integers(-4, 5, size=(5, 3))
        tries += 1
        e = eps_of_vectors(V)
        if 0 in e or e in wit:
            continue
        wit[e] = V.tolist()
    print(f"realised by integer tangents (entries in [-4,4]): {len(wit)} / {len(allc)} after {tries} draws")
    assert set(wit) <= set(allc)
    perms = list(itertools.permutations(range(5)))
    # orbits under S5 and under S5 x {+-1}
    seen, orbits = set(), []
    for e in sorted(allc):
        if e in seen:
            continue
        orb = {act(e, p) for p in perms}
        orb |= {tuple(-x for x in f) for f in orb}
        seen |= orb
        orbits.append(sorted(orb))
    s5only = set()
    s5orbits = 0
    for e in sorted(allc):
        if e in s5only:
            continue
        s5only |= {act(e, p) for p in perms}
        s5orbits += 1
    print(f"orbits under S5: {s5orbits};  under S5 x global sign: {len(orbits)}")
    report = []
    for k, orb in enumerate(orbits):
        sig = sorted({sigma(e) for e in orb})
        nz = [s for s in sig if any(s)]
        has4 = [s for s in sig if 4 in map(abs, s)]
        pair = [s for s in sig if sorted(map(abs, s)) == [0, 0, 2, 2]]
        chain = [s for s in sig if s[3] == 0 and s[1] == s[2] and s[0] != 0]
        reps = sorted({max(s, tuple(-x for x in s)) for s in sig})
        print(f"\norbit {k}: |orbit| = {len(orb)} eps, {len(sig)} sigma ({len(reps)} up to sign); "
              f"with +-4: {len(has4)}; without: {len(sig) - len(has4)}")
        print("   sigma up to sign:", reps)
        print("   chain form (alpha,beta,beta,0):", chain)
        print("   pair form |sigma| = (2,2,0,0) up to order:", pair)
        report.append(dict(orbit=k, size=len(orb), sigma=sig, sigma_up_to_sign=reps,
                           has4=has4, chain=chain, pair=pair,
                           witness={str(e): wit.get(e) for e in orb[:3]}))
    # sigma -> orbit map, check no sigma lies in two orbits
    owner = {}
    for r in report:
        for s in r["sigma"]:
            owner.setdefault(tuple(s), set()).add(r["orbit"])
    multi = {s: o for s, o in owner.items() if len(o) > 1}
    print(f"\ndistinct sigma: {len(owner)} (BR: 171); sigma occurring in >1 orbit: {len(multi)}")
    with open(out, "w") as f:
        json.dump(dict(n_chirotopes=len(allc), orbits=report,
                       witnesses={str(k): v for k, v in wit.items()}), f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "orbits.json")

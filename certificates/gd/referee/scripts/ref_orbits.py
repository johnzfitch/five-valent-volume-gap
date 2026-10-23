"""Referee script (independent of the author's code).

Valence-5 chirotopes, relabelling orbits, sigma reduction, normal forms.

* enumerate all eps in {+-1}^10 satisfying the 3-term Grassmann-Pluecker
  sign condition of a uniform rank-3 chirotope;
* realise each by random integer tangents (exact integer determinants);
* sigma-reduction derived here from the identity, valid on Inv,
  T_{IJ5} = - sum_{K not in {I,J,5}} T_{IJK}   (T_{IJK} = J_I.(J_J x J_K)),
  and compared with BR II (4.12);
* orbits under S5 relabelling; sigma counts per orbit; +-4 property;
* every normal-form presentation  sigma(eps) = lam*(T_abc + t*T_ade)
  for every leg a and every ordered split, to find all t values and the
  distinguished edge;
* sorted-convention patterns of the orbit containing (4,2,2,0).
Usage: python -I ref_orbits.py OUTFILE
"""
import itertools, sys, random
from fractions import Fraction

L = 5
TRIPLES = list(itertools.combinations(range(L), 3))
TIDX = {t: i for i, t in enumerate(TRIPLES)}
BASIS = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]   # BR order 123,124,134,234


def perm_sign(seq):
    s = 1
    seq = list(seq)
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                s = -s
    return s


def chi(eps, I, J, K):
    if len({I, J, K}) < 3:
        return 0
    t = tuple(sorted((I, J, K)))
    return perm_sign((I, J, K)) * eps[TIDX[t]] if True else 0


def chi_fix(eps, I, J, K):
    # sign of permutation taking sorted triple to (I,J,K)
    t = sorted((I, J, K))
    s = perm_sign([t.index(x) for x in (I, J, K)])
    return s * eps[TIDX[tuple(t)]]


def gp_ok(eps):
    for x in range(L):
        a, b, c, d = [y for y in range(L) if y != x]
        t1 = chi_fix(eps, x, a, b) * chi_fix(eps, x, c, d)
        t2 = -chi_fix(eps, x, a, c) * chi_fix(eps, x, b, d)
        t3 = chi_fix(eps, x, a, d) * chi_fix(eps, x, b, c)
        s = {t1, t2, t3}
        if not (1 in s and -1 in s):
            return False
    return True


def det3(u, v, w):
    return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


def eps_of(tangents):
    out = []
    for (i, j, k) in TRIPLES:
        d = det3(tangents[i], tangents[j], tangents[k])
        if d == 0:
            return None
        out.append(1 if d > 0 else -1)
    return tuple(out)


def T_vec(I, J, K):
    """Coefficient vector (length 4, BASIS) of T_IJK on Inv, alternating."""
    if len({I, J, K}) < 3:
        return [0, 0, 0, 0]
    t = sorted((I, J, K))
    s = perm_sign([t.index(x) for x in (I, J, K)])
    if 4 not in t:
        v = [0, 0, 0, 0]
        v[BASIS.index(tuple(t))] = s
        return v
    I2, J2 = t[0], t[1]           # T_{I2 J2 4} = - sum_K T_{I2 J2 K}
    v = [0, 0, 0, 0]
    for K in range(4):
        if K in (I2, J2):
            continue
        w = T_vec(I2, J2, K)
        v = [a - b for a, b in zip(v, w)]
    return [s * a for a in v]


def sigma_lin(eps):
    v = [0, 0, 0, 0]
    for (I, J, K) in TRIPLES:
        w = T_vec(I, J, K)
        v = [a + eps[TIDX[(I, J, K)]] * b for a, b in zip(v, w)]
    return tuple(v)


def sigma_br(eps):
    e = lambda I, J, K: eps[TIDX[(I, J, K)]]
    return tuple(e(I, J, K) - e(I, J, 4) + e(I, K, 4) - e(J, K, 4)
                 for (I, J, K) in BASIS)


def relabel(eps, pi):
    # new leg I is old leg pi[I]
    return tuple(chi_fix(eps, pi[I], pi[J], pi[K]) for (I, J, K) in TRIPLES)


def normal_forms(sig):
    """All (a, (b,c), (d,e), t) with sig = lam*(T_abc + t T_ade), lam != 0.
    Also t = inf encoded by representation with roles swapped (covered by
    enumerating both orders of the pairs)."""
    res = []
    for a in range(L):
        rest = [x for x in range(L) if x != a]
        for pair in itertools.combinations(rest, 2):
            other = tuple(x for x in rest if x not in pair)
            u = T_vec(a, *pair)
            w = T_vec(a, *other)
            # solve sig = lam*u + mu*w  (u,w independent?)
            sols = []
            for i, j in itertools.combinations(range(4), 2):
                det = u[i] * w[j] - u[j] * w[i]
                if det != 0:
                    lam = Fraction(sig[i] * w[j] - sig[j] * w[i], det)
                    mu = Fraction(u[i] * sig[j] - u[j] * sig[i], det)
                    if all(lam * u[k] + mu * w[k] == sig[k] for k in range(4)):
                        sols.append((lam, mu))
                    break
            for lam, mu in sols:
                if lam != 0:
                    res.append((a, pair, other, lam, mu))
    return res


def main():
    out = []
    P = lambda *s: out.append(' '.join(str(x) for x in s))
    # 1. GP enumeration
    gp = [e for e in itertools.product((1, -1), repeat=10) if gp_ok(e)]
    P('GP-uniform sign vectors:', len(gp))
    # 2. realisations
    rng = random.Random(12345)
    real = {}
    for _ in range(400000):
        T = [tuple(rng.randint(-4, 4) for _ in range(3)) for _ in range(L)]
        e = eps_of(T)
        if e is not None and e not in real:
            real[e] = T
        if len(real) == len(gp):
            break
    P('realised (entries in [-4,4]):', len(real), ' all GP vectors realised:',
      set(real) == set(gp))
    # 3. sigma formula check
    P('sigma_lin == BR(4.12) for all 384:', all(sigma_lin(e) == sigma_br(e) for e in gp))
    # 4. orbits
    perms = list(itertools.permutations(range(L)))
    seen, orbits = set(), []
    for e in gp:
        if e in seen:
            continue
        orb = {relabel(e, p) for p in perms}
        seen |= orb
        orbits.append(sorted(orb))
    orbits.sort(key=lambda o: (len(o), min(map(lambda x: sigma_lin(x), o))))
    # order: the zero orbit first
    def has(o, s):
        return any(sigma_lin(x) == s for x in o)
    named = {}
    for o in orbits:
        if all(sigma_lin(x) == (0, 0, 0, 0) for x in o):
            named[0] = o
        elif has(o, (2, 0, 0, 0)):
            named[1] = o
        elif has(o, (2, 2, 2, 0)):
            named[2] = o
        elif has(o, (2, 2, 4, 0)):
            named[3] = o
    P('orbit sizes:', [len(o) for o in orbits], ' named:', sorted(named))
    allsig = set()
    for k in range(4):
        o = named[k]
        sigs = {sigma_lin(x) for x in o}
        allsig |= sigs
        neg_closed = all(tuple(-y for y in x) in set(o) for x in o)
        has4 = [any(abs(c) == 4 for c in s) for s in sigs]
        P(f'orbit {k}: size {len(o)}, distinct sigma {len(sigs)}, closed under -eps {neg_closed},'
          f' sigmas with +-4: {sum(has4)}/{len(sigs)}, contains (4,2,2,0): {(4,2,2,0) in sigs}')
    P('total distinct sigma:', len(allsig), ' nonzero up to sign:',
      len({max(s, tuple(-c for c in s)) for s in allsig if any(s)}))
    # sigma -> orbit uniqueness
    owner = {}
    clash = 0
    for k in range(4):
        for x in named[k]:
            s = sigma_lin(x)
            if s in owner and owner[s] != k:
                clash += 1
            owner[s] = k
    P('sigma occurring in two orbits:', clash)
    # eps -> sigma injective?
    P('eps->sigma injective on nonzero orbits:',
      len({sigma_lin(x) for k in (1, 2, 3) for x in named[k]}) == 360)
    # 5. normal forms
    for k in (1, 2, 3):
        ts, dist, nform_sigma = set(), set(), set()
        per_eps_ok = True
        for x in named[k]:
            nf = normal_forms(sigma_lin(x))
            tv = {abs(mu / lam) for (_, _, _, lam, mu) in nf}
            ts |= tv
            if not nf:
                per_eps_ok = False
            # distinguished legs: legs a appearing with t != 0
            legs = {a for (a, _, _, lam, mu) in nf if mu != 0}
            dist.add(len(legs))
            # lam values with t chosen <=1
            for (a, p, q, lam, mu) in nf:
                if abs(mu) <= abs(lam):
                    nform_sigma.add((abs(lam), abs(mu)))
        P(f'orbit {k}: every eps has a normal form {per_eps_ok}; |t| values {sorted(ts)};'
          f' #legs a with t!=0 per eps {sorted(dist)}; (|lam|,|mu|) with |mu|<=|lam| (in units of T, q=-4iT): {sorted(nform_sigma)}')
    # list normal forms for representative eps of orbits 2,3
    for k, s in ((2, (2, 2, 2, 0)), (3, (4, 2, 2, 0)), (1, (2, 0, 0, 0))):
        x = next(x for x in named[k] if sigma_lin(x) == s)
        P(f'orbit {k} eps with sigma {s}: eps={x} tangents={real[x]}')
        for nf in normal_forms(s):
            P('   a=%d pair1=%s pair2=%s lam=%s mu=%s' % (nf[0] + 1, tuple(i + 1 for i in nf[1]),
                                                       tuple(i + 1 for i in nf[2]), nf[3], nf[4]))
    # orbit 1: which legs carry the operator (support of T-combination)
    # 6. sorted patterns of orbit 3
    pats = {}
    for x in named[3]:
        s = sigma_lin(x)
        key = max(s, tuple(-c for c in s))
        nf = normal_forms(s)
        # choose presentation with |mu/lam| = 1/2 : lam is the alpha pair
        reps = {(a, tuple(sorted(p)), tuple(sorted(q))) for (a, p, q, lam, mu) in nf
                if abs(mu / lam) == Fraction(1, 2)}
        assert len(reps) == 1, reps
        a, p, q = reps.pop()
        pats.setdefault((a + 1, tuple(i + 1 for i in p)), set()).add(key)
    P('orbit 3 sorted patterns (distinguished position, alpha pair) :', len(pats),
      ' sigma up to sign:', sum(len(v) for v in pats.values()))
    for kpat in sorted(pats):
        P('   k=%d alpha-pair=%s : %s' % (kpat[0], kpat[1], sorted(pats[kpat])))
    six = sorted(set().union(*[v for kk, v in pats.items() if kk[0] == 1 and 2 in kk[1]]))
    P('Corollary A-prime set (k=1, alpha pair contains 2):', len(six), six)
    # orbit 2 and orbit 1 distinguished data
    for k in (1, 2):
        tri = {}
        for x in named[k]:
            s = sigma_lin(x)
            nf = normal_forms(s)
            P_ = {(a + 1, tuple(i + 1 for i in sorted(p)), tuple(i + 1 for i in sorted(q)))
                  for (a, p, q, lam, mu) in nf if (k == 2 and abs(mu) == abs(lam)) or (k == 1 and mu == 0)}
            tri[s] = P_
        nmax = max(len(v) for v in tri.values())
        P(f'orbit {k}: max number of normal-form presentations per sigma: {nmax};'
          f' example {list(tri.items())[0]}')
    with open(sys.argv[1], 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()

"""Validation of the operator and its closed forms.

V1  4-valent: brute force (generators) = Racah recoupling = closed form a_k, all
    tuples with 2j_i <= 4 (all orderings); exact squares of Racah entries vs closed form
    for 2j_i <= 12.
V2  qhat_IJK = [(J_I+J_J)^2,(J_J+J_K)^2] = -4i J_I.(J_J x J_K) (brute force).
V3  5-valent: the full operator sum over all ten eps-weighted triples, built from
    integer tangents of each orbit, against the closed-form grid of the normal form,
    for every tuple with 2j_i <= 3 and a sample with 2j_i <= 5.
V4  BR (4.12): sigma-reduced operator = full eps operator (brute force).
"""
import itertools, sys, random
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
import su2, grid, orbits


def spec_mod(M):
    """Eigenvalue moduli of a normal matrix (Hermitian or anti-Hermitian)."""
    return np.sort(np.abs(np.linalg.eigvals(M)))


def v1(out):
    worst = 0.0; n = 0
    for j in itertools.product(range(1, 5), repeat=4):
        if sum(j) % 2:
            continue
        legs, V = su2.invariant_space(list(j))
        if V.shape[1] == 0:
            continue
        Qb = su2.triple(legs, V, 0, 1, 2)
        Qr, ks = su2.Q4_racah(j)
        H = su2.chain_4valent(j)
        e1 = spec_mod(Qb); e2 = spec_mod(Qr); e3 = np.sort(np.abs(np.linalg.eigvalsh(H)))
        worst = max(worst, np.max(np.abs(e1 - e2)), np.max(np.abs(e1 - e3)))
        n += 1
    print(f"V1a 4-valent, all ordered tuples 2j<=4 with Inv != 0: {n} tuples, max spectral diff {worst:.2e}", file=out)
    # exact: Racah squared entries vs closed form
    bad = 0; cnt = 0
    for j in itertools.product(range(1, 13), repeat=4):
        if sum(j) % 2 or j[0] > j[1] or j[2] > j[3]:
            continue
        ks, ls, U = su2.recoupling_4valent(j)
        if len(ks) < 2:
            continue
        Qr, _ = su2.Q4_racah(j)
        for i in range(1, len(ks)):
            cnt += 1
            if abs(abs(Qr[i - 1, i]) ** 2 - float(su2.a2_closed(j, ks[i]))) > 1e-9 * max(1, float(su2.a2_closed(j, ks[i]))):
                bad += 1
            if abs(Qr[i - 1, i - 1]) > 1e-9:
                bad += 1
    print(f"V1b Racah entries vs closed form (pairs sorted, 2j<=12): {cnt} links, {bad} mismatches; diagonal zero", file=out)


def v2(out):
    worst = 0.0
    for j in [(1, 1, 2, 2, 2), (2, 2, 2, 2, 2), (1, 3, 2, 2, 4), (3, 3, 2, 1, 3)]:
        legs, V = su2.invariant_space(list(j))
        n = len(legs.dims)

        def tot2(a, b):
            M = 0
            for c in range(3):
                X = legs.op(a, c) + legs.op(b, c)
                M = M + X @ X
            return M
        for (I, J, K) in itertools.combinations(range(n), 3):
            C = tot2(I, J) @ tot2(J, K) - tot2(J, K) @ tot2(I, J)
            Cr = V.conj().T @ (C @ V)
            worst = max(worst, np.max(np.abs(Cr - su2.qhat(legs, V, I, J, K))))
    print(f"V2 qhat = commutator of pair Casimirs on Inv: max diff {worst:.2e}", file=out)


def full_eps_operator(spins2, eps):
    legs, V = su2.invariant_space(list(spins2))
    A = np.zeros((V.shape[1], V.shape[1]), dtype=complex)
    for e, (I, J, K) in zip(eps, orbits.TRI):
        A = A + e * su2.qhat(legs, V, I, J, K)
    return A


def v3(out):
    orbs, real = orbits.main(open('nul' if sys.platform.startswith('win') else '/dev/null', 'w'))
    rng = random.Random(7)
    worst = 0.0; cases = 0; kerbad = 0
    tuples = [t for t in itertools.product(range(1, 4), repeat=5) if sum(t) % 2 == 0]
    tuples += [tuple(rng.randint(1, 5) for _ in range(5)) for _ in range(60)]
    tuples = [t for t in tuples if sum(t) % 2 == 0]
    for k in (1, 2, 3):
        for rep in rng.sample(orbs[k], 3):
            nfs = [x for x in orbits.normal_forms(rep)]
            a, (b, c), (d, e), (lam, mu) = nfs[0]
            for t in tuples:
                A = full_eps_operator(t, rep)
                if A.shape[0] == 0:
                    continue
                m1 = spec_mod(A)
                sp = (t[a], t[b], t[c], t[d], t[e])
                W = grid.full_matrix(sp, alpha=lam, beta=mu)
                m2 = np.sort(np.abs(np.linalg.eigvalsh(W))) if W.size else np.array([])
                if len(m1) != len(m2):
                    kerbad += 1; continue
                worst = max(worst, np.max(np.abs(m1 - m2)) / max(1, m1.max()))
                cases += 1
    print(f"V3 full ten-triple operator vs normal-form grid, orbits 1-3, 3 chirotopes each: {cases} cases, "
          f"dimension mismatches {kerbad}, max rel. spectral diff {worst:.2e}", file=out)


def v4(out):
    rng = random.Random(11)
    orbs, real = orbits.main(open('nul' if sys.platform.startswith('win') else '/dev/null', 'w'))
    worst = 0.0; n = 0
    for _ in range(40):
        eps = rng.choice(list(real))
        t = tuple(rng.randint(1, 4) for _ in range(5))
        if sum(t) % 2:
            continue
        A = full_eps_operator(t, eps)
        if A.shape[0] == 0:
            continue
        sig = orbits.sigma_of(eps)
        B = su2.sigma_operator(list(t), sig, list(itertools.combinations(range(4), 3)))
        worst = max(worst, np.max(np.abs(A - B))); n += 1
    print(f"V4 BR (4.12) sigma-reduction = full operator as matrices on Inv: {n} cases, max diff {worst:.2e}", file=out)


if __name__ == "__main__":
    with open(sys.argv[1] if len(sys.argv) > 1 else "outputs/validate_out.txt", "w") as f:
        for fn in (v1, v2, v3, v4):
            fn(f); f.flush()

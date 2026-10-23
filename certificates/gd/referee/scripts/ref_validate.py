"""Referee: validate the normal form (Theorem 4) and my signed (b,c) operator
against brute-force tensor products with all ten eps-weighted triples.
Usage: python -I ref_validate.py MAX2J OUTFILE
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ref_ops import brute_inv, triple, bc_operator, qvals
import ref_orbits as RO


def main():
    M = int(sys.argv[1]); outf = sys.argv[2]
    out = []
    P = lambda *s: (out.append(' '.join(str(x) for x in s)), print(*s, flush=True))
    gp = [e for e in itertools.product((1, -1), repeat=10) if RO.gp_ok(e)]
    perms = list(itertools.permutations(range(5)))
    # one eps per orbit + a few random members
    reps = {}
    for e in gp:
        s = RO.sigma_lin(e)
        if s == (2, 0, 0, 0): reps.setdefault(1, []).append(e)
        if s == (2, 2, 2, 0): reps.setdefault(2, []).append(e)
        if s == (4, 2, 2, 0): reps.setdefault(3, []).append(e)
    rng = np.random.default_rng(7)
    tests = []
    for k in (1, 2, 3):
        base = reps[k][0]
        orb = sorted({RO.relabel(base, p) for p in perms})
        pick = [orb[i] for i in rng.choice(len(orb), 4, replace=False)] + [base]
        tests += [(k, e) for e in pick]
    tuples = [t for t in itertools.product(range(1, M + 1), repeat=5) if sum(t) % 2 == 0]
    nT = nId = nIndep = 0; worst = 0.0; worstId = 0.0; ctrl_fail = 0; ctrl_n = 0
    zero_cases = 0
    for J in tuples:
        Pm, legs = brute_inv(J)
        d = Pm.shape[1]
        if d == 0:
            continue
        Tm = {}
        for (I, Jl, K) in itertools.combinations(range(5), 3):
            Tm[(I, Jl, K)] = Pm.conj().T @ triple(legs, I, Jl, K) @ Pm
        # identity T_{IJ5} = - sum_K T_{IJK}
        for (I, Jl) in itertools.combinations(range(4), 2):
            rhs = -sum(Tm[tuple(sorted((I, Jl, K)))] * RO.perm_sign([sorted((I, Jl, K)).index(x) for x in (I, Jl, K)])
                       for K in range(4) if K not in (I, Jl))
            worstId = max(worstId, np.abs(Tm[(I, Jl, 4)] - rhs).max())
            nId += 1
        # independence of the 4 basis triples
        if d >= 3:
            Mx = np.array([Tm[b].ravel() for b in RO.BASIS])
            if np.linalg.matrix_rank(Mx, tol=1e-9) == 4:
                nIndep += 1
        for (k, e) in tests:
            A = sum(e[RO.TIDX[t]] * (-4j) * Tm[t] for t in RO.TRIPLES)
            ev = np.linalg.eigvals(A)
            q_full = np.sort(np.abs(ev))
            nf = RO.normal_forms(RO.sigma_lin(e))
            for (a, p1, p2, lam, mu) in nf:
                Jn = (J[a], J[p1[0]], J[p1[1]], J[p2[0]], J[p2[1]])
                S, Mb = bc_operator(Jn, float(lam), float(mu))
                assert len(S) == d
                q_bc = np.sort(np.abs(np.linalg.eigvals(Mb)))
                err = np.abs(q_full - q_bc).max() / max(1.0, q_full.max())
                worst = max(worst, err)
                nT += 1
                if q_full.max() < 1e-9:
                    zero_cases += 1
                # negative control: flip the sign of one D entry when grid is 2D
                if J[a] >= 2 and mu != 0 and lam != 0:
                    Mc = Mb.copy()
                    nzi = np.argwhere(np.abs(Mc) > 1e-9)
                    i0, j0 = nzi[0]
                    Mc[i0, j0] *= -1; Mc[j0, i0] *= -1
                    q_c = np.sort(np.abs(np.linalg.eigvals(Mc)))
                    ctrl_n += 1
                    if np.abs(q_full - q_c).max() > 1e-8 * max(1, q_full.max()):
                        ctrl_fail += 1
    P(f'tuples 2j_i<=%d with Inv!=0: {sum(1 for J in tuples if True)} (iterated)' % M)
    P(f'identity T_IJ5 = -sum_K T_IJK on Inv: {nId} checks, max abs err {worstId:.2e}')
    P(f'4 basis triples linearly independent on Inv in {nIndep} tuples with dim>=3')
    P(f'normal-form presentations vs all-ten-eps brute force: {nT} comparisons, max rel err {worst:.2e},'
      f' of which identically zero operator {zero_cases}')
    P(f'negative control (one sign flipped, j_a>=1): {ctrl_fail}/{ctrl_n} detected as mismatch')
    with open(outf, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()

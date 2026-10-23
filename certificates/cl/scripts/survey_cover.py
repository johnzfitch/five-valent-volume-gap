"""Main survey and theorem cross-check (float64, CPU multiprocessing).

Uses the full S4 symmetry of |spec T| in legs 2..5 (Prop. 1): enumerate 2s = 1..M and multisets
2j2 <= 2j3 <= 2j4 <= 2j5 <= M, total spin integer.  For each tuple:
  q      = 8 * smallest nonzero |eig T| (BR normalisation, orbit-2 operator), from pairing P1.
  For each of the three pairings P1=(23|45), P2=(24|35), P3=(25|34):
     q_P (consistency, all equal by S4), closed colour class?, N-sum parity, ladder (a spin-1/2 in a pair)?
     covered_P = closed and (N-sum odd  or ladder)          [Theorem C / C']
  covered = (2s == 1) [Theorem B2 of the earlier note] or any covered_P.
  For covered tuples (first covering pairing): check lam(T0) >= 1, mu >= mu_lb(s), q >= 2 sqrt(mu_lb(s)).
  For all tuples: best over pairings of the proved-if-Conjecture-S quantity 2 sqrt(mu) lam(T0) (<= q checked).
Writes a JSON digest.  Usage: python survey_cover.py M workers out.json
"""
import sys, os, json, time, math
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import grid, sites, bip
from explore_T0 import T0, scale, nzmin
from closed_class import closed_classes

TOL = 1e-9


def mu_lb(s2):
    """(2s+2) * Phi_full(s): Phi_full = 2s (s integer), 2s / W_{s-1/2} (s half-integer)."""
    s = s2 / 2
    if s2 % 2 == 0:
        Phi = 2 * s
    else:
        K = (s2 - 1) // 2
        W = 1.0
        for l in range(1, K + 1):
            W *= 4 * l * l / (4 * l * l - 1)
        Phi = 2 * s / W
    return (2 * s + 2) * Phi


def mu_actual(tj, S):
    sig = scale(tj)
    ev = [sig[p] for p in S if ((p[0] + p[1]) // 2) % 2 == 0]
    od = [sig[p] for p in S if ((p[0] + p[1]) // 2) % 2 == 1]
    return min(ev) * min(od)


def pairings(tj):
    s, a, b, c, d = tj
    return [(s, a, b, c, d), (s, a, c, b, d), (s, a, d, b, c)]


def work(tj):
    s2 = tj[0]
    S, W = grid(tj)
    if len(S) < 2:
        return None
    lt, _ = nzmin(W, S, tol=1e-10)
    if lt is None:
        return None
    q = 8 * lt
    res = {'tj': tj, 'q': q, 'cov': None, 'checks': [], 'best_cond': 0.0, 'qdiff': 0.0}
    covered = (s2 == 1)
    if covered:
        res['cov'] = 'B2'
    for k, P in enumerate(pairings(tj)):
        SP = sites(P)
        if len(SP) < 2:
            continue
        if k > 0:
            _, WP = grid(P)
            lP, _ = nzmin(WP, SP, tol=1e-10)
            res['qdiff'] = max(res['qdiff'], abs(8 * lP - q) / q)
        cl = closed_classes(P)
        N1 = min(P[1], P[2]) + 1; N2 = min(P[3], P[4]) + 1
        odd = (N1 + N2) % 2 == 1
        ladder = (min(P[1], P[2]) == 1) or (min(P[3], P[4]) == 1)
        S0, W0 = T0(P)
        l0, _ = nzmin(W0, S0, tol=1e-10)
        mu = mu_actual(P, S0)
        cond = 2 * math.sqrt(mu) * l0 if l0 else 0.0
        res['best_cond'] = max(res['best_cond'], cond)
        if cond > q * (1 + TOL):
            res['checks'].append(('scaling bound exceeds q', P, cond, q))
        if mu < mu_lb(s2) * (1 - TOL):
            res['checks'].append(('mu below mu_lb', P, mu, mu_lb(s2)))
        if any(cl) and (odd or ladder):
            if l0 < 1 - TOL:
                res['checks'].append(('closed but lam0<1', P, l0))
            if not covered:
                covered = True
                res['cov'] = 'C-odd' if odd else 'C-ladder'
                res['bound'] = 2 * math.sqrt(mu_lb(s2))
                if res['bound'] > q * (1 + TOL):
                    res['checks'].append(('proved bound exceeds q', P, res['bound'], q))
        res.setdefault('lam0', []).append(l0)
    res['covered'] = covered
    return res


def tuples(M):
    for s2 in range(1, M + 1):
        for a in range(1, M + 1):
            for b in range(a, M + 1):
                for c in range(b, M + 1):
                    for d in range(c, M + 1):
                        if (s2 + a + b + c + d) % 2:
                            continue
                        yield (s2, a, b, c, d)


if __name__ == '__main__':
    M, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t0 = time.time()
    by = {}
    bad = []
    qdiff = 0.0
    glob = {'min_q': (9e9, None), 'min_q_uncovered': (9e9, None), 'min_bestcond_uncovered': (9e9, None)}
    with Pool(nw) as p:
        for r in p.imap_unordered(work, tuples(M), chunksize=32):
            if r is None:
                continue
            tj = r['tj']; m = max(tj)
            st = by.setdefault(m, {'n': 0, 'covered': 0, 'B2': 0, 'C-odd': 0, 'C-ladder': 0,
                                   'min_q': (9e9, None), 'min_q_unc': (9e9, None)})
            st['n'] += 1
            if r['covered']:
                st['covered'] += 1; st[r['cov']] += 1
            if r['q'] < st['min_q'][0]: st['min_q'] = (r['q'], tj)
            if r['q'] < glob['min_q'][0]: glob['min_q'] = (r['q'], tj)
            if not r['covered']:
                if r['q'] < st['min_q_unc'][0]: st['min_q_unc'] = (r['q'], tj)
                if r['q'] < glob['min_q_uncovered'][0]: glob['min_q_uncovered'] = (r['q'], tj)
                if r['best_cond'] < glob['min_bestcond_uncovered'][0]:
                    glob['min_bestcond_uncovered'] = (r['best_cond'], tj)
            qdiff = max(qdiff, r['qdiff'])
            bad.extend(r['checks'])
    dt = time.time() - t0
    lines = [f"survey_cover M={M}: {sum(v['n'] for v in by.values())} tuples (2s<=M, multisets of 2j2..2j5<=M), {dt:.1f}s",
             f"max relative difference of q between the three pairings (S4 check): {qdiff:.2e}",
             f"theorem/consistency violations: {len(bad)}"]
    for bb in bad[:20]:
        lines.append('  ' + str(bb))
    lines.append(" 2jmax   tuples  covered(B2/C-odd/C-ladder)   min q (argmin)        min q over uncovered (argmin)")
    for m in sorted(by):
        v = by[m]
        lines.append(f" {m:5d} {v['n']:8d} {v['covered']:8d} ({v['B2']}/{v['C-odd']}/{v['C-ladder']})  "
                     f"{v['min_q'][0]:.9f} {v['min_q'][1]}   {v['min_q_unc'][0]:.6f} {v['min_q_unc'][1]}")
    lines.append(f"global min q = {glob['min_q']}")
    lines.append(f"min q over uncovered = {glob['min_q_uncovered']}")
    lines.append(f"min over uncovered of max over pairings of 2 sqrt(mu) lam(T0) = {glob['min_bestcond_uncovered']}")
    print('\n'.join(lines))
    json.dump({'by_jmax': {str(k): v for k, v in by.items()}, 'global': glob, 'violations': bad[:200],
               'qdiff': qdiff, 'seconds': dt}, open(out, 'w'), indent=1, default=str)

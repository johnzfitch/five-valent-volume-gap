"""Exhaustive float64 survey of the open regions (half-integer s >= 3/2, no pairing closed).

Enumerates 2s and multisets 2j2<=2j3<=2j4<=2j5<=M (|spec T| is S4-invariant in legs 2..5, closing note Prop 1(d)).
For each open tuple:
   q   = 8 * smallest nonzero singular value of T (pairing P1)
   for each pairing P:  b0_P = 2 sqrt(mu_P) lam(W0_P)    (closing note)
                        bG_P = 2 sqrt(Phi_P) lam(KG_P)    (phi-only split, this note)
Writes per-(2jmax) and per-(2s) minima as JSON.  Usage: python oc_survey.py Mmin M workers out.json
"""
import sys, os, json, time, math
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_fast import build, smin_nz, is_open, pairings


def tuples(Mmin, M):
    for s2 in range(3, M + 1, 2):
        for a in range(1, M + 1):
            for b in range(a, M + 1):
                for c in range(b, M + 1):
                    for d in range(max(c, 1), M + 1):
                        if max(s2, d) < Mmin:
                            continue
                        if (s2 + a + b + c + d) % 2 == 0:
                            yield (s2, a, b, c, d)


def work(tj):
    if not is_open(tj):
        return None
    r = {'tj': tj}
    bG = 0.0; b0 = 0.0
    for k, P in enumerate(pairings(tj)):
        o = build(P, ('T', 'KG', 'K') if k == 0 else ('KG', 'K'))
        if k == 0:
            l, ker = smin_nz(o['T'])
            if l is None:
                return None
            r['q'] = 8 * l; r['ker'] = ker; r['imb'] = abs(o['ev'] - o['od'])
        lG, _ = smin_nz(o['KG']); l0, _ = smin_nz(o['K'])
        if lG:
            bG = max(bG, 2 * math.sqrt(o['Phi']) * lG)
        if l0:
            b0 = max(b0, 2 * math.sqrt(o['mu']) * l0)
    r['bG'] = bG; r['b0'] = b0
    return r


if __name__ == '__main__':
    Mmin, M, nw, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    t0 = time.time()
    lev = {}; bys = {}
    glob = {'min_q': [9e9, None], 'min_bG': [9e9, None], 'min_b0': [9e9, None], 'max_q_over_bG': [0, None],
            'bG_exceeds_q': 0, 'b0_exceeds_q': 0, 'n': 0, 'extra_kernel': 0}
    with Pool(nw) as p:
        for r in p.imap_unordered(work, tuples(Mmin, M), chunksize=16):
            if r is None:
                continue
            tj = r['tj']; m = max(tj); s2 = tj[0]
            glob['n'] += 1
            if r['ker'] > r['imb']:
                glob['extra_kernel'] += 1
            for D, key in ((lev, m), (bys, s2)):
                e = D.setdefault(key, {'n': 0, 'min_q': [9e9, None], 'min_bG': [9e9, None], 'min_b0': [9e9, None],
                                       'max_q_over_bG': [0, None]})
                e['n'] += 1
                for f in ('q', 'bG', 'b0'):
                    if r[f] < e['min_' + f][0]:
                        e['min_' + f] = [r[f], tj]
                rat = r['q'] / r['bG'] if r['bG'] > 0 else 9e9
                if rat > e['max_q_over_bG'][0]:
                    e['max_q_over_bG'] = [rat, tj]
            for f in ('q', 'bG', 'b0'):
                if r[f] < glob['min_' + f][0]:
                    glob['min_' + f] = [r[f], tj]
            rat = r['q'] / r['bG'] if r['bG'] > 0 else 9e9
            if rat > glob['max_q_over_bG'][0]:
                glob['max_q_over_bG'] = [rat, tj]
            if r['bG'] > r['q'] * (1 + 1e-9):
                glob['bG_exceeds_q'] += 1
            if r['b0'] > r['q'] * (1 + 1e-9):
                glob['b0_exceeds_q'] += 1
    glob['seconds'] = time.time() - t0
    json.dump({'global': glob, 'by_jmax': {str(k): lev[k] for k in sorted(lev)},
               'by_2s': {str(k): bys[k] for k in sorted(bys)}}, open(out, 'w'), indent=1, default=str)
    print(json.dumps(glob, default=str))
    for k in sorted(lev):
        e = lev[k]
        print(k, e['n'], 'min q', round(e['min_q'][0], 4), e['min_q'][1], ' min bG', round(e['min_bG'][0], 4), e['min_bG'][1],
              ' min b0', round(e['min_b0'][0], 4), ' max q/bG', round(e['max_q_over_bG'][0], 3), e['max_q_over_bG'][1])

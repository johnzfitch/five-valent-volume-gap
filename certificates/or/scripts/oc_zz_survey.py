"""Survey of the zigzag reduction (Theorem Z) over all half-integer-s tuples (multisets, S4 symmetry).

For each tuple (2s odd >= 1, 2j2<=..<=2j5 <= M, integer total, region >= 2 sites):
  q       = 8 * smallest nonzero sv of T (pairing P1)
  kerT    = dim ker T, and per pairing: zigzag imbalances imb_k, monotone?  (no -1 before a +1)
  For each monotone pairing and each end choice ('low','high'):
      check dim ker T == sum |imb_k|  (Theorem Z(ii)),   bound = 8 sigma_min(B')  (Theorem Z(iii)),
      check bound <= q.
  best = max bound over monotone pairings / end choices.
Output: per 2s and per 2jmax: #tuples, #without monotone pairing, #open, min best (argmin), min best over open,
        and global violation counts.   Usage: python oc_zz_survey.py M workers out.json
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_reduce import reduce_square, reduce_choice, ker_dim, monotone_ok
from oc_fast import region, is_open, pairings, smin_nz
from oc_zigzag import zig_blocks


def tuples(M, smax):
    for s2 in range(1, smax + 1, 2):
        for a in range(1, M + 1):
            for b in range(a, M + 1):
                for c in range(b, M + 1):
                    for d in range(c, M + 1):
                        if (s2 + a + b + c + d) % 2 == 0:
                            yield (s2, a, b, c, d)


def work(tj):
    """Admissible pairings (proof level): let S_P = sum_k |imb_k^P|.  If some pairing P* is monotone, then
    dim ker T = S_{P*} (Theorem Z(ii)); every pairing with S_P = S_{P*} then has rank B'_P = rank B_P
    (since dim ker T <= S_P always), so lam(T) >= sigma_min(B'_P)."""
    if len(region(tj)[0]) < 2:
        return None
    B0 = zig_blocks(tj)[0]
    lam, _ = smin_nz(B0)
    if lam is None:
        return None
    q = 8 * lam
    kT = ker_dim(B0)
    res = {'tj': tj, 'q': q, 'open': is_open(tj), 'best': 0.0, 'mono': False, 'viol_ker': 0, 'viol_bound': 0,
           'best_mono_only': 0.0}
    info = []
    for P in pairings(tj):
        B, Bp, kr, kc, imb = reduce_square(P, 'low')
        info.append((P, imb, sum(abs(x) for x in imb), monotone_ok(imb)))
    Sstar = [S for (P, imb, S, mono) in info if mono]
    if not Sstar:
        return res
    res['mono'] = True
    S0 = Sstar[0]
    if kT != S0:
        res['viol_ker'] += 1
    for (P, imb, S, mono) in info:
        if S != S0:
            continue
        for ch in (['low'] * len(imb), ['high'] * len(imb),
                   [('low' if x == -1 else 'high') for x in imb], [('high' if x == -1 else 'low') for x in imb]):
            _, Bp, _ = reduce_choice(P, ch)
            if Bp.size == 0:
                continue
            sB = 8 * np.linalg.svd(Bp, compute_uv=False).min()
            if sB > q * (1 + 1e-9):
                res['viol_bound'] += 1
            res['best'] = max(res['best'], sB)
            if mono:
                res['best_mono_only'] = max(res['best_mono_only'], sB)
    return res


if __name__ == '__main__':
    M, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    smax = int(sys.argv[4]) if len(sys.argv) > 4 else M
    t0 = time.time()
    by_s = {}; by_m = {}
    G = {'n': 0, 'nomono': 0, 'nomono_open': 0, 'viol_ker': 0, 'viol_bound': 0, 'min_best': [9e9, None],
         'min_best_open': [9e9, None], 'min_ratio_best_q': [9e9, None]}
    with Pool(nw) as p:
        for r in p.imap_unordered(work, tuples(M, smax), chunksize=16):
            if r is None:
                continue
            tj = r['tj']; G['n'] += 1
            G['viol_ker'] += r['viol_ker']; G['viol_bound'] += r['viol_bound']
            for D, key in ((by_s, tj[0]), (by_m, max(tj))):
                e = D.setdefault(key, {'n': 0, 'nomono': 0, 'open': 0, 'min_best': [9e9, None], 'min_q': [9e9, None]})
                e['n'] += 1; e['open'] += r['open']
                if r['q'] < e['min_q'][0]:
                    e['min_q'] = [r['q'], tj]
                if not r['mono']:
                    e['nomono'] += 1
                elif r['best'] < e['min_best'][0]:
                    e['min_best'] = [r['best'], tj]
            if not r['mono']:
                G['nomono'] += 1; G['nomono_open'] += r['open']; continue
            if r['best'] < G['min_best'][0]:
                G['min_best'] = [r['best'], tj]
            if r['open'] and r['best'] < G['min_best_open'][0]:
                G['min_best_open'] = [r['best'], tj]
            if r['best'] / r['q'] < G['min_ratio_best_q'][0]:
                G['min_ratio_best_q'] = [r['best'] / r['q'], tj]
    G['seconds'] = time.time() - t0
    json.dump({'global': G, 'by_2s': {str(k): by_s[k] for k in sorted(by_s)}, 'by_2jmax': {str(k): by_m[k] for k in sorted(by_m)}},
              open(out, 'w'), indent=1, default=str)
    print(json.dumps(G, default=str))
    print('2s: n, no-monotone-pairing, open, min 8sigma(B\') [argmin], min q [argmin]')
    for k in sorted(by_s):
        e = by_s[k]
        print(f"  {k:3d} {e['n']:7d} {e['nomono']:6d} {e['open']:7d}  {e['min_best'][0]:.4f} {e['min_best'][1]}  {e['min_q'][0]:.4f} {e['min_q'][1]}")

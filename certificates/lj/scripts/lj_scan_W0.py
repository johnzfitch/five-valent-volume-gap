"""For each 2s and all sorted (2j2<=2j3, 2j4<=2j5, ...) others with 2j<=Mo: true q, and the Lemma-4 bound
2 sqrt(mu) lam(W0) maximised over the three pairings; split covered / uncovered ([DC] Cor. 9).
Usage: python lj_scan_W0.py Mo s2min s2max"""
import sys, os, itertools, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import lj_core as L


def job(args):
    s2, Mo = args
    st = dict(n=0, qmin=(9e9, None), unc=0, unc_q=(9e9, None), unc_bound=(9e9, None), unc_l0=(9e9, None))
    for o4 in itertools.combinations_with_replacement(range(1, Mo + 1), 4):
        tj = (s2,) + o4
        if sum(tj) % 2 or s2 > sum(o4): continue
        r = L.analyse(tj)
        if r is None: continue
        q, res = r
        if q is None: continue
        st['n'] += 1
        if q < st['qmin'][0]: st['qmin'] = (q, tj)
        if any(x['cov'] for x in res): continue
        st['unc'] += 1
        if q < st['unc_q'][0]: st['unc_q'] = (q, tj)
        bnd = max((2 * np.sqrt(x['mu']) * x['l0']) if x['l0'] else 0 for x in res)
        l0b = max((x['l0'] or 0) for x in res)
        if bnd < st['unc_bound'][0]: st['unc_bound'] = (bnd, tj)
        if l0b < st['unc_l0'][0]: st['unc_l0'] = (l0b, tj)
    return s2, st


if __name__ == '__main__':
    Mo, a, b = map(int, sys.argv[1:4])
    t0 = time.time()
    with Pool(8) as P:
        for s2, st in P.imap(job, [(s2, Mo) for s2 in range(a, b + 1)]):
            s = s2 / 2
            print(f"2s={s2:3d} n={st['n']:6d} minq={st['qmin'][0]:8.3f} {st['qmin'][1]}  q/(s+1)={st['qmin'][0]/(s+1):.4f} | "
                  f"uncov={st['unc']:6d} minq_unc={st['unc_q'][0]:8.3f} | min best 2sqrt(mu)lam0 = {st['unc_bound'][0]:8.3f} "
                  f"(/(s+1)={st['unc_bound'][0]/(s+1):.3f}) {st['unc_bound'][1]} | min best lam0={st['unc_l0'][0]:.4f} {st['unc_l0'][1]}", flush=True)
    print(f"elapsed {time.time()-t0:.1f}s")

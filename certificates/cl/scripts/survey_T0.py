"""Exhaustive survey over spins (doubled, max 2j_i <= M; pairs sorted inside, pair-swap reduced):
  lamT  = smallest nonzero |eig T|          (T = J1.(J2xJ3+J4xJ5); q = 8 lamT)
  lam0  = smallest nonzero |eig T0|         (T0: Stark links h23(b), h45(c) on region R)
  bnd   = (1/4) sqrt(min_even sigma^2 * min_odd sigma^2) * lam0   (Heron site-scaling bound)
Records per 2s the minima. Usage: python survey_T0.py M workers out.json
"""
import sys, os, json, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import grid, sites, bip
from explore_T0 import T0, scale, nzmin


def tuples(M):
    pairs = [(p, q) for q in range(1, M + 1) for p in range(1, q + 1)]
    for s in range(1, M + 1):
        for P in pairs:
            for Q in pairs:
                if Q < P:
                    continue
                tj = (s,) + P + Q
                if sum(tj) % 2:
                    continue
                yield tj


def work(tj):
    S = sites(tj)
    if len(S) < 2:
        return None
    S, W = grid(tj)
    lt, _ = nzmin(W, S)
    if lt is None:
        return None
    S0, W0 = T0(tj)
    l0, _ = nzmin(W0, S0)
    sig = scale(tj)
    B, ev, od = bip(S, W)
    me = min(sig[S[i]] for i in ev); mo = min(sig[S[i]] for i in od)
    bnd = 0.25 * np.sqrt(me * mo) * l0
    return (tj, lt, l0, bnd)


if __name__ == '__main__':
    M, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t = time.time()
    best = {}
    with Pool(nw) as p:
        for r in p.imap_unordered(work, tuples(M), chunksize=64):
            if r is None:
                continue
            tj, lt, l0, bnd = r
            s = tj[0]
            b = best.setdefault(s, {'lamT': (9e9,), 'lam0': (9e9,), 'ratio': (9e9,), 'bnd': (9e9,)})
            if lt < b['lamT'][0]: b['lamT'] = (lt, tj)
            if l0 < b['lam0'][0]: b['lam0'] = (l0, tj)
            if bnd < b['bnd'][0]: b['bnd'] = (bnd, tj)
            if bnd > lt * (1 + 1e-9): print('VIOLATION', tj, lt, bnd, flush=True)
            if lt / bnd < b['ratio'][0]: b['ratio'] = (lt / bnd, tj)
    json.dump({str(k): v for k, v in sorted(best.items())}, open(out, 'w'), indent=1)
    for s in sorted(best):
        b = best[s]
        print(f"2s={s:3d} min q=8lamT {8*b['lamT'][0]:9.4f} {b['lamT'][1]}  min lam0 {b['lam0'][0]:.4f} {b['lam0'][1]}  min 8*bound {8*b['bnd'][0]:.4f} {b['bnd'][1]}")
    print('time', round(time.time() - t, 1))

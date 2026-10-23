"""Certify all spin tuples with a given 2j_max (doubled spins; j1 arbitrary <= 2j_max; legs 2-5 as sorted multisets,
by the S4 symmetry of |spec T|, [CL] Prop. 1(d)). Writes one JSON line per tuple: tuple, dims, kernel dim, certified, min q (float).
Usage: python run_cert.py Mlo Mhi nproc"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1')
import sys, json, time, itertools, numpy as np
from fractions import Fraction as Fr
from multiprocessing import Pool
from certcore import matrices, rank_exact
from verify import certify as certify1
from verify2 import certify as certify2
FAST = os.environ.get('CERT_FAST','1') == '1'
C_TARGET = 4.5579607                      # certified bound on |eig 8T|
EPS = (C_TARGET / 8) ** 2 * (1 + 1e-15)   # sigma^2 threshold (rounded up)
def tuples(M):
    for s in range(1, M + 1):
        for j2 in range(1, M + 1):
            for j3 in range(j2, M + 1):
                for j4 in range(j3, M + 1):
                    for j5 in range(j4, M + 1):
                        if max(s, j5) != M: continue
                        if (s + j2 + j3 + j4 + j5) % 2: continue
                        yield (s, j2, j3, j4, j5)
def work(tj):
    D = matrices(tj)
    if D is None or D['nO'] == 0: return dict(t=tj, dim=0 if D is None else D['nE'], skip=True)
    rk = rank_exact(D)
    zB = D['nO'] - rk
    if FAST:
        ok, why, w = certify2(D['Bf'], zB, EPS)
        w = np.sort(w)[zB:]; nz = np.sqrt(np.maximum(w, 0))
    else:
        ok, why = certify1(D['Bf'], zB, EPS)
        sv = np.linalg.svd(D['Bf'], compute_uv=False)
        nz = sv[sv > 1e-9 * max(1, sv.max())]
    return dict(t=tj, nE=D['nE'], nO=D['nO'], kerT=D['nE'] + D['nO'] - 2 * rk, cert=ok, why=why,
                qmin=float(8 * nz.min()) if nz.size else None)
if __name__ == '__main__':
    Mlo, Mhi, P = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for M in range(Mlo, Mhi + 1):
        t0 = time.time(); tl = list(tuples(M))
        with Pool(P) as p:
            res = p.map(work, tl, chunksize=4)
        nc = sum(1 for r in res if not r.get('skip') and not r['cert'])
        qs = [r['qmin'] for r in res if not r.get('skip') and r['qmin'] is not None]
        with open(f"../outputs/cert_M{M:03d}.jsonl", "w") as f:
            for r in res: f.write(json.dumps(r) + "\n")
        print(f"2j_max={M}: tuples {len(tl)}, nonempty {sum(1 for r in res if not r.get('skip'))}, NOT certified {nc}, "
              f"min q {(min(qs) if qs else float('nan')):.6f}, wall {time.time()-t0:.1f}s", flush=True)

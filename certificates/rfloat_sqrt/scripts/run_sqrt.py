"""Re-run of the rigorous-float (RFLOAT) certificates with math.sqrt and an exact per-entry check of the
square-root error (certcore.sqrt_checked), so that no assumption about the C runtime remains.

Two ranges, matching the original runs:
  all  : every tuple with 2j_max = M (j1 arbitrary, legs 2-5 as sorted multisets), as fc/scripts/run_cert.py
  j52  : only 2j1 = 5, as rr/scripts/run52.py
The certificate logic (verify2.certify, threshold 4.5579607, exact kernel) is unchanged.

Usage (from any directory):  python run_sqrt.py all 1 38 6   |   python run_sqrt.py j52 39 70 6
Writes ../outputs/<range>_M<level>.jsonl next to this script and appends a line per level to ../outputs/<range>.log."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from certcore import matrices, rank_exact
from verify2 import certify as certify2
OUT = os.path.join(HERE, '..', 'outputs')
C_TARGET = 4.5579607
EPS = (C_TARGET / 8) ** 2 * (1 + 1e-15)


def tuples(M, mode):
    svals = range(1, M + 1) if mode == 'all' else (5,)
    for s in svals:
        for j2 in range(1, M + 1):
            for j3 in range(j2, M + 1):
                for j4 in range(j3, M + 1):
                    for j5 in range(j4, M + 1):
                        if max(s, j5) != M: continue
                        if (s + j2 + j3 + j4 + j5) % 2: continue
                        yield (s, j2, j3, j4, j5)


def work(tj):
    D = matrices(tj)
    if D is None or D['nO'] == 0:
        return dict(t=tj, dim=0 if D is None else D['nE'], skip=True)
    rk = rank_exact(D)
    zB = D['nO'] - rk
    ok, why, w = certify2(D['Bf'], zB, EPS)
    w = np.sort(w)[zB:]
    nz = np.sqrt(np.maximum(w, 0))
    return dict(t=tj, nE=D['nE'], nO=D['nO'], kerT=D['nE'] + D['nO'] - 2 * rk, cert=ok, why=why,
                qmin=float(8 * nz.min()) if nz.size else None)


if __name__ == '__main__':
    mode, Mlo, Mhi, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    assert mode in ('all', 'j52')
    os.makedirs(OUT, exist_ok=True)
    for M in range(Mlo, Mhi + 1):
        t0 = time.time()
        tl = list(tuples(M, mode))
        with Pool(P) as p:
            res = p.map(work, tl, chunksize=4)
        nc = sum(1 for r in res if not r.get('skip') and not r['cert'])
        qs = [r['qmin'] for r in res if not r.get('skip') and r['qmin'] is not None]
        with open(os.path.join(OUT, f"{mode}_M{M:03d}.jsonl"), "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        line = (f"2j_max={M}: tuples {len(tl)}, nonempty {sum(1 for r in res if not r.get('skip'))}, "
                f"NOT certified {nc}, min q {(min(qs) if qs else float('nan')):.6f}, wall {time.time() - t0:.1f}s")
        print(line, flush=True)
        with open(os.path.join(OUT, f"{mode}.log"), "a") as f:
            f.write(line + "\n")

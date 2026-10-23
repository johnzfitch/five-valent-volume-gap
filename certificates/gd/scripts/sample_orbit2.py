"""Random large-spin probe of orbit 2 (alpha=beta=2) beyond the exhaustive survey: 2j_i up to 120, drawn from
several regimes (uniform; distinguished edge large with small pairs; one pair large, other small; near-symmetric
pairs; spin-1/2 distinguished edge).  Float64 SVD of the bipartite block; zero threshold 1e-12*s_max*dim.
Reports min q_min per regime and the smallest q_min/(2j1+2) (the per-j1 minima in the survey grow ~2.5*(2j1))."""
import sys, os, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import W_matrix
from survey_grid import bip_block, analyse_sv
def q_of(J):
    S, W = W_matrix(J, 2, 2)
    if len(S) < 2: return J, None, len(S)
    B = bip_block(S, W)
    if B.size == 0: return J, None, len(S)
    q, mz, _ = analyse_sv(np.linalg.svd(B, compute_uv=False), len(S))
    return J, q, len(S)
def draws(n, seed):
    rng = np.random.default_rng(seed); out = []
    R = lambda lo, hi: int(rng.integers(lo, hi + 1))
    regimes = {
      "uniform": lambda: (R(1,120), R(1,120), R(1,120), R(1,120), R(1,120)),
      "big_j1_small_pairs": lambda: (R(60,120), R(1,8), R(1,120), R(1,8), R(1,120)),
      "pair_big_pair_small": lambda: (R(1,20), R(60,120), R(60,120), R(1,6), R(1,6)),
      "near_symmetric": lambda: (lambda a,b,d: (R(1,40), a, b, a, b+d))(R(20,100), R(20,100), R(-3,3)),
      "spinhalf_j1": lambda: (1, R(1,120), R(1,120), R(1,120), R(1,120)),
    }
    for name, f in regimes.items():
        k = 0
        while k < n:
            J = f()
            if min(J) < 1 or sum(J) % 2: continue
            out.append((name, J)); k += 1
    return out
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    D = draws(n, 11)
    t0 = time.time()
    with Pool(8) as p:
        res = p.map(q_of, [J for _, J in D], chunksize=2)
    best = {}
    for (name, _), (J, q, d) in zip(D, res):
        if q is None: continue
        r = q / (J[0] + 2)
        b = best.setdefault(name, [1e99, None, 1e99, None, 0, 0])
        if q < b[0]: b[0], b[1] = q, J
        if r < b[2]: b[2], b[3] = r, J
        b[4] += 1; b[5] = max(b[5], d)
    for k, v in best.items():
        print(f"{k:22s} n={v[4]:4d} maxdim={v[5]:5d}  min q={v[0]:.4f} at {v[1]};  min q/(2j1+2)={v[2]:.4f} at {v[3]}")
    print(f"total {time.time()-t0:.0f}s")

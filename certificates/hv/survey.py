"""Survey of the smallest nonzero |eigenvalue| of the AL volume operator at gauge-invariant 5-valent vertices,
over all sorted spin tuples j1<=...<=j5 with 2*j5 <= M and all 85 realizable sigma-configurations
(mod overall sign; sigma5.json from sign_configs.py), BR conventions (hv_core).

For each (tuple, sigma) the operator A = sum sigma qhat (real antisymmetric, float64) is formed and its
singular values (= |eigenvalues| of iA, each nonzero one twice) computed.  An eigenvalue is counted as zero if
s <= TOL * s_max * dim with TOL = 1e-13; we also record the largest "zero" to flag ambiguous separations.

Output (in --out): summary.jsonl, one line per (2j_max, sigma) with min over tuples of q_min and argmin,
plus meta.json with timing and backend.
Usage: python survey.py --M 20 --backend numpy|torch --workers 8 --out survey_M20
       python survey.py --bench 12 --backend numpy|torch   (bounded workload: all tuples with 2j_max = 12)
"""
import argparse
import json
import os
import time
import numpy as np
from multiprocessing import Pool

from hv_core import Vertex

TOL = 1e-13
HERE = os.path.dirname(os.path.abspath(__file__))


def load_sigmas():
    with open(os.path.join(HERE, "sigma5.json")) as f:
        return [tuple(s) for s in json.load(f)]


def tuples_jmax(J5):
    out = []
    for J4 in range(1, J5 + 1):
        for J3 in range(1, J4 + 1):
            for J2 in range(1, J3 + 1):
                for J1 in range(1, J2 + 1):
                    if (J1 + J2 + J3 + J4 + J5) % 2 == 0:
                        out.append((J1, J2, J3, J4, J5))
    return out


TRIP = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def build(Jd):
    v = Vertex(Jd)
    if v.dim == 0:
        return Jd, None
    Q = np.stack([v.qhat(*t) for t in TRIP])        # (4, d, d)
    return Jd, Q


_S = None


def build_and_solve(Jd):
    """CPU worker: build the four qhat and return the per-sigma analysis (numpy backend only)."""
    global _S
    if _S is None:
        _S = np.array(load_sigmas(), dtype=np.float64)
    Jd, Q = build(Jd)
    if Q is None:
        return Jd, None
    A = np.einsum("sk,kij->sij", _S, Q)
    sv = np.linalg.svd(A, compute_uv=False)[:, ::-1]
    return Jd, analyse(sv)


def analyse(sv):
    """sv: (nsig, d) ascending singular values. Returns per-sigma (nzero, qmin_nonzero, max_zero, smax)."""
    d = sv.shape[1]
    smax = sv[:, -1]
    thr = TOL * np.maximum(smax, 1.0) * d
    res = []
    for s, t, m in zip(sv, thr, smax):
        z = s <= t
        nz = s[~z]
        res.append((int(z.sum()), float(nz[0]) if nz.size else None,
                    float(s[z][-1]) if z.any() else 0.0, float(m)))
    return res


def run_tuples(tups, sig, backend, workers, device=None):
    S = np.array(sig, dtype=np.float64)                   # (85, 4)
    t_build = t_eig = 0.0
    results = []
    t0 = time.time()
    if backend == "numpy" and workers > 1:
        pool = Pool(workers)
        for Jd, rr in pool.imap_unordered(build_and_solve, tups, chunksize=2):
            if rr is not None:
                results.append((Jd, rr))
        pool.close()
        pool.join()
        return results, time.time() - t0, float("nan")
    if workers > 1:
        pool = Pool(workers)
        it = pool.imap_unordered(build, tups, chunksize=4)
    else:
        it = map(build, tups)
    for Jd, Q in it:
        if Q is None:
            continue
        t1 = time.time()
        if backend == "numpy":
            A = np.einsum("sk,kij->sij", S, Q)
            sv = np.linalg.svd(A, compute_uv=False)[:, ::-1]
        else:
            import torch
            Qt = torch.as_tensor(Q, device=device)
            St = torch.as_tensor(S, device=device)
            A = torch.einsum("sk,kij->sij", St, Qt)
            sv = torch.linalg.svdvals(A).flip(-1).cpu().numpy()
        t_eig += time.time() - t1
        results.append((Jd, analyse(sv)))
    if workers > 1:
        pool.close()
        pool.join()
    t_total = time.time() - t0
    return results, t_total, t_eig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--Mmin", type=int, default=1)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", type=int, default=None)
    a = ap.parse_args()
    sig = load_sigmas()
    device = None
    if a.backend == "torch":
        import torch
        device = "cuda"
        torch.zeros(1, device=device)
    if a.bench is not None:
        tups = tuples_jmax(a.bench)
        res, tt, te = run_tuples(tups, sig, a.backend, a.workers, device)
        q = min(r[1] for Jd, rr in res for r in rr if r[1] is not None)
        print(json.dumps({"bench_2jmax": a.bench, "backend": a.backend, "workers": a.workers,
                          "tuples": len(tups), "total_s": round(tt, 3), "eig_s": round(te, 3),
                          "global_min_q": q}))
        return
    os.makedirs(a.out, exist_ok=True)
    meta = {"M": a.M, "backend": a.backend, "workers": a.workers, "TOL": TOL, "n_sigma": len(sig),
            "sigmas": sig, "timing": {}}
    with open(os.path.join(a.out, "summary.jsonl"), "w") as fo:
        for J5 in range(a.Mmin, a.M + 1):
            tups = tuples_jmax(J5)
            res, tt, te = run_tuples(tups, sig, a.backend, a.workers, device)
            meta["timing"][J5] = [len(tups), round(tt, 2), round(te, 2)]
            for k, s in enumerate(sig):
                best = None
                amb = 0.0
                for Jd, rr in res:
                    nzero, q, mz, smax = rr[k]
                    if q is not None and (best is None or q < best[0]):
                        best = (q, Jd, nzero, mz, smax)
                    if q is not None and mz > 0:
                        amb = max(amb, mz / q)
                line = {"2jmax": J5, "sigma": s, "q_min": best[0] if best else None,
                        "argmin": best[1] if best else None,
                        "nzero_at_argmin": best[2] if best else None,
                        "max_zero_over_qmin_any_tuple": amb}
                fo.write(json.dumps(line) + "\n")
            fo.flush()
            print(J5, len(tups), f"{tt:.1f}s", flush=True)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()

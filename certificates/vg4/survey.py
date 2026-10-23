"""Exhaustive survey of the smallest positive eigenvalue of Q = J1.(J2xJ3) on
Inv(j1..j4) over all sorted spin tuples 1/2 <= j1<=j2<=j3<=j4 <= N/2.

For each tuple: lam = smallest positive eigenvalue (batched Sturm bisection, float64),
S = sum_k 1/a_k, d, and the Bohr-Sommerfeld/WKB prediction pred = c_d * pi / T0 with
T0 = 8 K(m) / sqrt((r4-r2)(r3-r1)) the zero-volume period (areas A = j+1/2),
c_d = 2 (d odd) or 1 (d even).
Per J4 = 2 j_max a summary row is written; per-tuple data are kept for J4 <= keep.

usage: python survey.py --N 200 --backend torch --out survey_N200
"""
import argparse, os, sys, time, json
import numpy as np
from scipy.special import ellipk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vg_core as vg


def tuples_for_J4(J4):
    r = np.arange(1, J4 + 1)
    J3, J2, J1 = np.meshgrid(r, r, r, indexing="ij")
    m = (J2 <= J3) & (J1 <= J2) & ((J1 + J2 + J3 + J4) % 2 == 0) & (J4 <= J1 + J2 + J3)
    return np.stack([J1[m], J2[m], J3[m], np.full(int(m.sum()), J4)], axis=1).astype(np.int64)


def wkb_pred(T, d):
    A = T.astype(np.float64) / 2 + 0.5
    A1, A2, A3, A4 = A.T
    roots = np.stack([(A1 - A2) ** 2, (A3 - A4) ** 2, (A1 + A2) ** 2, (A3 + A4) ** 2], axis=1)
    roots.sort(axis=1)
    r1, r2, r3, r4 = roots.T
    den = (r4 - r2) * (r3 - r1)
    m = np.where(den > 0, (r3 - r2) * (r4 - r1) / np.where(den > 0, den, 1), 1.0)
    m = np.clip(m, 0, 1)
    K = ellipk(m)
    T0 = 8 * K / np.sqrt(np.where(den > 0, den, np.nan))
    cd = np.where(d % 2 == 1, 2.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cd * np.pi / T0, m


def one_J4(args):
    J4, backend, keep, iters = args
    import torch
    dev = "cuda" if (backend == "torch" and torch.cuda.is_available()) else "cpu"
    t0 = time.time()
    T = tuples_for_J4(J4)
    J1, J2, J3 = T[:, 0], T[:, 1], T[:, 2]
    kmin2 = np.maximum(np.abs(J1 - J2), np.abs(J3 - J4))
    kmax2 = np.minimum(J1 + J2, J3 + J4)
    d = (kmax2 - kmin2) // 2 + 1
    lam = np.full(len(T), np.nan); S = np.full(len(T), np.nan)
    for dd in np.unique(d):
        idx = np.nonzero(d == dd)[0]
        if dd == 1:
            continue
        for c0 in range(0, len(idx), 400000):
            ii = idx[c0:c0 + 400000]
            b2 = vg.offdiag2_batch(T[ii])
            S[ii] = np.sum(1.0 / np.sqrt(b2), axis=1)
            if backend == "torch":
                lam[ii] = vg.smallest_positive_eig(torch.as_tensor(b2, device=dev), iters=iters, xp=torch).cpu().numpy()
            else:
                lam[ii] = vg.smallest_positive_eig(b2, iters=iters)
    pred, m = wkb_pred(T, d)
    ok = d >= 2
    jmax = J4 / 2
    L, SS, TT = lam[ok], S[ok], T[ok]
    P, MM, D = pred[ok], m[ok], d[ok]
    prod = L * SS
    i_min = np.argmin(L); i_ps = np.argmin(prod); i_px = np.argmax(prod); i_br = np.argmin(4 * L / jmax)
    d3 = D >= 3
    i_ps3 = np.nonzero(d3)[0][np.argmin(prod[d3])] if d3.any() else i_ps
    nd = MM < 1 - 1e-12
    ratio = L[nd] / P[nd]
    row = dict(J4=int(J4), n_tuples=int(len(T)), n_nonzero=int(ok.sum()),
               lam_min=float(L[i_min]), lam_min_tuple=[int(x) for x in TT[i_min]],
               lamS_min=float(prod[i_ps]), lamS_min_tuple=[int(x) for x in TT[i_ps]],
               lamS_min_d3=float(prod[i_ps3]), lamS_min_d3_tuple=[int(x) for x in TT[i_ps3]],
               lamS_max=float(prod[i_px]), lamS_max_tuple=[int(x) for x in TT[i_px]],
               thm_violations=int(np.sum(prod < 0.5)), eighth_violations=int(np.sum(L < 0.125)),
               BR_ratio_min=float(4 * L[i_br] / jmax), BR_ratio_min_tuple=[int(x) for x in TT[i_br]],
               BR_violations=int(np.sum(4 * L < jmax)),
               lam_over_jmax_min=float(L[i_br] / jmax),
               wkb_ratio_min=float(ratio.min()) if len(ratio) else None,
               wkb_ratio_max=float(ratio.max()) if len(ratio) else None,
               seconds=time.time() - t0)
    ev = (D % 2 == 0)
    row["lamS_max_even"] = float(prod[ev].max()) if ev.any() else None
    row["lamS_max_even_tuple"] = [int(x) for x in TT[ev][np.argmax(prod[ev])]] if ev.any() else None
    row["lamS_max_odd"] = float(prod[~ev].max()) if (~ev).any() else None
    row["argmin_is_half_half_j_j"] = bool(list(TT[i_min]) == [1, 1, J4, J4])
    row["lam_min_equals_sqrt_j_jp1_over_2"] = float(L[i_min] - 0.5 * np.sqrt(jmax * (jmax + 1)))
    if d3.any():
        r3 = L[d3] / np.sqrt(jmax * (jmax + 1))
        row["d3_lam_over_sqrtjj1_min"] = float(r3.min()); row["d3_lam_over_sqrtjj1_tuple"] = [int(x) for x in TT[d3][np.argmin(r3)]]
    C = np.sqrt(2 * np.sqrt(6.0)); th3 = 1 / np.sqrt(2) + 0.5
    row["violations_lamS_lt_1overC"] = int(np.sum(prod < 1 / C))
    row["violations_S_bound"] = int(np.sum(SS * (jmax + 1) > th3 * np.sqrt(96.0)))
    row["violations_lam_lt_jp1_over_27"] = int(np.sum(L < (jmax + 1) / 27))
    row["max_S_times_jmax_plus_1"] = float(np.max(SS * (jmax + 1)))
    kept = np.column_stack([T, d, lam, S, pred, m]) if J4 <= keep else None
    return row, kept


def run(N, Nmin, backend, out, keep, iters, workers):
    os.makedirs(out, exist_ok=True)
    t_all = time.time()
    jobs = [(J4, backend, keep, iters) for J4 in range(N, Nmin - 1, -1)]
    rows, kept = [], {}
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for row, k in pool.imap_unordered(one_J4, jobs):
                rows.append(row)
                if k is not None:
                    kept[row["J4"]] = k
                print(json.dumps({x: row[x] for x in ("J4", "lam_min", "lamS_min_d3", "lamS_max", "thm_violations", "BR_violations", "seconds")}), flush=True)
    else:
        for jb in jobs:
            row, k = one_J4(jb)
            rows.append(row)
            if k is not None:
                kept[row["J4"]] = k
    rows.sort(key=lambda r: r["J4"])
    with open(os.path.join(out, "summary.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + chr(10))
    if kept:
        np.savez_compressed(os.path.join(out, f"pertuple_J4le{keep}.npz"),
                            data=np.concatenate([kept[k] for k in sorted(kept)]),
                            columns=np.array("J1 J2 J3 J4 d lam S wkb_pred m".split()))
    meta = dict(N=N, Nmin=Nmin, backend=backend, iters=iters, workers=workers,
                total_seconds=time.time() - t_all, n_tuples=int(sum(r["n_tuples"] for r in rows)))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=60)
    ap.add_argument("--Nmin", type=int, default=1)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--out", default="survey_out")
    ap.add_argument("--keep", type=int, default=0)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    run(a.N, a.Nmin, a.backend, a.out, a.keep, a.iters, a.workers)

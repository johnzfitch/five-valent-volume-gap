"""Exhaustive sweep of spin-1/2 5-valent tuples (and optionally all tuples).

Usage: python ref_sweep.py M [all]   (M = max doubled spin)
Leg 0 carries spin 1/2 (doubled 1); legs 1..4 sorted; Q_RS is symmetric under
leg permutations (sum over all triples), so sorting loses nothing.
Outputs one JSON line per tuple to ../outputs/sweep_half_M{M}.jsonl and a summary.
"""
import sys, json, time, itertools, math
import numpy as np
from multiprocessing import Pool
from ref_core import q_triples, absq, dimInv

R3 = math.sqrt(3)


def analyse(js5, half=True):
    d = dimInv(js5)
    if d == 0:
        return None
    rec = {"js": js5, "d": d}
    if d == 1:
        qs, _ = q_triples(js5)
        rec["maxabs_q"] = max(float(np.max(np.abs(q))) for q in qs.values())
        return rec
    qs, _ = q_triples(js5)
    absd = {}; spec = {}
    for T, q in qs.items():
        A, w, U = absq(q)
        absd[T] = (A, w, U)
    scale = max(np.max(np.abs(w)) for (_, w, _) in absd.values())
    tol = 1e-9 * max(1.0, scale)
    Q = sum(v[0] for v in absd.values())
    wQ = np.linalg.eigvalsh(Q)
    rec["Qmin"] = float(wQ[0]); rec["Qker"] = int(np.sum(wQ < tol))
    rec["Qminpos"] = float(wQ[wQ >= tol][0]) if np.any(wQ >= tol) else None
    # Proposition 2 quantity: positive part of sum_T P_T (P_T = range projector of q_T)
    PS = 0
    for T, (A, w, U) in absd.items():
        R = U[:, np.abs(w) >= tol]
        PS = PS + R @ R.conj().T
    wP = np.linalg.eigvalsh(PS)
    rec["Pminpos"] = float(wP[wP >= 1e-9][0]) if np.any(wP >= 1e-9) else None
    rec["Pker"] = int(np.sum(wP < 1e-9))
    # Giesel-Thiemann variant: sum_T sqrt|q_T|
    G = 0
    for T, (A, w, U) in absd.items():
        G = G + (U * np.sqrt(np.abs(w))) @ U.conj().T
    rec["GTmin"] = float(np.linalg.eigvalsh(G)[0])
    # unsigned |sum q_T| (all orientation signs +1), information only
    S = sum(qs.values())
    rec["unsigned_min"] = float(np.min(np.abs(np.linalg.eigvalsh(1j * S))))
    if half:
        T1 = [T for T in qs if 0 in T]
        S1 = sum(absd[T][0] for T in T1)
        rec["S1min"] = float(np.linalg.eigvalsh(S1)[0])
        minnz = math.inf; kmax = 0
        Pi = {}
        for T in T1:
            A, w, U = absd[T]
            nz = np.abs(w[np.abs(w) >= tol])
            if len(nz):
                minnz = min(minnz, float(nz.min()))
            K = U[:, np.abs(w) < tol]
            kmax = max(kmax, K.shape[1])
            Pi[T] = K @ K.conj().T
        rec["minnz_leg1"] = minnz if minnz < math.inf else None
        rec["kerdim_leg1_max"] = kmax
        # complementary pairs {0,J,K},{0,L,M}
        orth = 0.0
        for (J, K) in itertools.combinations(range(1, 5), 2):
            L, M = [x for x in range(1, 5) if x not in (J, K)]
            if J < L:
                orth = max(orth, float(np.linalg.norm(Pi[(0, J, K)] @ Pi[(0, L, M)], 2)))
        rec["orth"] = orth
    return rec


def tuples_half(M):
    for rest in itertools.combinations_with_replacement(range(1, M + 1), 4):
        js5 = (1,) + rest  # leg 4 (root) is the largest
        if sum(js5) % 2:
            continue
        yield js5


def tuples_all(M):
    for js5 in itertools.combinations_with_replacement(range(1, M + 1), 5):
        if sum(js5) % 2:
            continue
        yield js5


def work(js5):
    return analyse(js5, half=(js5[0] == 1))


if __name__ == "__main__":
    M = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else "half"
    tl = list(tuples_half(M) if mode == "half" else tuples_all(M))
    # biggest first for load balance
    tl.sort(key=lambda t: -sum(t))
    t0 = time.time()
    out = f"../outputs/sweep_{mode}_M{M}.jsonl"
    with Pool(8) as pool, open(out, "w") as fh:
        res = []
        for r in pool.imap_unordered(work, tl, chunksize=8):
            if r is None:
                continue
            res.append(r); fh.write(json.dumps(r) + "\n")
    el = time.time() - t0
    d2 = [r for r in res if r["d"] >= 2]
    d1 = [r for r in res if r["d"] == 1]
    lines = [f"mode={mode} M=2j_max={M}: tuples with dim>=1: {len(res)}, dim>=2: {len(d2)}, dim=1: {len(d1)}; {el:.1f}s on 8 CPU processes"]
    if d1:
        lines.append(f"dim=1: max |q_T| entry {max(r['maxabs_q'] for r in d1):.2e} (Q_RS = 0)")
    if d2:
        r = min(d2, key=lambda r: r["Qmin"]); lines.append(f"min lambda(Q_RS) over dim>=2 = {r['Qmin']:.6f} at {r['js']}; max kernel dim of Q_RS = {max(x['Qker'] for x in d2)}")
        ker = [x for x in d2 if x["Qker"] > 0]
        lines.append(f"tuples with nontrivial ker Q_RS: {len(ker)}" + (f" e.g. {[x['js'] for x in ker[:10]]}" if ker else ""))
        pos = [x for x in d2 if x["Qminpos"] is not None]
        r = min(pos, key=lambda r: r["Qminpos"]); lines.append(f"min positive eigenvalue of Q_RS = {r['Qminpos']:.6f} at {r['js']}")
        pp = [x for x in d2 if x.get("Pminpos") is not None]
        if pp:
            r = min(pp, key=lambda r: r["Pminpos"]); lines.append(f"min positive eigenvalue of sum_T P_T = {r['Pminpos']:.6f} at {r['js']}; ker(sum P_T) == ker(Q_RS) dims agree: {all(x['Pker']==x['Qker'] for x in pp)}")
            # check Prop 2 inequality lambda+_min(Q) >= sqrt3 lambda+_min(sum P)
            viol = [x for x in pp if x["Qminpos"] is not None and x["Qminpos"] < R3 * x["Pminpos"] - 1e-9]
            lines.append(f"Prop 2 inequality lambda+_min(Q_RS) >= sqrt3*lambda+_min(sum P_T): violations {len(viol)}")
        r = min(d2, key=lambda r: r["GTmin"]); lines.append(f"min lambda(sum_T sqrt|q_T|) = {r['GTmin']:.6f} at {r['js']}  (3*3^(1/4) = {3*3**0.25:.6f})")
        r = min(d2, key=lambda r: r["unsigned_min"]); lines.append(f"min |eig| of unsigned sum_T q_T = {r['unsigned_min']:.3e} at {r['js']}")
        if mode == "half":
            r = min(d2, key=lambda r: r["S1min"]); lines.append(f"min lambda(sum_(T contains leg 1)|q_T|) = {r['S1min']:.6f} at {r['js']}  (3 sqrt3 = {3*R3:.6f})")
            v = [x["minnz_leg1"] for x in d2 if x["minnz_leg1"] is not None]
            lines.append(f"min nonzero |eig| of q_T, T contains leg 1 = {min(v):.12f} (sqrt3 = {R3:.12f}); count within 1e-9 of sqrt3: {sum(abs(x-R3)<1e-9 for x in v)}")
            lines.append(f"max kernel dim of a leg-1 triple = {max(x['kerdim_leg1_max'] for x in d2)}; max ||Pi_JK Pi_LM|| = {max(x['orth'] for x in d2):.2e}")
        # per-2jmax minimum positive eigenvalue of Q_RS
        per = {}
        for x in pos:
            m = max(x["js"])
            if m not in per or x["Qminpos"] < per[m]["Qminpos"]:
                per[m] = x
        lines.append("per 2j_max: min positive eigenvalue of Q_RS [argmin]: " + "; ".join(f"{m}: {per[m]['Qminpos']:.3f} {per[m]['js']}" for m in sorted(per)))
    txt = "\n".join(lines)
    print(txt)
    open(f"../outputs/sweep_{mode}_M{M}_summary.txt", "w").write(txt + "\n")

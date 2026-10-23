"""Spin-1/2 tuples beyond 2j_max = 40: structured families and a random sample, 41 <= 2j_max <= 80."""
import json, math, random, time
from multiprocessing import Pool
from ref_sweep import analyse
from ref_core import dimInv


def fams():
    out = []
    for t in range(41, 81):
        cand = [(1, 2, 2, t - 1, t), (1, t - 1, t, t, t), (1, 1, 1, t, t), (1, 1, 2, t, t - 1), (1, t, t, t, t - 1),
                (1, 2, t - 2, t - 1, t), (1, 1, t - 1, t, t), (1, t // 2, t // 2, t - 1, t) , (1, 3, 3, t - 1, t)]
        for c in cand:
            c = (1,) + tuple(sorted(c[1:]))
            if sum(c) % 2 == 0 and dimInv(c) >= 2:
                out.append(c)
    random.seed(12345)
    while len(out) < 3000:
        t = random.randint(41, 80)
        r = sorted([random.randint(1, t) for _ in range(3)] + [t])
        c = (1,) + tuple(r)
        if sum(c) % 2 == 0 and dimInv(c) >= 2:
            out.append(c)
    return sorted(set(out), key=lambda c: -sum(c))


if __name__ == "__main__":
    tl = fams(); t0 = time.time()
    with Pool(8) as p:
        res = [r for r in p.imap_unordered(analyse, tl, chunksize=4) if r]
    R3 = math.sqrt(3)
    with open("../outputs/large_half_41_80.jsonl", "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    q = min(res, key=lambda r: r["Qmin"]); s = min(res, key=lambda r: r["S1min"])
    ratio = min(res, key=lambda r: r["Qmin"] / max(r["js"]))
    lines = [f"{len(res)} spin-1/2 tuples with 41 <= 2j_max <= 80 and dim>=2 (families + random); {time.time()-t0:.0f}s on 8 CPU processes",
             f"min lambda(Q_RS) = {q['Qmin']:.4f} at {q['js']}; any kernel: {any(r['Qker'] for r in res)}",
             f"min lambda(sum_(T contains 1)|q_T|) = {s['S1min']:.4f} at {s['js']} (3 sqrt3 = {3*R3:.4f})",
             f"min lambda(Q_RS)/(2j_max) = {ratio['Qmin']/max(ratio['js']):.4f} at {ratio['js']}",
             f"min nonzero leg-1 |eig| = {min(r['minnz_leg1'] for r in res):.12f}; max leg-1 kernel dim = {max(r['kerdim_leg1_max'] for r in res)}; max ||Pi Pi|| = {max(r['orth'] for r in res):.2e}",
             f"max dim Inv in sample = {max(r['d'] for r in res)}"]
    txt = "\n".join(lines); print(txt)
    open("../outputs/large_half_41_80_summary.txt", "w").write(txt + "\n")

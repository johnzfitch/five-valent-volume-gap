"""Sorted-convention survey (BR 2007 convention: spins sorted j1<=...<=j5, sigma computed in that labelling).
A sorted sigma of orbit 3 (resp. 2) is determined, up to overall sign and the two sign variants with the same
spectrum, by its *pattern*: the sorted position k of the distinguished edge and the sorted positions of the
alpha-pair (|alpha| = 4 for orbit 3; for orbit 2 the two pairs are interchangeable).  outputs/patterns.json maps
each sorted sigma to its pattern (computed exactly from orbits.json here).

Every labelled normal-form tuple (j1; a<=b; c<=d) is evaluated once (grid.py, numpy SVD of the bipartite block),
and credited to every pattern consistent with sorting its five spins (all tie-breaks).  Output per pattern and
2j_max = m: min q and argmin (in sorted spins).
Usage: python survey_sorted.py --orbit 3 --M 30 --workers 8 --out outputs/survey_sorted_orbit3_M30.jsonl
"""
import argparse, itertools, json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import W_matrix
from survey_grid import bip_block, analyse_sv
from orbits import act, sigma

AB = {2: (2, 2), 3: (4, 2)}


def patterns(orbit, orbits_json):
    O = json.load(open(orbits_json))
    allc = [tuple(int(x) for x in k.strip("()").split(",")) for k in O["witnesses"]]
    owner = {}
    for o in O["orbits"]:
        for s in o["sigma"]:
            owner[tuple(s)] = o["orbit"]
    pat = {}
    for e in allc:
        s = sigma(e)
        if owner[s] != orbit:
            continue
        found = set()
        for p in itertools.permutations(range(5)):
            t = sigma(act(e, p))
            if t[3] == 0 and t[1] == t[2] and t[0] != 0 and abs(t[0]) >= abs(t[1]):
                found.add((p[0] + 1, tuple(sorted((p[1] + 1, p[2] + 1)))))
        key = max(s, tuple(-x for x in s))
        pat.setdefault(key, set()).update(found)
    return {k: sorted(v) for k, v in pat.items()}


def compat(J):
    """J = (j1, a, b, c, d) doubled; return set of patterns (k, alpha-pair positions) consistent with sorting."""
    out = set()
    for perm in itertools.permutations(range(5)):
        vals = [J[i] for i in perm]
        if all(vals[i] <= vals[i + 1] for i in range(4)):
            pos = {idx: r + 1 for r, idx in enumerate(perm)}
            out.add((pos[0], tuple(sorted((pos[1], pos[2])))))
    return out


_orb = None


def init(o):
    global _orb
    _orb = o


def work(J):
    al, be = AB[_orb]
    S, W = W_matrix(J, al, be)
    if len(S) < 2:
        return J, None
    B = bip_block(S, W)
    if B.size == 0:
        return J, None
    q, mz, _ = analyse_sv(np.linalg.svd(B, compute_uv=False), len(S))
    return J, q


def tuples(m, orbit):
    pairs = [(p, q) for q in range(1, m + 1) for p in range(1, q + 1)]
    out = []
    for j1 in range(1, m + 1):
        for P in pairs:
            for Q in pairs:
                J = (j1,) + P + Q
                if max(J) == m and sum(J) % 2 == 0:
                    out.append(J)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbit", type=int, default=3)
    ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orbits_json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "orbits.json"))
    a = ap.parse_args()
    pats = patterns(a.orbit, a.orbits_json)
    allpat = sorted({p for v in pats.values() for p in v})
    with open(a.out.replace(".jsonl", "_patterns.json"), "w") as f:
        json.dump({str(k): v for k, v in pats.items()}, f, indent=0)
    print(f"orbit {a.orbit}: {len(pats)} sorted sigma (up to sign), {len(allpat)} patterns", flush=True)
    with open(a.out, "w") as fo, Pool(a.workers, initializer=init, initargs=(a.orbit,)) as pool:
        for m in range(2, a.M + 1):
            t0 = time.time()
            tup = tuples(m, a.orbit)
            best = {}
            for J, q in pool.imap_unordered(work, tup, chunksize=32):
                if q is None:
                    continue
                for p in compat(J):
                    if p not in best or q < best[p][0]:
                        best[p] = (q, sorted(J))
            for p in allpat:
                fo.write(json.dumps(dict(m=m, pattern=[p[0], list(p[1])], q_min=best[p][0] if p in best else None,
                                         argmin_sorted=best[p][1] if p in best else None)) + "\n")
            fo.flush()
            print(m, len(tup), f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

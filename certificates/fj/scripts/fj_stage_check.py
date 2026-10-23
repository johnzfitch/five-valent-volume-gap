"""Checks of the stage machinery (fj_stage):
 [L] fj_stage.link2 (exact) equals the squared weights of fj_core.links (float, validated against [OR]/[LS] builders)
     on all tuples with 2j_i <= 9 and 2j1 in {3,4,5,6,7};
 [R] Theorem 2.1 (ownership inequality): for random tuples, classes and schemes, min over stages of lambda_min(Stage)
     <= sigma_min(B)^2 (float), never violated.
Usage: python fj_stage_check.py > ../outputs/fj_stage_check_out.txt"""
import sys, os, itertools, random
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fj_core as Fc, fj_stage as St, fj_stagecert as SC
cnt = 0; worst = 0.0
for s2 in (3, 4, 5, 6, 7):
    for tj in itertools.product(range(1, 10), repeat=4):
        if (sum(tj) + s2) % 2: continue
        n1, m1 = Fc.pair_nm(tj[0], tj[1]); n2, m2 = Fc.pair_nm(tj[2], tj[3])
        Bd, Cd, lo, up, w2 = Fc.links(s2, n1, m1, n2, m2)
        if Bd.size == 0: continue
        G = St.Geometry(s2, n1, m1, n2, m2)
        for i, j, v in zip(lo, up, w2):
            a = (int(Bd[i]), int(Cd[i])); b = (int(Bd[j]), int(Cd[j]))
            e = float(G.w2(b, a))
            worst = max(worst, abs(e - v) / max(1.0, abs(v))); cnt += 1
print(f"[L] {cnt} links compared, max relative difference {worst:.2e}")
rng = random.Random(7); viol = 0; tests = 0; ratio = []
for _ in range(1500):
    s2 = rng.choice([3, 4, 5, 6, 7])
    t = sorted(rng.randint(1, 16) for _ in range(4))
    if (sum(t) + s2) % 2: t[3] += 1
    lab, p = rng.choice(list(Fc.pairings(t)))
    G = St.Geometry(s2, *p)
    if len(G.sites) < 2: continue
    M, _ = Fc.bipartite(s2, *p, colclass=None)
    for cc in (0, 1):
        Mc, _ = Fc.bipartite(s2, *p, colclass=cc)
        A = Mc.toarray()
        if A.shape[1] == 0: continue
        smin2 = np.linalg.svd(A, compute_uv=False).min() ** 2 if A.shape[0] >= A.shape[1] else 0.0
        for pat in SC.schemes(s2, cc):
            ts = {r: rng.uniform(0.1, 0.99) for r in pat}
            v = SC.stage_values(G, cc, pat, ts)
            tests += 1
            if v > smin2 * (1 + 1e-9) + 1e-9: viol += 1
print(f"[R] {tests} (tuple, class, scheme, random t) tests: min stage value > sigma_min(B)^2 in {viol} cases")

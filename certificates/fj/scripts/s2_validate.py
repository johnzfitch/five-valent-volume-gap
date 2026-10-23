"""Validation of the j1 = 2 argument against exact matrices (float64) and exact stage certificates.

[V1] lemma checks on random W2-type configurations: exact count-form sums vs closed forms, and the count forms vs the
     true sigma_min of the chain matrices; the coupling bound vs the true coupling sup ||M- x||^2/||W- x||^2.
[V2] routing lemma (THEOREM.md Lemma 4.2) on all gap patterns g1,g2,g3 <= 12.
[V3] every W2 tuple at 2j_max = 39..48 and 60, 80: routed bound (implied sigma^2 lower bound) <= true q, and >= target.
[V4] exact rational stage certificates (fj_stage) for every W2 tuple with 39 <= 2j_max <= 44.
Usage: python s2_validate.py > ../outputs/s2_validate_out.txt
"""
import sys, os, math, itertools, random, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fj_core as Fc
import fj_stage as St
from s2_proof import F_wall, s_M_closed, W_C3, star_ok, LAM0, Q0, S_of, T0, H
from multiprocessing import Pool

def P(n, m, L): return (n * n - L * L) * (L * L - m * m)

def wide(n1, m1, n2, m2): return m1 >= 1 and m2 <= m1 + 1 and n2 <= n1 + 1

def wtype(n1, m1, n2, m2):
    sR = (m2 >= m1 + 2) or (m1 == 0 and m2 <= 1); eR = n2 >= n1 + 2
    return ('R' if sR else 'C') + ('R' if eR else 'C')

def blocks(n1, m1, n2, m2):
    """exact (float) chain blocks from the full builder: W- (rows d=-2, cols d=-1), M- (rows 0, cols -1),
    M+ (rows 0, cols +1), W+ (rows +2, cols +1)."""
    M, (Bd, Cd, rows, cols) = Fc.bipartite(4, 2 * n1, 2 * m1, 2 * n2, 2 * m2, colclass=1)
    M = M.toarray(); d = Bd - Cd; dr = d[rows]; dc = d[cols]
    im = np.where(dc == -2)[0]; ip = np.where(dc == 2)[0]
    g = lambda a, idx: M[np.ix_(dr == a, idx)]
    return g(-4, im), g(0, im), g(0, ip), g(4, ip), M

def smin2(A):
    if A.shape[1] == 0: return np.inf
    if A.shape[0] < A.shape[1]: return 0.0
    return np.linalg.svd(A, compute_uv=False).min() ** 2

def routed(n1, m1, n2, m2):
    """returns (regime, implied sigma^2 lower bound) using the CLOSED forms (and exact W for C3)"""
    if wide(n1, m1, n2, m2): return None
    D0, *_ = blocks(n1, m1, n2, m2)
    tW = wtype(n1, m1, n2, m2)
    sWm = 1 / F_wall(n1, m1, n2, m2, tW) if tW != 'CC' else 0
    if not wide(n2, m2, n1, m1):
        tWp = wtype(n2, m2, n1, m1)
        return 'NN', min(sWm, 1 / F_wall(n2, m2, n1, m1, tWp))
    sM = s_M_closed(n1, m1, n2, m2)
    if n2 >= n1 + 2:
        C = 1.5; reg = 'C1'
    else:
        C = 1.5 + W_C3(n1, m1, n2, m2); reg = 'C3'
    # implied bound: max_t min((1-(1/t-1)C) sW, (1-t) sM)
    lo, hi = 1e-12, 1 - 1e-15
    for _ in range(200):
        mid = (lo + hi) / 2
        if (1 - (1 / mid - 1) * C) * sWm < (1 - mid) * sM: lo = mid
        else: hi = mid
    t = (lo + hi) / 2
    return reg, max(0.0, min((1 - (1 / t - 1) * C) * sWm, (1 - t) * sM))

def route_tuple(t):
    """Lemma 4.2: ordered pairing for a sorted doubled tuple t = (2j2..2j5) of the W2 class"""
    a, b, c, d = t
    g1, g2, g3 = (b - a) // 2, (c - b) // 2, (d - c) // 2
    nm = lambda x, y: ((x + y) // 2 + 1, abs(x - y) // 2)
    P1 = nm(a, b) + nm(c, d); P1r = nm(c, d) + nm(a, b); P3r = nm(b, c) + nm(a, d)
    if abs(g1 - g3) <= 1 and g1 + g3 >= 2: return P3r, 'I'
    if (g1, g3) == (0, 0): return P1, 'II00'
    if (g1, g3) == (1, 0): return (P1, 'II10') if g2 >= 1 else (P1r, 'II10b')
    if (g1, g3) == (0, 1): return P1, 'II01'
    if g3 >= g1 + 2: return P1, 'III3'
    return P1r, 'III1'

def regime_of(p):
    n1, m1, n2, m2 = p
    if wide(n1, m1, n2, m2): return 'BAD'
    if not wide(n2, m2, n1, m1): return 'NN'
    if abs(n2 - n1) <= 1: return 'C3'
    sR = (m2 >= m1 + 2) or (m1 == 0 and m2 <= 1)
    return 'C1' if sR else 'C2'

def V2():
    bad = 0; cnt = 0; reg = {}
    for g1 in range(0, 13):
        for g2 in range(0, 13):
            for g3 in range(0, 13):
                for j2 in (2, 3, 10):        # doubled base spin (W2: integer spins 1, half-integer 3/2, large)
                    t = (j2, j2 + 2 * g1, j2 + 2 * g1 + 2 * g2, j2 + 2 * g1 + 2 * g2 + 2 * g3)
                    p, case = route_tuple(t)
                    r = regime_of(p); cnt += 1
                    reg[r] = reg.get(r, 0) + 1
                    if r not in ('NN', 'C1', 'C3'): bad += 1
    print(f"[V2] routing lemma on {cnt} gap patterns: regimes {reg}, {bad} not in NN/C1/C3")

def V1(nsamp=3000, seed=20261021):
    rng = random.Random(seed)
    worst = {}; cnt = 0
    def upd(k, v):
        if k not in worst or v > worst[k][0]: worst[k] = (v, cur)
    while cnt < nsamp:
        n1 = rng.randint(3, 70); m1 = rng.randint(0, n1 - 3)
        n2 = rng.randint(3, 70); m2 = rng.randint(0, n2 - 3)
        if wide(n1, m1, n2, m2): continue
        cur = (n1, m1, n2, m2)
        Wm, Mm, Mp, Wp, B = blocks(n1, m1, n2, m2)
        if Wm.shape[1] == 0 or B.shape[1] == 0: continue
        cnt += 1
        t = wtype(n1, m1, n2, m2)
        s_true = smin2(Wm)
        upd('closed-form sigma_W^2 / true (must be <=1)', (1 / F_wall(n1, m1, n2, m2, t)) / s_true)
        if wide(n2, m2, n1, m1) and Mp.shape[1] > 0 and (m2 >= m1 + 2 or (m1, m2) == (0, 1)):
            sMt = smin2(Mp)
            upd('closed-form sigma_M^2 / true (must be <=1)', s_M_closed(n1, m1, n2, m2) / sMt)
            Ctrue = np.linalg.norm(Mm @ np.linalg.pinv(Wm), 2) ** 2
            if n2 >= n1 + 2:
                upd('true coupling / 1.5 (C1, must be <=1)', Ctrue / 1.5)
            elif abs(n2 - n1) <= 1:
                upd('true coupling / (1.5+W_C3) (C3, must be <=1)', Ctrue / (1.5 + W_C3(n1, m1, n2, m2)))
    print(f"[V1] {cnt} random configurations (n<=70):")
    for k, (v, c) in worst.items():
        print(f"     max {k}: {v:.6f} at {c}  {'OK' if v <= 1 + 1e-9 else 'VIOLATION'}")

def one_tuple(t):
    p, case = route_tuple(t)
    r = routed(*p)
    q, ker, _ = Fc.qmin_tuple(4, *Fc.pair_nm(t[0], t[1]), *Fc.pair_nm(t[2], t[3]))
    return t, p, case, r, q

def V3(levels):
    for M in levels:
        tups = [t for t in itertools.combinations_with_replacement(range(1, M + 1), 4)
                if t[3] == M and sum(t) % 2 == 0 and len(set(x % 2 for x in t)) == 1 and not (t[0] % 2 == 1 and t[0] < 3)]
        with Pool(8) as pool: res = pool.map(one_tuple, tups, chunksize=8)
        res = [x for x in res if x[4] is not None]
        below = [x for x in res if x[3] is None or 8 * math.sqrt(x[3][1]) < Q0]
        above = [x for x in res if x[3] is not None and 8 * math.sqrt(x[3][1]) > x[4] * (1 + 1e-9)]
        regs = {}
        for x in res: regs[x[3][0] if x[3] else None] = regs.get(x[3][0] if x[3] else None, 0) + 1
        mn = min(res, key=lambda x: 8 * math.sqrt(x[3][1]) if x[3] else -1)
        print(f"[V3] 2j_max={M}: {len(res)} W2 tuples, regimes {regs}; below target {len(below)}; bound above true q {len(above)}; "
              f"min bound 8 sigma = {8*math.sqrt(mn[3][1]):.3f} at {mn[0]} ({mn[3][0]}, true q {mn[4]:.2f})", flush=True)

def stage_cert(t):
    """exact rational stage certificate for the routed pairing (independent of the closed forms)"""
    p, case = route_tuple(t)
    n1, m1, n2, m2 = p
    G = St.Geometry(4, 2 * n1, 2 * m1, 2 * n2, 2 * m2)
    reg = regime_of(p)
    cands = [{0: 'drop'}] if reg == 'NN' else []
    for tt in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99):
        cands.append({0: ('hi', St.fmpq(int(round(tt * 1000)), 1000))})     # x+ owns the middle rows
    for sch in cands:
        comp = St.stage_data(G, 1, sch)
        ok = True
        for d in comp:
            if not G.diags.get(d): continue
            dg, o2 = St.stage_matrix_exact(G, d, comp[d], St.LAM0)
            neg, zero = St.ldl_neg(dg, o2)
            if neg or zero: ok = False; break
        if ok: return True, sch
    return False, None

def V4(levels):
    for M in levels:
        tups = [t for t in itertools.combinations_with_replacement(range(1, M + 1), 4)
                if t[3] == M and sum(t) % 2 == 0 and len(set(x % 2 for x in t)) == 1 and not (t[0] % 2 == 1 and t[0] < 3)]
        tups = [t for t in tups if Fc.region(4, *Fc.pair_nm(t[0], t[1]), *Fc.pair_nm(t[2], t[3]))[0].size > 1]
        t0 = time.time()
        with Pool(8) as pool: res = pool.map(stage_cert, tups, chunksize=4)
        nf = sum(1 for ok, _ in res if not ok)
        print(f"[V4] 2j_max={M}: exact stage certificates at q0={Q0} for {len(tups)} W2 tuples (routed pairing): {len(tups)-nf} certified, {nf} not  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == '__main__':
    t00 = time.time()
    V2()
    V1()
    V3([39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 60, 80])
    V4([39, 40, 41, 42, 43, 44])
    print(f"total {time.time()-t00:.0f}s")

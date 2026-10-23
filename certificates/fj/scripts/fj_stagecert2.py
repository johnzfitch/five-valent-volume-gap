"""Version 2 (opt-in; fj_stagecert.py is the producing source of the j1=5/2 run 39..70 and is unchanged).
Changes: t may approach 1 (cap 1-1e-9, 60 bisection steps, finer grid near 1); t is certified exactly at its
binary64 value (no rounding to 1e-6, which broke ill-conditioned fat tuples); the printed bound is labelled.

Exact stage certificates for any j1 (THEOREM.md section 2, Theorem 2.3), all tuples of given levels.

For a tuple (doubled spins), we try ordered pairings x colour classes x "ownership schemes" (Theorem 2.3).
Schemes: column diagonals of the class are numbered from one wall; every row diagonal touching two column
diagonals is either dropped or owned by one of them with a parameter t.  For half-integer j1 we use the forced
triangular chain from the C'-wall (and its mirror); for integer j1 we use every assignment in which each column
diagonal owns at least one row diagonal, dropping the rest.  The parameters t are chosen by a float search
(backward greedy for chains), rounded to rationals, and the stage matrices are certified by exact LDL^T.
Usage: python fj_stagecert.py S2 Mlo Mhi [q0] > out   (S2 = 2 j1)
"""
import sys, os, itertools, time, math
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from flint import fmpq
import fj_stage as St
import fj_core as Fc
from multiprocessing import Pool


def schemes(s2, cc):
    """yield (description, list of (row_d, owner_side)) for interior row diagonals; owner_side 'lo'/'hi'/'drop'"""
    cds, rds = St.column_classes(s2, cc)
    inter = [r for r in rds if (r - 2) in cds and (r + 2) in cds]
    for pat in itertools.product(('lo', 'hi', 'drop'), repeat=len(inter)):
        own = {d: 0 for d in cds}
        for r in rds:
            nb = [d for d in (r - 2, r + 2) if d in cds]
            if len(nb) == 1: own[nb[0]] += 1
        for r, a in zip(inter, pat):
            if a == 'lo': own[r - 2] += 1
            elif a == 'hi': own[r + 2] += 1
        if min(own.values()) == 0: continue
        yield dict(zip(inter, pat))


def fgram(G, col_d, row_d):
    """cached float Gram (diag, off) of row diagonal row_d on column diagonal col_d"""
    if not hasattr(G, '_fg'): G._fg = {}
    key = (col_d, row_d)
    if key not in G._fg:
        _, dg, o2 = G.gram(col_d, row_d)
        G._fg[key] = (np.array([float(x) for x in dg]), np.sqrt(np.array([float(x) for x in o2])))
    return G._fg[key]


def stage_values(G, cc, pat, ts):
    sch = {r: ('drop' if a == 'drop' else (a, ts[r])) for r, a in pat.items()}
    comp = St.stage_data(G, cc, sch)
    vals = []
    for d in comp:
        cols = G.diags.get(d)
        if not cols: continue
        n = len(cols); dg = np.zeros(n); off = np.zeros(max(n - 1, 0))
        for coef, r in comp[d]:
            a, o = fgram(G, d, r)
            dg += float(coef) * a; off += float(coef) * o
        vals.append(St.lam_min_tri(dg, off))
    return min(vals) if vals else np.inf


def best_ts(G, cc, pat, lam):
    """coordinate search over t in (0,1) for each owned interior row diagonal, maximising the min stage value"""
    keys = [r for r, a in pat.items() if a != 'drop']
    ts = {r: 0.8 for r in keys}
    best = stage_values(G, cc, pat, ts)
    grid = [0.3, 0.5, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.97, 0.985, 0.993, 0.997, 0.999, 0.9997, 0.9999, 0.99997, 0.99999, 0.999997]
    for sweep in range(3):
        for r in keys:
            for g in grid:
                tt = dict(ts); tt[r] = g
                v = stage_values(G, cc, pat, tt)
                if v > best: best, ts = v, tt
    return best, ts


def chain_backward(G, cc, pat, target=None, wall=None):
    """parameters for a single chain (half-integer j1) by a backward pass at the target value.
    Column diagonals are ordered from the C'-wall; column k owns the row diagonal between k-1 and k."""
    s2 = G.s2
    cds, rds = St.column_classes(s2, cc)
    if wall is None: wall = [w for w in (-s2, s2) if w in rds][0]
    order = sorted(cds, key=lambda d: abs(d - wall))
    own_row = {order[k]: (order[k - 1] + order[k]) // 2 for k in range(1, len(order))}
    keys = [r for r, a in pat.items() if a != 'drop']
    def stage_val(d, ts):
        sch = {r: ('drop' if a == 'drop' else (a, ts.get(r, 0.5))) for r, a in pat.items()}
        comp = St.stage_data(G, cc, sch)
        cols = G.diags.get(d)
        if not cols: return np.inf
        n = len(cols); dg = np.zeros(n); off = np.zeros(max(n - 1, 0))
        for coef, r in comp[d]:
            a, o = fgram(G, d, r); dg += float(coef) * a; off += float(coef) * o
        return St.lam_min_tri(dg, off)
    lam = target
    ts = {r: 0.999 for r in keys}
    for k in range(len(order) - 1, 0, -1):
        d = order[k]; r = own_row[d]
        if r not in ts: continue
        # stage d depends on t_r (own, decreasing in t) and on t of the next owned row (already fixed)
        if stage_val(d, {**ts, r: 1e-6}) < lam: return 0.0, None
        lo, hi = 1e-6, 1 - 1e-9
        for _ in range(60):
            mid = (lo + hi) / 2
            if stage_val(d, {**ts, r: mid}) >= lam: lo = mid
            else: hi = mid
        ts[r] = lo
    if stage_val(order[0], ts) < lam: return 0.0, None
    return lam, ts

def certify_exact(G, cc, pat, ts, lam):
    sch = {r: ('drop' if a == 'drop' else (a, fmpq(*ts[r].as_integer_ratio()))) for r, a in pat.items()}
    comp = St.stage_data(G, cc, sch)
    for d in comp:
        if not G.diags.get(d): continue
        dg, o2 = St.stage_matrix_exact(G, d, comp[d], lam)
        neg, zero = St.ldl_neg(dg, o2)
        if neg or zero: return False
    return True


def cert_tuple(args):
    s2, t, q0 = args
    lam = (fmpq(int(round(q0 * 10 ** 7)), 10 ** 7) / 8) ** 2
    lamf = float(lam)
    tried = []
    for lab, p in Fc.pairings(t):
        G = St.Geometry(s2, *p)
        if len(G.sites) == 0: continue
        for cc in (0, 1):
            for pat in schemes(s2, cc):
                if s2 % 2 == 1:
                    try:
                        v, ts = chain_backward(G, cc, pat, target=lamf * 1.02)
                    except Exception:
                        v, ts = 0.0, None
                    if ts is None: v, ts = best_ts(G, cc, pat, lamf)
                    else: ts = {r: float(ts[r]) for r in ts}
                else:
                    v, ts = 0.0, None
                    for wl in (-s2, s2):          # integer j1: try the backward chain from either wall first
                        try:
                            v, ts = chain_backward(G, cc, pat, target=lamf * 1.02, wall=wl)
                        except Exception:
                            v, ts = 0.0, None
                        if ts is not None: break
                    if ts is None: v, ts = best_ts(G, cc, pat, lamf)
                    else: ts = {r: float(ts[r]) for r in ts}
                tried.append((v, lab, cc, pat, ts, G))
    tried.sort(key=lambda x: -x[0])
    for v, lab, cc, pat, ts, G in tried[:6]:
        if v <= lamf: break
        if certify_exact(G, cc, pat, ts, lam):
            return t, True, 8 * math.sqrt(max(v, 0)), lab, cc
    return t, False, (8 * math.sqrt(max(tried[0][0], 0)) if tried else None), None, None


def tuples(s2, M):
    out = []
    for t in itertools.combinations_with_replacement(range(1, M + 1), 4):
        if t[3] != M or (sum(t) + s2) % 2: continue
        if Fc.region(s2, *Fc.pair_nm(t[0], t[1]), *Fc.pair_nm(t[2], t[3]))[0].size < 2: continue
        out.append(t)
    return out


if __name__ == '__main__':
    s2, Mlo, Mhi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    q0 = float(sys.argv[4]) if len(sys.argv) > 4 else 4.5579607
    only_w2 = (s2 % 2 == 0) and ('--w2' in sys.argv)
    tot = 0; fails = []
    for M in range(Mlo, Mhi + 1):
        T = tuples(s2, M)
        if only_w2:
            T = [t for t in T if len(set(x % 2 for x in t)) == 1 and not (t[0] % 2 == 1 and t[0] < 3)]
        t0 = time.time()
        with Pool(8) as pool:
            res = pool.map(cert_tuple, [(s2, t, q0) for t in T], chunksize=4)
        f = [r for r in res if not r[1]]
        fails += f; tot += len(res)
        mn = min((r[2] for r in res if r[1]), default=None)
        print(f"2j1={s2} 2j_max={M}: {len(res)} tuples, {len(res)-len(f)} certified at q0={q0}, {len(f)} not; "
              f"max float stage value found (chains: the search target 1.02*lam0) {mn if mn is None else round(mn,3)}  ({time.time()-t0:.0f}s)"
              + (f"  uncertified: {[r[0] for r in f][:8]}" if f else ''), flush=True)
    print(f"TOTAL 2j1={s2} levels {Mlo}..{Mhi}: {tot} tuples, {len(fails)} uncertified")

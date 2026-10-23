"""Checks of [FJ] (j1 = 2 closed; j1 = 5/2 to 2j_max = 70) with independent code.

[L] Lemma 4.3 (a),(b),(d) exactly against the Racah operator rb_core (links of W-, M+ and the Lambda identity on the
    columns of diagonal -1, with signs), all j1 = 2 tuples with legs 2j <= MAXL.
[R] Lemma 4.4 routing on all W2 tuples with legs 2j <= MAXR: from rb_core's own site sets, the routed pairing has
    W- non-wide, the stated regime (NN / C1 / C3) holds, and n2 >= 22 (C1) / min(n1,n2) >= 21 (C3) once 2j_max >= 39.
[V] On random W2 tuples (legs up to 2j <= 160) in the routed pairing, with matrices from rb_core:
    Lemma 4.6 closed forms <= true sigma_min^2 of W- and M+;  Lemma 4.7: C <= 3/2 (C1), C <= 3/2 + W (C3);
    the Lemma 4.8 conclusion  sigma_min(B)^2 >= lam0 on the class {d = +-1}; and W <= Wbar (tail [T3]) on C3 configurations.
[K] j1 = 5/2 tuple counts per level 2j_max = 39 and 70 (Theorem 5.1 reports 4,729 and 25,827).
Usage: python rb_fj.py MAXL MAXR NV > ../outputs/rb_fj_out.txt
"""
import sys, os, math, time, random, itertools
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C

LAM0 = (4.5579607 / 8) ** 2


def Pf(n, m, L): return (n * n - L * L) * (L * L - m * m)


def links_of(tj):
    S, ent = C.R_entries(tj)
    sc = C.scale_sq(tj[0])
    L = {}
    for (i, j), (sg, sq) in ent.items():
        L[frozenset((S[i], S[j]))] = (sg, sq * sc)
    return S, L


def lk(L, p1, p2):
    st = lambda b, c: (int(2 * b), int(2 * c))
    return L.get(frozenset((st(*p1), st(*p2))), (0, Fr(0)))


def partL(MAXL):
    bad = 0; badid = 0; badsg = 0; n = 0
    for tj in itertools.product([4], *[range(1, MAXL + 1)] * 4):
        if sum(tj) % 2:
            continue
        S, L = links_of(tj)
        if len(S) < 2:
            continue
        n += 1
        a, b, c, d = (Fr(x, 2) for x in tj[1:])
        n1, m1, n2, m2 = a + b + 1, abs(a - b), c + d + 1, abs(c - d)
        P = lambda x: Pf(n1, m1, x); Q = lambda x: Pf(n2, m2, x)
        Sset = set((Fr(B, 2), Fr(Cc, 2)) for B, Cc in S)
        for (bb, cc) in Sset:
            if cc == bb + 2:          # row of W-: (b, b+2)
                if (bb, bb + 1) in Sset:
                    v = lk(L, (bb, bb + 2), (bb, bb + 1))[1]; bad += v != Q(bb + 2) * bb / (2 * (2 * bb + 3))
                if (bb + 1, bb + 2) in Sset:
                    v = lk(L, (bb, bb + 2), (bb + 1, bb + 2))[1]; bad += v != P(bb + 1) * (bb + 3) / (2 * (2 * bb + 3))
            if cc == bb:              # row (e,e) of M+
                e = bb
                if (e, e - 1) in Sset:
                    v = lk(L, (e, e), (e, e - 1))[1]; bad += v != Fr(3, 4) * Q(e) * (e - 1) * (2 * e + 3) / ((2 * e - 1) * (2 * e + 1))
                if (e + 1, e) in Sset:
                    v = lk(L, (e, e), (e + 1, e))[1]; bad += v != Fr(3, 4) * P(e + 1) * (e + 2) * (2 * e - 1) / ((2 * e + 1) * (2 * e + 3))
        # (d) on columns (c, c+1) of diagonal -1
        cols = sorted(bb for (bb, cc) in Sset if cc == bb + 1)
        for cb in cols:
            # W-: rows (cb, cb+2) [q link] and (cb-1, cb+1) [p link]; M-: rows (cb, cb) [D link? (cb,cb)-(cb,cb+1)] and (cb+1, cb+1) [X link]
            w_diag = lk(L, (cb, cb + 1), (cb, cb + 2))[1] + lk(L, (cb, cb + 1), (cb - 1, cb + 1))[1]
            m_diag = lk(L, (cb, cb + 1), (cb, cb))[1] + lk(L, (cb, cb + 1), (cb + 1, cb + 1))[1]
            K = cb * (2 * cb * cb + 6 * cb + 5)
            N = (n1 * n1 + m1 * m1 - n2 * n2 - m2 * m2) * K + 2 * n1 * n1 * m1 * m1 * (cb + 3) - 2 * n2 * n2 * m2 * m2 * (cb - 1) \
                - 2 * (n2 * n2 + m2 * m2) + 2 * (2 * cb * cb + 4 * cb + 1) ** 2
            badid += (m_diag - Fr(3, 2) * w_diag) != 3 * N / (4 * (2 * cb + 1) * (2 * cb + 3))
            if cb + 1 in cols:
                w1 = lk(L, (cb, cb + 1), (cb, cb + 2)); w2 = lk(L, (cb + 1, cb + 2), (cb, cb + 2))         # shared row (cb, cb+2)
                m1_ = lk(L, (cb, cb + 1), (cb + 1, cb + 1)); m2_ = lk(L, (cb + 1, cb + 2), (cb + 1, cb + 1))  # shared row (cb+1, cb+1)
                if w1[1] and w2[1]:
                    badid += m1_[1] * m2_[1] != Fr(9, 4) * w1[1] * w2[1]
                    badsg += m1_[0] * m2_[0] != w1[0] * w2[0]
    print(f"[L] Lemma 4.3 exact vs Racah: {n} j1=2 tuples (legs 2j<={MAXL}); link mismatches {bad}, Lambda/proportionality mismatches {badid}, sign mismatches {badsg}")


def w2(legs):
    return all(x % 2 == 0 for x in legs) or all(x % 2 == 1 and x >= 3 for x in legs)


def route(sp):
    """[FJ] Lemma 4.4 on sorted doubled legs; returns (name, ordered doubled pairing (A,B|C,D))"""
    a, b, c, d = sp
    g1, g2, g3 = (b - a) // 2, (c - b) // 2, (d - c) // 2
    P1, P1p, P3p = (a, b, c, d), (c, d, a, b), (b, c, a, d)
    if abs(g1 - g3) <= 1 and g1 + g3 >= 2: return 'C3', P3p
    if (g1, g3) == (0, 0): return 'NN', P1
    if (g1, g3) == (1, 0): return ('NN', P1) if g2 >= 1 else ('C3', P1p)
    if (g1, g3) == (0, 1): return ('C3', P1) if g2 == 0 else ('C1', P1)
    if g3 >= g1 + 2: return 'C1', P1
    if g1 >= g3 + 2: return 'NN', P1p
    raise ValueError(sp)


def diag_lists(tj):
    S = C.sites_of(tj)
    z = {}
    for (B, Cc) in S:
        z.setdefault((B - Cc) // 2, []).append(Fr(B, 2))
    return {k: sorted(v) for k, v in z.items()}


def partR(MAXR):
    cnt = bad = 0; per = {}; nwp = nwp_empty = 0
    for sp in itertools.combinations_with_replacement(range(1, MAXR + 1), 4):
        if not w2(sp) or (4 + sum(sp)) % 2:
            continue
        name, pr = route(sp)
        tj = (4,) + pr
        if C.dim_inv(tj) < 2:
            continue
        cnt += 1; per[name] = per.get(name, 0) + 1
        z = diag_lists(tj)
        wide_m = len(z.get(-1, [])) > len(z.get(-2, []))
        wide_p = len(z.get(1, [])) > len(z.get(2, []))
        a, b, c, d = (Fr(x, 2) for x in pr)
        n1, m1, n2, m2 = a + b + 1, abs(a - b), c + d + 1, abs(c - d)
        mc = (m2 >= m1 + 2) or (m1, m2) == (0, 1)
        # critical: what the proof of Thm 4.2 uses (W- non-wide; W+ non-wide in NN; the n/m conditions)
        if name == 'NN': ok = not wide_m and not wide_p
        elif name == 'C1': ok = (not wide_m) and n2 >= n1 + 2 and mc
        else: ok = (not wide_m) and abs(n2 - n1) <= 1 and mc
        if max(sp) >= 39:
            if name == 'C1': ok &= n2 >= 22
            if name == 'C3': ok &= min(n1, n2) >= 21
        bad += not ok
        if not ok and bad < 5: print('   routing failure', sp, name, pr)
        if name in ('C1', 'C3') and not wide_p:
            nwp += 1
            if not (z.get(1) or z.get(2)): nwp_empty += 1
    print(f"[R] Lemma 4.4 routing on {cnt} W2 tuples (legs 2j<={MAXR}): failures of the conditions the proof uses {bad}; per regime {per}")
    print(f"    C1/C3 tuples where W+ is not actually wide (statement 'W+ wide' false, unused by the proof): {nwp}, of which W+ empty: {nwp_empty}")


def mats(tj):
    """float blocks W-, M-, M+, W+ from rb_core moduli (nonnegative gauge, validated by rb_links) on the class d = +-1"""
    S, L = links_of(tj)
    Sset = set((Fr(B, 2), Fr(Cc, 2)) for B, Cc in S)
    rows = {k: sorted(p for p in Sset if p[0] - p[1] == k) for k in (-2, 0, 2)}
    cols = {k: sorted(p for p in Sset if p[0] - p[1] == k) for k in (-1, 1)}
    def blk(rk, ck):
        R_, C_ = rows[rk], cols[ck]
        A = np.zeros((len(R_), len(C_)))
        for i, r in enumerate(R_):
            for j, c in enumerate(C_):
                v = lk(L, r, c)[1]
                if v: A[i, j] = float(v) ** 0.5
        return A
    return blk(-2, -1), blk(0, -1), blk(0, 1), blk(2, 1), blk


def smin2(A):
    if A.shape[1] == 0: return float('inf')
    if A.shape[0] < A.shape[1]: return 0.0
    return np.linalg.svd(A, compute_uv=False).min() ** 2


def S_of(m): return math.pi ** 2 / 6 if m == 0 else sum(1 / i for i in range(1, int(2 * m) + 1)) / (2 * m)
def Hn(n): return sum(1 / i for i in range(1, int(n) + 1))
T0 = 2.2118984606


def sW_closed(n1, m1, n2, m2):
    startR = (m2 >= m1 + 2) or (m1 == 0 and m2 <= 1); endR = n2 >= n1 + 2
    if startR and not endR: return (n1 + m1 + 1) / (4 * S_of(m1)), 'RC'
    if not startR and endR: return n2 / (4 * math.pi ** 2 / 6), 'CR'
    if startR and endR: return 1 / max(4 * S_of(m1) / (n1 + m1 + 1), 1.25 * Hn(n2 - 3) / n2), 'RR'
    return None, 'CC'


def sM_closed(n1, m1, n2, m2):
    fl = (4 / 3) * T0 / (n1 + m2 + 1)
    fu = (40 / 7) * Hn(n2) / (n2 * (n2 + 1)) if n2 >= n1 else 0
    return 1 / max(fl, fu)


def W_C3(n1, m1, n2, m2):
    dn = n2 - n1; k0 = max(m1, m2 - 2, 0); c_lo, c_hi = k0 + 1, n2 - 2
    W = 0.0; acc = 0.0
    ks = list(range(int(k0), int(c_hi)))
    A = []
    for k in ks:
        L_ = k + 1
        p2 = (n1 * n1 - L_ * L_) * (L_ * L_ - m1 * m1) * (L_ + 2) / (2 * (2 * L_ + 1))
        g = {-1: (n2 - k - 1) * (n2 - k - 2), 0: (n2 - k - 2) * (n2 + k + 2), 1: (n2 + k + 1) * (n2 + k + 2)}[dn]
        acc += (k + 1) / (g * p2); A.append(acc)
    for cc in range(int(c_lo), int(c_hi) + 1):
        Fc = {-1: (n2 - cc) * (n2 - cc - 1), 0: (n2 - cc - 1) * (n2 + cc + 1), 1: (n2 + cc) * (n2 + cc + 1)}[dn]
        Rc = Fc / cc * A[cc - 1 - int(k0)]
        K = cc * (2 * cc * cc + 6 * cc + 5)
        N = (n1 * n1 + m1 * m1 - n2 * n2 - m2 * m2) * K + 2 * n1 * n1 * m1 * m1 * (cc + 3) - 2 * n2 * n2 * m2 * m2 * (cc - 1) \
            - 2 * (n2 * n2 + m2 * m2) + 2 * (2 * cc * cc + 4 * cc + 1) ** 2
        W += 3 * max(N, 0) / (4 * (2 * cc + 1) * (2 * cc + 3)) * Rc
    return W


def Wbar(n):
    H = Hn(n)
    return max(12 * H * (6 + (2 * H + 5.6) / n), 24 * H * (1 + (2 * H + 5.6) / n), 48 * H * (1 + (2 * H + 5.6) / n))


def partV(NV):
    rng = random.Random(5)
    st = {'NN': 0, 'C1': 0, 'C3': 0}; viol = []; minratio = {'W': 9, 'M': 9}; worstC = {'C1': 0, 'C3': 0}; lemma48 = 0; tbad = 0
    tried = 0
    while sum(st.values()) < NV and tried < 50 * NV:
        tried += 1
        par = rng.choice([0, 1])
        legs = sorted(2 * rng.randint(1, 80) + (1 if par else 0) for _ in range(4))
        if par and min(legs) < 3: continue
        # bias toward the C1 / C3 regimes
        if rng.random() < 0.6:
            base = rng.randint(1, 60) * 2 + par; g = [rng.randint(0, 3) for _ in range(3)]
            legs = [base, base + 2 * g[0], base + 2 * g[0] + 2 * g[1], base + 2 * (g[0] + g[1] + g[2])]
            if par and legs[0] < 3: continue
        sp = tuple(legs)
        if (4 + sum(sp)) % 2: continue
        name, pr = route(sp)
        tj = (4,) + pr
        if C.dim_inv(tj) < 2: continue
        st[name] += 1
        Wm, Mm, Mp, Wp, blk = mats(tj)
        a, b, c, d = (x / 2 for x in pr)
        n1, m1, n2, m2 = a + b + 1, abs(a - b), c + d + 1, abs(c - d)
        sw, typ = sW_closed(n1, m1, n2, m2)
        true_w = smin2(Wm)
        if sw is None or sw > true_w * (1 + 1e-10): viol.append(('sW', sp, typ, sw, true_w))
        else: minratio['W'] = min(minratio['W'], true_w / sw)
        if name in ('C1', 'C3'):
            sm = sM_closed(n1, m1, n2, m2); true_m = smin2(Mp)
            if sm > true_m * (1 + 1e-10): viol.append(('sM', sp, sm, true_m))
            else: minratio['M'] = min(minratio['M'], true_m / sm)
            # true coupling C = sup |M- x|^2/|W- x|^2
            G = Wm.T @ Wm; Hh = Mm.T @ Mm
            w, V = np.linalg.eigh(G)
            Gi = V @ np.diag(w ** -0.5) @ V.T
            Ctrue = np.linalg.eigvalsh(Gi @ Hh @ Gi).max()
            Cbar = 1.5 if name == 'C1' else 1.5 + W_C3(n1, m1, n2, m2)
            if Ctrue > Cbar * (1 + 1e-9): viol.append(('C', sp, name, Ctrue, Cbar))
            worstC[name] = max(worstC[name], Ctrue / Cbar)
            if name == 'C3' and W_C3(n1, m1, n2, m2) > Wbar(n1): tbad += 1
        # sigma_min(B)^2 on class d=+-1 vs lam0
        Bfull = np.block([[Wm, np.zeros((Wm.shape[0], Wp.shape[1]))], [Mm, Mp], [np.zeros((Wp.shape[0], Wm.shape[1])), Wp]])
        if max(sp) >= 39 and smin2(Bfull) < LAM0: lemma48 += 1
    print(f"[V] {sum(st.values())} random W2 tuples (legs <= 160) per regime {st}:")
    print(f"    closed-form violations (Lemma 4.6 sW, sM; Lemma 4.7 C): {len(viol)} {viol[:4]}")
    print(f"    min true/closed: sigma_min(W-)^2 {minratio['W']:.3f}, sigma_min(M+)^2 {minratio['M']:.3f};  max true C / bound: C1 {worstC['C1']:.4f}, C3 {worstC['C3']:.4f}")
    print(f"    tuples with 2j_max >= 39 and sigma_min(B)^2 < lam0 on {{d=+-1}}: {lemma48};  C3 configurations with W > Wbar(n1): {tbad}")


def partK():
    for lev in (39, 70):
        cnt = 0
        for sp in itertools.combinations_with_replacement(range(1, lev + 1), 3):
            tj = (5,) + sp + (lev,)
            if sum(tj) % 2: continue
            if C.dim_inv(tj) >= 2: cnt += 1
        print(f"[K] j1=5/2, level 2j_max={lev}: {cnt} tuples with >= 2 sites")


if __name__ == '__main__':
    t0 = time.time()
    MAXL, MAXR, NV = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    partL(MAXL); print(f"   ({time.time()-t0:.0f}s)", flush=True)
    partR(MAXR); print(f"   ({time.time()-t0:.0f}s)", flush=True)
    partV(NV); print(f"   ({time.time()-t0:.0f}s)", flush=True)
    partK(); print(f"total {time.time()-t0:.0f}s")

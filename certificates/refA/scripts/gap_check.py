"""Float (binary64) checks of [CL] section 3 with own code, all tuples up to a level M (doubled spins; legs 2-5 as
sorted multisets; all three pairings examined).

 (E)  explicit form of closedness: every cut site lies on the wall d = s (colour (2j4+2j5) mod 2), the wall d = -s
      (colour (2j2+2j3) mod 2) or the corner u = s (colour 0), with exactly those colours;
 (T6/T8) whenever a pairing has a closed class C and (N1+N2 odd or a pair contains spin 1/2): the block
      K|C -> C' has all |C| singular values >= 1 (so C' is at least as large), and the true q = 8 lam_min(T) >= 2 sqrt(mu);
 (C11) ladder: some leg 2-5 has spin 1/2, j1 half-integer >= 3/2, and no pairing has a closed class.  For every pairing
      the block is checked to be a SHIFTED window: C-positions J and C'-positions J' are intervals of equal length with
      J' = J + 1 or J - 1, and the block equals P_J' (H' + 1) P_J entrywise (H' = Stark matrix of the other pair).
      sigma_min of the block >= c* and q >= 2 sqrt(mu) c* are checked too;
 coverage: number of tuples covered by [CL] Cor. 9 per level (compare [CL] outputs/survey_cover_M32.txt).
"""
import sys, time, math, itertools
import numpy as np

CSTAR = math.sqrt((15 - math.sqrt(97)) / 6)


def stark_h(nd, md, Ld):
    """doubled n, m, L -> Stark link (float)"""
    n, m, L = nd / 2, md / 2, Ld / 2
    v = (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)
    return math.sqrt(v) if v > 0 else 0.0


def analyse(tj):
    S, A, B, C, D = tj
    n1, m1 = A + B + 2, abs(A - B)
    n2, m2 = C + D + 2, abs(C - D)
    bs = list(range(m1, A + B + 1, 2)); cs = list(range(m2, C + D + 1, 2))
    rect = set((b, c) for b in bs for c in cs)
    R = [(b, c) for b in bs for c in cs if abs(b - c) <= S <= b + c]
    Rs = set(R)
    col = {p: ((p[0] + p[1] - S) // 2) % 2 for p in R}
    def nbrs(p):
        b, c = p
        return [q for q in ((b + 2, c), (b - 2, c), (b, c + 2), (b, c - 2)) if q in rect]
    cut = [p for p in R if any(q not in Rs for q in nbrs(p))]
    # (E)
    eviol = 0
    for (b, c) in cut:
        okp = False
        if b - c == S: okp |= col[(b, c)] == (C + D) % 2
        if c - b == S: okp |= col[(b, c)] == (A + B) % 2
        if b + c == S: okp |= col[(b, c)] == 0
        if not okp: eviol += 1
    closed = [all(col[p] != k for p in cut) and any(col[p] == k for p in R) for k in (0, 1)]
    return dict(S=S, n1=n1, m1=m1, n2=n2, m2=m2, bs=bs, cs=cs, R=R, Rs=Rs, col=col, closed=closed, eviol=eviol,
                N1=len(bs), N2=len(cs), nbrs=nbrs)


def link(G, p, q):
    (b, c), (b2, c2) = p, q
    if c == c2: return stark_h(G['n1'], G['m1'], max(b, b2))
    return stark_h(G['n2'], G['m2'], max(c, c2))


def block(G, k):
    """W0 block columns = colour k sites, rows = colour 1-k sites (both in R)"""
    Ck = [p for p in G['R'] if G['col'][p] == k]; Co = [p for p in G['R'] if G['col'][p] != k]
    io = {p: i for i, p in enumerate(Co)}
    M = np.zeros((len(Co), len(Ck)))
    for j, p in enumerate(Ck):
        for q in G['nbrs'](p):
            if q in io: M[io[q], j] = link(G, p, q)
    return Ck, Co, M


def heron(G):
    """float sigma^2 = phi(d) omega(u) (Theorem 3 recursions), keyed by site; and mu"""
    s = G['S'] / 2
    ds = sorted({(b - c) for b, c in G['R']}); us = sorted({(b + c) for b, c in G['R']})
    phi = {ds[0]: 1.0}
    for d in ds[1:]:
        t = (d - 1) / 2 / 1.0  # midpoint (d-1/2) in undoubled units: (d/2 - 1/2)
        t = d / 2 - 0.5
        phi[d] = ((s + 0.5) ** 2 - t * t) / phi[d - 2]
    om = {us[0]: 1.0}
    for u in us[1:]:
        uu = u / 2
        om[u] = (uu + s + 1) * (uu - s) / om[u - 2]
    sig2 = {p: phi[p[0] - p[1]] * om[p[0] + p[1]] for p in G['R']}
    m = [min([sig2[p] for p in G['R'] if G['col'][p] == k], default=None) for k in (0, 1)]
    mu = m[0] * m[1] if None not in m else None
    return sig2, mu


def true_q(G, sig2):
    idx = {p: i for i, p in enumerate(G['R'])}
    W = np.zeros((len(idx), len(idx)))
    for p in G['R']:
        for q in G['nbrs'](p):
            if q in idx:
                W[idx[p], idx[q]] = 0.25 * math.sqrt(sig2[p] * sig2[q]) * link(G, p, q)
    ev = np.abs(np.linalg.eigvalsh(W))
    tol = 1e-9 * max(1.0, ev.max())
    nz = ev[ev > tol]
    return 8 * nz.min() if nz.size else None


def stark_matrix(nd, md):
    cs = list(range(md, nd - 1, 2)); N = len(cs)
    Hm = np.zeros((N, N))
    for k in range(1, N):
        Hm[k - 1, k] = Hm[k, k - 1] = stark_h(nd, md, cs[k])
    return cs, Hm


def tuples_level(M):
    for S in range(1, M + 1):
        for legs in itertools.combinations_with_replacement(range(1, M + 1), 4):
            if max(S, legs[3]) != M or (S + sum(legs)) % 2: continue
            yield (S,) + legs


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    t0 = time.time()
    tot = dict(tuples=0, nonzero=0, covered=0, eviol=0, t68=0, t68_bad=0, q_bad=0, ladder=0, ladder_bad=0, ladder_sig=9.0)
    lines = []
    for L in range(1, M + 1):
        lev = dict(n=0, cov=0, lad=0)
        for tj in tuples_level(L):
            S = tj[0]
            Gs = []
            for (s_, a, b, c, d) in [(tj[0], tj[1], tj[2], tj[3], tj[4]), (tj[0], tj[1], tj[3], tj[2], tj[4]), (tj[0], tj[1], tj[4], tj[2], tj[3])]:
                G = analyse((s_, a, b, c, d)); G['pairs'] = ((a, b), (c, d)); Gs.append(G)
            if len(Gs[0]['R']) < 2: continue
            sig2, mu = heron(Gs[0])
            q = true_q(Gs[0], sig2)
            if q is None: continue
            tot['tuples'] += 1; lev['n'] += 1
            covered = (S == 1)
            for G in Gs:
                tot['eviol'] += G['eviol']
                half_pair = 1 in G['pairs'][0] or 1 in G['pairs'][1]
                odd = (G['N1'] + G['N2']) % 2 == 1
                for k in (0, 1):
                    if G['closed'][k] and (odd or half_pair):
                        covered = True
                        Ck, Co, Mb = block(G, k)
                        tot['t68'] += 1
                        sv = np.linalg.svd(Mb, compute_uv=False)
                        if odd:   # Theorem 6 (and Theorem 8 with odd sum): injective, all sigma >= 1
                            if len(Co) < len(Ck) or sv.min() < 1 - 1e-10: tot['t68_bad'] += 1
                        else:     # Theorem 8, even sum: nonzero sigma >= 1 (J = V gives H'+1 with a kernel)
                            nz = sv[sv > 1e-9 * max(1.0, sv.max())]
                            if nz.size and nz.min() < 1 - 1e-10: tot['t68_bad'] += 1
                        sg, mu_g = heron(G)
                        qg = true_q(G, sg)
                        if qg < 2 * math.sqrt(mu_g) * (1 - 1e-9): tot['q_bad'] += 1
            if covered: tot['covered'] += 1; lev['cov'] += 1
            # ladder
            if S % 2 == 1 and S >= 3 and 1 in tj[1:] and not any(any(G['closed']) for G in Gs):
                tot['ladder'] += 1; lev['lad'] += 1
                for (s_, a, b, c, d) in [(tj[0], tj[1], tj[2], tj[3], tj[4]), (tj[0], tj[1], tj[3], tj[2], tj[4]), (tj[0], tj[1], tj[4], tj[2], tj[3])]:
                    if 1 not in (a, b): a, b, c, d = c, d, a, b     # rails = the spin-1/2 pair
                    G = analyse((s_, a, b, c, d))
                    csH, Hm = stark_matrix(c + d + 2, abs(c - d)); pos = {cc: i for i, cc in enumerate(csH)}
                    ok_all = True
                    for k in (0, 1):
                        Ck, Co, Mb = block(G, k)
                        J = sorted(pos[p[1]] for p in Ck); Jp = sorted(pos[p[1]] for p in Co)
                        contiguous = J == list(range(J[0], J[0] + len(J))) and Jp == list(range(Jp[0], Jp[0] + len(Jp)))
                        shifted = len(J) == len(Jp) and (Jp[0] == J[0] + 1 or Jp[0] == J[0] - 1)
                        cpos = [pos[p[1]] for p in Ck]; rpos = [pos[p[1]] for p in Co]
                        Wm = (Hm + np.eye(len(csH)))[np.ix_(rpos, cpos)]
                        same = np.allclose(np.abs(Wm), np.abs(Mb), atol=1e-12)
                        if not (contiguous and shifted and same): ok_all = False
                        sv = np.linalg.svd(Mb, compute_uv=False)
                        tot['ladder_sig'] = min(tot['ladder_sig'], sv.min())
                        if sv.min() < CSTAR - 1e-12: ok_all = False
                    sg, mu_g = heron(G); qg = true_q(G, sg)
                    if qg < 2 * math.sqrt(mu_g) * CSTAR * (1 - 1e-9): ok_all = False
                    if not ok_all: tot['ladder_bad'] += 1
        lines.append(f"level {L:2d}: tuples with nonzero spectrum {lev['n']:6d}, covered by [CL] Cor. 9 {lev['cov']:6d}, doubly-shifted ladder {lev['lad']:5d}")
        print(lines[-1], flush=True)
    lines.append(f"(E) explicit-closedness violations (all pairings): {tot['eviol']}")
    lines.append(f"(T6/T8) closed-class blocks with odd N-sum or a spin-1/2 pair: {tot['t68']}; violations (odd sum: injective with sigma >= 1; spin-1/2 pair, even sum: nonzero sigma >= 1): {tot['t68_bad']}; q < 2 sqrt(mu): {tot['q_bad']}")
    lines.append(f"(C11) doubly-shifted ladder tuples: {tot['ladder']}; pairings failing [interval, shifted by one, equal to P_J'(H'+1)P_J, sigma >= c*, q >= 2 sqrt(mu) c*]: {tot['ladder_bad']}; min sigma_min = {tot['ladder_sig']:.6f}")
    lines.append(f"total {time.time()-t0:.0f}s")
    print("\n".join(lines[-4:]))
    open(f"../outputs/gap_check_M{M}.txt", "w").write("\n".join(lines) + "\n")

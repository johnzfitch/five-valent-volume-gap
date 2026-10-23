"""High-precision check of Theorem Z(ii): for half-integer s and every pairing with a monotone zigzag pattern,
dim ker T = sum_k |imb_k|; for every pairing, dim ker T <= sum_k |imb_k|.
mpmath singular values at 40 digits (entries from exact rationals), all labelled tuples with 2j_i <= M.
Rank decided by a gap test: nonzero svs > 1e-25 * max, zero svs < 1e-30 * max (reports the separation).
Also checks lam(T) >= sigma_min(B') for both uniform end choices on monotone pairings (40 digits).
Usage: python oc_kernel_mp.py M out.txt
"""
import sys, os, itertools
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_core import links, sites, pairings, F, G, h2
from fractions import Fraction as Fr

mp.mp.dps = 40


def bip_mp(P):
    """Bipartite block with rows d+s even, cols d+s odd (mpmath), plus zigzag index of rows/cols."""
    S, L = links(P, 'T')
    s2 = P[0]
    rowsel = [i for i, (B, C) in enumerate(S) if (((B - C) + s2) // 2) % 2 == 0]
    colsel = [i for i, (B, C) in enumerate(S) if (((B - C) + s2) // 2) % 2 == 1]
    ri = {i: r for r, i in enumerate(rowsel)}; ci = {i: c for c, i in enumerate(colsel)}
    M = mp.zeros(len(rowsel), len(colsel))
    for (i, k), w in L.items():
        val = mp.sqrt(mp.mpf(w.numerator) / w.denominator)
        if i in ri:
            M[ri[i], ci[k]] = val
        else:
            M[ri[k], ci[i]] = val
    rk = [((S[i][0] - S[i][1]) + s2) // 4 for i in rowsel]
    ck = [((S[i][0] - S[i][1]) + s2 - 2) // 4 for i in colsel]
    useq = [(S[i][0] + S[i][1]) for i in rowsel], [(S[i][0] + S[i][1]) for i in colsel]
    return M, rk, ck, useq


def svals(M):
    if M.rows == 0 or M.cols == 0:
        return []
    return sorted([abs(x) for x in mp.svd_r(M, compute_uv=False)])


if __name__ == '__main__':
    Mx, out = int(sys.argv[1]), sys.argv[2]
    n = 0; viol_eq = 0; viol_le = 0; viol_red = 0; worst_sep = mp.mpf(1); nmono = 0
    for tj in itertools.product(range(1, Mx + 1), repeat=5):
        if tj[0] % 2 == 0 or sum(tj) % 2 or tj[1] > tj[2] or tj[3] > tj[4] or len(sites(tj)) < 2:
            continue
        n += 1
        kd = None
        for P in pairings(tj):
            M, rk, ck, (ur, uc) = bip_mp(P)
            sv = svals(M)
            smax = max(sv) if sv else mp.mpf(1)
            nz = [x for x in sv if x > mp.mpf('1e-25') * smax]
            z = [x for x in sv if x <= mp.mpf('1e-25') * smax]
            if z and max(z) > mp.mpf('1e-30') * smax:
                print('ambiguous rank', P); continue
            if nz and z:
                worst_sep = min(worst_sep, max(z) / min(nz))
            rank = len(nz)
            kT = M.rows + M.cols - 2 * rank
            if kd is None:
                kd = kT
            K = P[0] // 2 + 1
            imb = [rk.count(k) - ck.count(k) for k in range(K)]
            Ssum = sum(abs(x) for x in imb)
            if kT > Ssum:
                viol_le += 1
            w = False; mono = True
            for x in imb:
                if x == -1: w = True
                if x == 1 and w: mono = False
            if mono:
                nmono += 1
                if kT != Ssum:
                    viol_eq += 1
                lam = min(nz)
                for end in ('low', 'high'):
                    keepR, keepC = [], []
                    for k in range(K):
                        R = sorted([i for i in range(M.rows) if rk[i] == k], key=lambda i: ur[i])
                        C = sorted([j for j in range(M.cols) if ck[j] == k], key=lambda j: uc[j])
                        if len(R) - len(C) == 1:
                            R = R[1:] if end == 'low' else R[:-1]
                        if len(R) - len(C) == -1:
                            C = C[1:] if end == 'low' else C[:-1]
                        keepR += R; keepC += C
                    if keepR:
                        Bp = mp.matrix([[M[i, j] for j in keepC] for i in keepR])
                        sp = min(svals(Bp))
                        if sp > lam * (1 + mp.mpf('1e-30')):
                            viol_red += 1
    L = [f"Theorem Z checks at 40 digits, half-integer s, labelled tuples (j2<=j3, j4<=j5) with 2j_i<={Mx}: {n} tuples, {nmono} monotone pairings",
         f"  dim ker T > sum|imb| (any pairing): {viol_le}",
         f"  monotone pairing with dim ker T != sum|imb|: {viol_eq}",
         f"  lam(T) < sigma_min(B') (uniform end choices): {viol_red}",
         f"  largest (zero sv / smallest nonzero sv) over all cases: {mp.nstr(worst_sep, 5)}"]
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

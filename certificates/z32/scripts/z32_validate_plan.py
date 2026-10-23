"""Validate the regime plan on every j1=3/2 tuple with 2j <= M (exact link data, exact ratio tests):
  - chain types and imbalances as claimed (imbalances in {0,1});
  - pointwise ratio facts: CR chains r<=1 on every row, RC chains r>=1 on every row with q>0,
    RR chains: r strictly decreasing along interior rows (single crossing);
  - the 'count' bounds  F0, F1  (from the actual links) dominate 1/lam_min(D^T D);
  - the weighted bound W = sum_c Lambda+_c R_c (R_c row-norm bound) dominates sup x^T Lambda x/|D0 x|^2;
  - the resulting two-chain bound is <= the true q (consistency) and report its minimum.
Usage: python z32_validate_plan.py M
"""
import sys, os, collections, time
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, tuples32
from z32_regime_plan import regime, EXPECTED
from z32_regimes import ends

KAP = Fr(4, 3)


def chain_info(R, C, L):
    """Return dict with type, rows R, cols C, q_b=L[(b,b)], p_b=L[(b,b+1)] (squared, Fractions)."""
    q = {b: L.get((b, b), Fr(0)) for b in R}
    p = {b: L.get((b, b + 1), Fr(0)) for b in R}
    return ends(R, C), q, p


def count_bound(typ, R, C, q, p):
    """Upper bound for 1/lam_min(D^T D) by the count lemmas (exact links), or None if hypotheses fail."""
    if not C:
        return 0.0
    if typ == 'CR':      # matched q_b (C_b-R_b); needs p_b<=q_b on all rows
        if any(p[b] > q[b] for b in R):
            return None
        lo = R[0]
        return sum(float((b - lo + 1) / q[b]) for b in R)
    if typ == 'RC':      # matched p_b (R_b-C_{b+1}); needs q_b<=p_b
        if any(q[b] > p[b] for b in R):
            return None
        hi = R[-1]
        return sum(float((hi - b + 1) / p[b]) for b in R)
    if typ == 'RR':      # tall: single crossing; bound sum (hi-i)/p_i + sum (i-lo)/q_i
        lo, hi = R[0], R[-1]
        rs = [(b, p[b] / q[b]) for b in R if q[b] > 0 and p[b] > 0]
        if any(rs[i + 1][1] >= rs[i][1] for i in range(len(rs) - 1)):
            return None
        if q[lo] != 0 or p[hi] != 0:
            return None
        s = sum(float((hi - b) / p[b]) for b in R if b < hi)
        s += sum(float((b - lo) / q[b]) for b in R if b > lo)
        return s
    return None


def lam_vec(z):
    out = {}
    for b in z.C0:
        nn = sum(w for (r, c), w in z.N.items() if c == b)
        dd = sum(w for (r, c), w in z.D0.items() if c == b)
        out[b] = nn - KAP * dd
    return out


def rownorm_bound(typ, R, C, q, p):
    """Upper bound R_c >= sum_k (D0^+)_{c,k}^2 for each column c (left inverse used in the count lemma)."""
    out = {}
    if typ == 'CR':      # |E_{ck}| <= 1/q_k for k>=c
        for c in C:
            out[c] = sum(float(1 / q[k]) for k in R if k >= c)
    elif typ == 'RC':    # |E_{c,i}| <= 1/p_i for rows i<c
        for c in C:
            out[c] = sum(float(1 / p[i]) for i in R if i < c)
    elif typ == 'RR':    # use both pieces: column c gets min of lower-row and upper-row bounds
        for c in C:
            lowr = sum(float(1 / p[i]) for i in R if i < c)
            upr = sum(float(1 / q[i]) for i in R if i >= c and q[i] > 0)
            out[c] = max(lowr, upr)   # crossing unknown here: crude max
    return out


if __name__ == '__main__':
    M = int(sys.argv[1])
    t0 = time.time()
    stat = collections.Counter(); worst = {}; fails = []
    for tj in tuples32(M):
        name, P = regime(tj)
        z = Z32(P)
        if sum(len(x) for x in (z.R0, z.C0, z.R1, z.C1)) < 2:
            continue
        t0_, q0, p0 = chain_info(z.R0, z.C0, z.D0)
        t1_, q1, p1 = chain_info(z.R1, z.C1, z.D1)
        exp = EXPECTED[name]
        okt = (t0_ == exp[0] or not z.C0) and (t1_ == exp[1] or not z.C1)
        if not okt or min(z.imb) < 0:
            fails.append(('type', tj, name, t0_, t1_, z.imb)); continue
        F0 = count_bound(t0_, z.R0, z.C0, q0, p0)
        F1 = count_bound(t1_, z.R1, z.C1, q1, p1)
        if F0 is None or F1 is None:
            fails.append(('ratio', tj, name, F0, F1)); continue
        D0, N, D1 = z.mats()
        l0 = np.linalg.eigvalsh(D0.T @ D0)[0] if D0.size else np.inf
        l1 = np.linalg.eigvalsh(D1.T @ D1)[0] if D1.size else np.inf
        if F0 and 1 / l0 > F0 * (1 + 1e-9):
            fails.append(('F0', tj, name, 1 / l0, F0))
        if F1 and 1 / l1 > F1 * (1 + 1e-9):
            fails.append(('F1', tj, name, 1 / l1, F1))
        lam = lam_vec(z)
        Rb = rownorm_bound(t0_, z.R0, z.C0, q0, p0)
        W = sum(float(max(lam[c], 0)) * Rb[c] for c in z.C0)
        if D0.size:
            # true sup x^T Lam x / |D0 x|^2
            Lm = np.diag([float(lam[c]) for c in z.C0])
            G = D0.T @ D0
            Lc = np.linalg.cholesky(G)
            Li = np.linalg.inv(Lc)
            wtrue = np.linalg.eigvalsh(Li @ Lm @ Li.T)[-1]
            if wtrue > W * (1 + 1e-9) + 1e-12:
                fails.append(('W', tj, name, wtrue, W))
        s0 = 1 / F0 if F0 else np.inf
        s1 = 1 / F1 if F1 else np.inf
        best = 0
        for tau in np.linspace(0.005, 0.74, 148):
            v = min((1 - tau * (float(KAP) + W)) * s0, tau / (1 + tau) * s1)
            best = max(best, v)
        b8 = 8 * np.sqrt(max(best, 0))
        # true q
        B = np.zeros((D0.shape[0] + N.shape[0], D0.shape[1] + D1.shape[1]))
        B[:D0.shape[0], :D0.shape[1]] = D0; B[D0.shape[0]:, :D0.shape[1]] = N; B[D0.shape[0]:, D0.shape[1]:] = D1
        qtrue = 8 * np.linalg.svd(B, compute_uv=False).min()
        if b8 > qtrue * (1 + 1e-9):
            fails.append(('bound>q', tj, name, b8, qtrue))
        stat[name] += 1
        if name not in worst or b8 < worst[name][0]:
            worst[name] = (b8, tj, qtrue, W)
    print(f'M={M}: {sum(stat.values())} tuples in {time.time()-t0:.1f}s; failures {len(fails)}')
    for f in fails[:20]:
        print('  FAIL', f)
    for k in sorted(stat):
        w = worst[k]
        print(f'  {k:4s} {stat[k]:6d} tuples; min 8*sqrt(bound) = {w[0]:.4f} at {w[1]} (q={w[2]:.4f}, W={w[3]:.3f})')

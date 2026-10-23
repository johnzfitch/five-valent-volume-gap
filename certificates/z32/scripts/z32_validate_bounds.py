"""Validate the closed-form regime bounds (z32_bounds) against the exact matrices on every j1=3/2 tuple
with 2j <= M:  F0 >= 1/lam_min(D0^T D0), F1 >= 1/lam_min(D1^T D1), W >= sup x^T Lambda x/|D0 x|^2,
and the final two-chain bound <= sigma_min(B)^2 = (q/8)^2.  Also checks the pointwise ratio hypotheses
on the ACTUAL links (CR: s<=1; RC: r>=1; RR: strictly decreasing r).
Usage: python z32_validate_bounds.py M
"""
import sys, os, time, collections
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, tuples32
from z32_regime_plan import regime, EXPECTED
from z32_regimes import ends
from z32_bounds import regime_bounds, sigma2_bound, params_of
from z32_validate_plan import lam_vec


def hyp_ok(typ, R, L):
    q = {b: L.get((b, b), Fr(0)) for b in R}
    p = {b: L.get((b, b + 1), Fr(0)) for b in R}
    if typ == 'CR':
        return all(p[b] <= q[b + 1] for b in R if b + 1 in q)
    if typ == 'RC':
        return all(q[b] <= p[b] for b in R)
    if typ == 'RR':
        rs = [p[b] / q[b] for b in R if q[b] > 0 and p[b] > 0]
        return all(rs[i + 1] < rs[i] for i in range(len(rs) - 1)) and q[R[0]] == 0 and p[R[-1]] == 0
    return True


if __name__ == '__main__':
    M = int(sys.argv[1])
    t0 = time.time(); fails = []; stat = collections.Counter(); worst = {}
    cache = {}
    for tj in tuples32(M):
        name, P5 = regime(tj)
        z = Z32(P5)
        if sum(len(x) for x in (z.R0, z.C0, z.R1, z.C1)) < 2:
            continue
        t0_, t1_ = ends(z.R0, z.C0), ends(z.R1, z.C1)
        exp = EXPECTED[name]
        if (z.C0 and t0_ != exp[0]) or (z.C1 and t1_ != exp[1]) or min(z.imb) < 0:
            fails.append(('type', tj, name, t0_, t1_, z.imb)); continue
        if not (hyp_ok(t0_, z.R0, z.D0) and hyp_ok(t1_, z.R1, z.D1)):
            fails.append(('hyp', tj, name)); continue
        prm = params_of(name, P5)
        key = (name, tuple(sorted(prm.items())))
        if key not in cache:
            cache[key] = regime_bounds(name, prm)
        F0, F1, W = cache[key]
        D0, N, D1 = z.mats()
        l0 = np.linalg.eigvalsh(D0.T @ D0)[0] if D0.size else np.inf
        l1 = np.linalg.eigvalsh(D1.T @ D1)[0] if D1.size else np.inf
        if D0.size and 1 / l0 > float(F0) * (1 + 1e-9):
            fails.append(('F0', tj, name, 1 / l0, float(F0)))
        if D1.size and 1 / l1 > float(F1) * (1 + 1e-9):
            fails.append(('F1', tj, name, 1 / l1, float(F1)))
        if D0.size and D0.shape[0] == D0.shape[1]:
            lam = lam_vec(z)
            Lm = np.diag([float(lam[c]) for c in z.C0])
            E = np.linalg.inv(D0)
            wtrue = np.linalg.eigvalsh(E.T @ Lm @ E)[-1]
            if wtrue > float(W) * (1 + 1e-9) + 1e-12:
                fails.append(('W', tj, name, wtrue, float(W)))
        s2 = sigma2_bound(F0, F1, W)
        B = np.zeros((D0.shape[0] + N.shape[0], D0.shape[1] + D1.shape[1]))
        B[:D0.shape[0], :D0.shape[1]] = D0; B[D0.shape[0]:, :D0.shape[1]] = N; B[D0.shape[0]:, D0.shape[1]:] = D1
        sv = np.linalg.svd(B, compute_uv=False).min()
        if s2 > sv * sv * (1 + 1e-9):
            fails.append(('final', tj, name, s2, sv * sv))
        b8 = 8 * np.sqrt(s2)
        stat[name] += 1
        if name not in worst or b8 < worst[name][0]:
            worst[name] = (b8, tj, 8 * sv, prm)
    print(f'M={M}: {sum(stat.values())} tuples, {len(cache)} parameter sets, {time.time()-t0:.1f}s; failures {len(fails)}')
    for f in fails[:25]:
        print('  FAIL', f)
    for k in sorted(stat):
        w = worst[k]
        print(f'  {k:4s} {stat[k]:6d} tuples; min analytic 8*sigma >= {w[0]:.4f} at {w[1]} (true q={w[2]:.4f}) prm={ {a: str(b) for a, b in w[3].items()} }')

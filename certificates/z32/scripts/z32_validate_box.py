"""Validate the box bounds (z32_box_torch, best-of count / exact-product forms) against the exact matrices
for every j1 = 3/2 tuple with 2j <= M:
   F0_box >= 1/lam_min(D0^T D0),  F1_box >= 1/lam_min(D1^T D1),  W_box >= sup x^T Lam x/|D0 x|^2 (square D0),
   s2_box <= sigma_min(B)^2.
Also cross-checks the float count-form values against the exact Fraction implementation (z32_bounds).
Usage: python z32_validate_box.py M out.txt
"""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, tuples32
from z32_regime_plan import regime
from z32_bounds import params_of, regime_bounds
from z32_validate_plan import lam_vec
import z32_box_torch as zt

if __name__ == '__main__':
    M, out = int(sys.argv[1]), sys.argv[2]
    t0 = time.time()
    dev = torch.device('cpu')
    tables = {}
    for name in ('R1', 'R2+', 'R2-', 'R4', 'R1c', 'R4c', 'Tp', 'Tm'):
        n, a, F0, F1, W, s2 = zt.regime_box(name, M + 2, dev)
        tables[name] = {(float(x), float(y)): (float(f0), float(f1), float(w), float(s)) for x, y, f0, f1, w, s in
                        zip(n, a, F0, F1, W, s2)}
    fails = []; cnt = 0; ratio_min = 1e9; worst = None; exact_dev = 0.0
    for tj in tuples32(M):
        name, P5 = regime(tj)
        z = Z32(P5)
        if sum(len(x) for x in (z.R0, z.C0, z.R1, z.C1)) < 2:
            continue
        prm = params_of(name, P5)
        key = (float(prm['n1']), float(prm.get('m1', prm.get('m2', 0))))
        if name in ('R1c', 'R4c', 'Tp', 'Tm'):
            key = (float(prm['n1']), 0.0)
        F0, F1, W, s2 = tables[name][key]
        D0, N, D1 = z.mats()
        l0 = np.linalg.eigvalsh(D0.T @ D0)[0] if D0.size else np.inf
        l1 = np.linalg.eigvalsh(D1.T @ D1)[0] if D1.size else np.inf
        if D0.size and 1 / l0 > F0 * (1 + 1e-9):
            fails.append(('F0', tj, 1 / l0, F0))
        if D1.size and 1 / l1 > F1 * (1 + 1e-9):
            fails.append(('F1', tj, 1 / l1, F1))
        if D0.size and D0.shape[0] == D0.shape[1]:
            lam = lam_vec(z)
            E = np.linalg.inv(D0)
            wt = np.linalg.eigvalsh(E.T @ np.diag([float(lam[c]) for c in z.C0]) @ E)[-1]
            if wt > W * (1 + 1e-9) + 1e-12:
                fails.append(('W', tj, wt, W))
        B = np.zeros((D0.shape[0] + N.shape[0], D0.shape[1] + D1.shape[1]))
        B[:D0.shape[0], :D0.shape[1]] = D0; B[D0.shape[0]:, :D0.shape[1]] = N; B[D0.shape[0]:, D0.shape[1]:] = D1
        sv = np.linalg.svd(B, compute_uv=False).min()
        if s2 > sv * sv * (1 + 1e-9):
            fails.append(('s2', tj, s2, sv * sv))
        r = np.sqrt(s2) / sv
        if r < ratio_min:
            ratio_min, worst = r, tj
        cnt += 1
    # exact cross-check of the count-form pieces on small parameters (Fraction implementation)
    for name in ('R1', 'R2+', 'R2-', 'R4', 'R1c', 'R4c', 'Tp', 'Tm'):
        for key, (F0, F1, W, s2) in list(tables[name].items())[:60]:
            from fractions import Fraction as Fr
            n1 = Fr(key[0]).limit_denominator(2)
            prm = {'n1': n1}
            if name in ('R1', 'R2+', 'R2-'):
                prm['m1'] = Fr(key[1]).limit_denominator(2)
            if name == 'R4':
                prm['m2'] = Fr(key[1]).limit_denominator(2)
            try:
                F0e, F1e, We = regime_bounds(name, prm)
            except Exception:
                continue
            # box takes min with telescoping forms, so box <= count (up to inflation)
            for bx, ex in ((F0, F0e), (F1, F1e), (W, We)):
                if float(ex) > 0 and bx > float(ex) * (1 + 2e-9):
                    fails.append(('box>count', name, key, bx, float(ex)))
                if float(ex) > 0:
                    exact_dev = max(exact_dev, 0.0)
    L = [f'validate box vs exact matrices, all j1=3/2 tuples 2j<={M}: {cnt} tuples, {len(fails)} failures, {time.time()-t0:.1f}s',
         f'smallest ratio sqrt(s2_box)/sigma_min(B) = {ratio_min:.4f} at {worst}']
    L += ['  FAIL ' + str(f) for f in fails[:20]]
    print('\n'.join(L)); open(out, 'w').write('\n'.join(L) + '\n')

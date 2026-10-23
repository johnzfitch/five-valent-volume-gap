"""Exact (rational) certificate on actual j1 = 3/2 tuples with 2j <= M.

For each tuple, in the all-tall ordered pairing of the regime plan, B = [[D0,0],[N,D1]] is injective and
   q^2/64 = sigma_min(B)^2 >= min( lam_min(D0^T D0 - tau N^T N),  tau/(1+tau) lam_min(D1^T D1) ).
Both matrices are tridiagonal in the column index with RATIONAL diagonal and RATIONAL squared off-diagonal
(off-diagonals are products of square roots whose squares are rational).  We certify
   D0^T D0 - tau N^T N - s0 I  >= 0   and   D1^T D1 - s1 I >= 0
by exact LDL^T pivots (all pivots > 0 for a positive definite tridiagonal matrix; a zero pivot only at the
last step is also allowed) with rational tau, s0, s1 chosen from a float estimate, and report the certified
value  qcert = 8 sqrt(min(s0, tau/(1+tau) s1)).
Usage: python z32_exact_cert.py M out.txt
"""
import sys, os, time, math
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, tuples32
from z32_regime_plan import regime


def gram_tridiag(z, which, tau):
    """exact (diag, offdiag^2) of the column Gram matrix:
       which='0': D0^T D0 - tau N^T N  on zz0 columns;  which='1': D1^T D1 on zz1 columns."""
    if which == '0':
        cols, LD, LN = z.C0, z.D0, z.N
    else:
        cols, LD, LN = z.C1, z.D1, {}
    idx = {c: k for k, c in enumerate(cols)}
    K = len(cols)
    dg = [Fr(0)] * K
    od = [Fr(0)] * max(K - 1, 0)        # off-diagonal value as (sign, square) -> store signed sqrt-free form
    # contributions: each row contributes to diag (link^2) and to off-diag (product of its two links)
    rows_D = {}
    for (r, c), w in LD.items():
        if c in idx:
            dg[idx[c]] += w
            rows_D.setdefault(r, []).append((c, w))
    rows_N = {}
    for (r, c), w in LN.items():
        if c in idx:
            dg[idx[c]] -= tau * w
            rows_N.setdefault(r, []).append((c, w))
    # off-diagonal (c, c+1): D part = +sqrt(w1 w2) at the shared zz-row; N part = -tau sqrt(v1 v2).
    # By Proposition 6 (row-product identity) sqrt(v1 v2) = (4/3) sqrt(w1 w2) whenever both rows exist.
    offD = [None] * max(K - 1, 0); offN = [None] * max(K - 1, 0)
    for r, lst in rows_D.items():
        if len(lst) == 2:
            (c1, w1), (c2, w2) = sorted(lst)
            offD[idx[c1]] = w1 * w2
    for r, lst in rows_N.items():
        if len(lst) == 2:
            (c1, w1), (c2, w2) = sorted(lst)
            offN[idx[c1]] = w1 * w2
    off2 = []
    for k in range(K - 1):
        a = offD[k]; b = offN[k]
        if b is None or tau == 0:
            off2.append(a if a is not None else Fr(0))
        elif a is None:
            off2.append(tau * tau * b)
        else:
            # verify the identity b = (16/9) a exactly, then (sqrt a - tau sqrt b)^2 = a (1 - 4 tau/3)^2
            assert b == Fr(16, 9) * a, ('row-product identity failed', z.tj)
            off2.append(a * (1 - Fr(4, 3) * tau) ** 2)
    return dg, off2


def psd_shift(dg, off2, s):
    """exact check that the tridiagonal matrix minus s*I is positive semidefinite (LDL^T pivots)."""
    d = None
    K = len(dg)
    for k in range(K):
        d = (dg[k] - s) if k == 0 else (dg[k] - s - off2[k - 1] / d)
        if d < 0:
            return False
        if d == 0 and k < K - 1:
            return False
    return True


def float_lmin(dg, off2):
    K = len(dg)
    if K == 0:
        return math.inf
    M = np.diag([float(x) for x in dg])
    for k in range(K - 1):
        v = math.sqrt(float(off2[k]))
        M[k, k + 1] = M[k + 1, k] = v
    return np.linalg.eigvalsh(M)[0]


def certify_tuple(tj):
    name, P5 = regime(tj)
    z = Z32(P5)
    if sum(len(x) for x in (z.R0, z.C0, z.R1, z.C1)) < 2:
        return None
    dg1, o1 = gram_tridiag(z, '1', Fr(0))
    l1 = float_lmin(dg1, o1)
    best = (0.0, None)
    for t in [i / 40 for i in range(0, 30)]:
        dg0, o0 = gram_tridiag(z, '0', Fr(t).limit_denominator(40))
        l0 = float_lmin(dg0, o0)
        v = min(l0, t / (1 + t) * l1 if l1 < math.inf else l0) if t > 0 else (l0 if l1 == math.inf else 0)
        if v > best[0]:
            best = (v, t)
    v, t = best
    tau = Fr(t).limit_denominator(40)
    dg0, o0 = gram_tridiag(z, '0', tau)
    l0 = float_lmin(dg0, o0)
    # certified shifts slightly below the float values
    s0 = Fr(l0 * (1 - 1e-6)).limit_denominator(10 ** 9) if l0 < math.inf else None
    s1 = Fr(l1 * (1 - 1e-6)).limit_denominator(10 ** 9) if l1 < math.inf else None
    ok0 = psd_shift(dg0, o0, s0) if s0 is not None else True
    ok1 = psd_shift(dg1, o1, s1) if s1 is not None else True
    if not (ok0 and ok1):
        return (name, tj, None)
    vals = []
    if s0 is not None:
        vals.append(s0)
    if s1 is not None:
        vals.append(tau / (1 + tau) * s1 if tau > 0 else Fr(0))
    sig2 = min(vals)
    return (name, tj, 8 * math.sqrt(float(sig2)), float(tau))


if __name__ == '__main__':
    M, out = int(sys.argv[1]), sys.argv[2]
    t0 = time.time()
    res = []; fails = []
    for tj in tuples32(M):
        r = certify_tuple(tj)
        if r is None:
            continue
        if r[2] is None:
            fails.append(r); continue
        res.append(r)
    res.sort(key=lambda x: x[2])
    L = [f'exact LDL^T certificate, all j1=3/2 tuples 2j<={M}: {len(res)} certified, {len(fails)} failed, {time.time()-t0:.1f}s',
         'smallest certified q lower bounds (regime, tuple, q_cert, tau):']
    L += [f'  {r[0]:4s} {r[1]}  {r[2]:.4f}  tau={r[3]}' for r in res[:15]]
    L += ['  FAIL ' + str(f) for f in fails[:10]]
    print('\n'.join(L)); open(out, 'w').write('\n'.join(L) + '\n')

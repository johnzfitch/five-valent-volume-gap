"""Certified linear law for tuples in which three of legs 2-5 are bounded (2j <= K).
For each multiset t of three doubled spins <= K, each M = s - j5 (j5 = the fourth leg), and each choice of
which bounded leg pairs with j5:
  * Y_inf = limit (W/s) bipartite block from the smaller colour class; sigma_inf = certified lower bound on
    sigma_min(Y_inf) (interval Cholesky of Y^T Y - tau I in mpmath.iv);
  * E(s0)^2 = sum over links of c2 * factor(s0)^2 / 16  (exact rational) bounds ||Y(s) - s Y_inf||_F, s >= s0;
  * hence for s >= s0:  q = 8 sigma_min(Y(s)) >= 8 (s sigma_inf - E(s0))   (Weyl for singular values).
s0 is the least admissible value (structure stable) with s0*sigma_inf - E(s0) >= 4.5579607/8.
Tuples with s < s0 are listed and checked against the finite certificate range 2j_max <= 38.
Usage: python lj_limit_cert.py K"""
import sys, os, itertools, json, numpy as np
from fractions import Fraction as Fr
from math import ceil
import mpmath
from mpmath import iv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lj_limit_core as LC
iv.dps = 40
QSTAR = Fr(45579607, 10 ** 7)


def chol_pos(G):
    n = len(G); A = [row[:] for row in G]
    for k in range(n):
        if not (A[k][k].a > 0):
            return False
        r = iv.sqrt(A[k][k]); A[k][k] = r
        for i in range(k + 1, n):
            A[i][k] = A[i][k] / r
        for i in range(k + 1, n):
            for j in range(k + 1, i + 1):
                A[i][j] = A[i][j] - A[i][k] * A[j][k]
                A[j][i] = A[i][j]
    return True


def analyse(t2, M2):
    S, Ls = LC.links(t2, M2)
    if not S or not Ls:
        return None
    col = [int((b + g) % 2) for (b, g) in S]
    E_ = [i for i in range(len(S)) if col[i] == 0]; O_ = [i for i in range(len(S)) if col[i] == 1]
    small, large = (O_, E_) if len(O_) <= len(E_) else (E_, O_)
    ps = {v: k for k, v in enumerate(small)}; pl = {v: k for k, v in enumerate(large)}
    Yf = np.zeros((len(large), len(small)))
    Yiv = [[iv.mpf(0) for _ in small] for _ in large]
    for lk in Ls:
        a, b = lk[0], lk[1]
        if a in ps: a, b = b, a
        val = iv.sqrt(iv.mpf(lk[3]['c2'].numerator) / lk[3]['c2'].denominator) / 2
        Yiv[pl[a]][ps[b]] = val; Yf[pl[a], ps[b]] = float(lk[3]['c2']) ** .5 / 2
    if len(small) == 0:
        return dict(kind='empty')
    sv = np.linalg.svd(Yf, compute_uv=False)
    smin = sv.min() if len(sv) == len(small) else 0.0
    if smin < 1e-8 * max(1, sv.max()):
        return dict(kind='noninjective', nsmall=len(small), nlarge=len(large))
    sigF = Fr(smin * (1 - 1e-6))
    tau = (iv.mpf(sigF.numerator) / sigF.denominator) ** 2
    G = [[sum((Yiv[r][i] * Yiv[r][j] for r in range(len(large))), iv.mpf(0)) - (tau if i == j else 0)
          for j in range(len(small))] for i in range(len(small))]
    ok = chol_pos(G)
    if not ok:
        return dict(kind='certfail')
    sig = sigF                                  # exact rational, certified lower bound
    return dict(kind='ok', sig=sig, sig_float=smin, Ls=Ls, S=S, nsmall=len(small), nlarge=len(large))


def s_min_struct(t2, M2):
    j2, j3, j4 = (Fr(x, 2) for x in t2); M = Fr(M2, 2)
    bmax = j2 + j3; gmin = -M - j4; gabs = max(abs(-M - j4), abs(j4 - M))
    return max(M + j4, (bmax - gmin) / 2, gabs + 1, Fr(1))


def E2(Ls, s0):
    return sum(lk[3]['c2'] * LC.err_bound(lk, s0) ** 2 / 16 for lk in Ls)


def best_s0(res, t2, M2):
    """least s0 (same parity class as admissible s) with s0*sig - E(s0) >= q*/8; returns (s0, E(s0))."""
    sig = res['sig']; base = s_min_struct(t2, M2)
    # admissible s: 2s = 2M + 2j5 with j5 integer/half-int: 2s parity = parity of (M2 + 2j5) -> any; scan by halves
    s0 = Fr(ceil(base * 2), 2)
    while True:
        e2 = E2(res['Ls'], s0)
        lhs = s0 * sig - QSTAR / 8
        if lhs > 0 and lhs * lhs >= e2:
            return s0, e2
        s0 += Fr(1, 2)
        if s0 > 10 ** 6:
            return None, None


if __name__ == '__main__':
    K = int(sys.argv[1])
    out = []; worst_slope = None; worst_s0 = Fr(0); fails = []
    for t in itertools.combinations_with_replacement(range(1, K + 1), 3):
        for M2 in range(-sum(t), sum(t) + 1, 2):
            best = None; trivial = True
            for k in range(3):                     # which bounded leg pairs with j5
                t2 = tuple(t[i] for i in range(3) if i != k) + (t[k],)
                r = analyse(t2, M2)
                if r is not None and r['kind'] != 'empty': trivial = False
                if r is None or r['kind'] != 'ok':
                    continue
                s0, e2 = best_s0(r, t2, M2)
                if s0 is None: continue
                cand = (s0, -r['sig'], t2, r, e2)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            if best is None and trivial:
                print(f"t={t} M2={M2}: limit operator has no links (Inv dim <= 1 or empty): no nonzero spectrum", flush=True); continue
            if best is None:
                fails.append((t, M2)); print(f"t={t} M2={M2}: NO certified assignment", flush=True); continue
            s0, _, t2, r, e2 = best
            E = float(e2) ** .5
            out.append(dict(t=t, M2=M2, assign=t2, sigma_inf=float(r['sig']), s0=float(s0), E_s0=E))
            worst_s0 = max(worst_s0, s0)
            if worst_slope is None or r['sig'] < worst_slope[0]: worst_slope = (r['sig'], t, M2)
            print(f"t={t} M2={M2:3d} assign={t2} sigma_inf>={float(r['sig']):.8f}  s0={float(s0):6.1f}  E(s0)={E:8.4f}", flush=True)
    print(f"cases certified {len(out)}, failures {len(fails)}: {fails}")
    if worst_slope:
        print(f"smallest certified slope sigma_inf = {float(worst_slope[0]):.8f} at t={worst_slope[1]} M2={worst_slope[2]};  8*sigma = {8*float(worst_slope[0]):.6f}")
    print(f"largest s0 = {float(worst_s0)}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'outputs', f'limit_cert_K{K}.json'), 'w'), indent=0)

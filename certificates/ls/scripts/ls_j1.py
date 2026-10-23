"""j1 = 1: exact one-dimensional reduction (Proposition R), exact certificate for max(n1,n2) <= 32,
and checks of the analytic Hardy bounds used for max(n1,n2) >= 33 (Theorem P of THEOREM.md).

Pencil: sites l = lo..hi (lo = max(m1,m2), raised to 1 if both m are 0; hi = min(n1,n2)-1), edges e = lo..hi+1,
P_i(e) = max(0,(n_i^2-e^2)(e^2-m_i^2)), A_e = (sqrtP1+sqrtP2)^2/(8e), M_e = (sqrtP1-sqrtP2)^2/(8e),
mu_l = (2l+1)/(l(l+1)).  Claim: {q^2/64} = nonzero eigenvalues of (E, diag mu).
Certificate: exact rational LDL^T pivots of E - c*mu, c = (4.558/8)^2: 0 negative pivots (positive definite
case) or exactly 1 negative pivot with identical pairs (kernel = constants).
Usage: python ls_j1.py > ../outputs/j1_out.txt
"""
import sys, os, math, itertools, time, warnings
from fractions import Fraction as Fr
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ls_core as L

QC = Fr(4558, 1000)
CC = (QC / 8) ** 2


def pencil_float(n1, m1, n2, m2):
    lo = max(m1, m2); lo = 1.0 if lo == 0 else lo
    hi = min(n1, n2) - 1
    ells = np.arange(lo, hi + 1)
    K = len(ells)
    if K == 0:
        return None
    E = np.arange(lo, hi + 2)
    P1 = np.maximum((n1 * n1 - E * E) * (E * E - m1 * m1), 0); P2 = np.maximum((n2 * n2 - E * E) * (E * E - m2 * m2), 0)
    A = (np.sqrt(P1) + np.sqrt(P2)) ** 2 / (8 * E); M = (np.sqrt(P1) - np.sqrt(P2)) ** 2 / (8 * E)
    Lm = np.zeros((K, K))
    for e in range(K + 1):
        i, j = e - 1, e
        for a, sa in ((i, -1), (j, 1)):
            for b, sb in ((i, -1), (j, 1)):
                if 0 <= a < K and 0 <= b < K:
                    Lm[a, b] += A[e] * sa * sb + M[e]
    mu = (2 * ells + 1) / (ells * (ells + 1))
    return Lm, mu


def q_from_pencil(n1, m1, n2, m2):
    r = pencil_float(n1, m1, n2, m2)
    if r is None:
        return None
    Lm, mu = r
    s = 1 / np.sqrt(mu)
    w = np.linalg.eigvalsh(Lm * s[:, None] * s[None, :])
    wp = w[w > 1e-9 * max(1.0, abs(w).max())]
    return 8 * math.sqrt(wp.min()) if wp.size else None


def certify(n1, m1, n2, m2):
    """exact: returns 'pd', 'kernel1' or a failure string."""
    n1, m1, n2, m2 = map(Fr, (n1, m1, n2, m2))
    lo = max(m1, m2); lo = Fr(1) if lo == 0 else lo
    hi = min(n1, n2) - 1
    if hi < lo:
        return 'empty'
    K = int(hi - lo) + 1
    P = lambda n, m, e: max(Fr(0), (n * n - e * e) * (e * e - m * m))
    d = None; neg = 0
    for k in range(K):
        l = lo + k
        diag = Fr(0)
        for e in (l, l + 1):
            diag += (P(n1, m1, e) + P(n2, m2, e)) / (4 * e)
        diag -= CC * (2 * l + 1) / (l * (l + 1))
        if k == 0:
            d = diag
        else:
            e = l
            b2 = P(n1, m1, e) * P(n2, m2, e) / (4 * e * e)
            d = diag - b2 / d
        if d == 0:
            return 'zero pivot'
        if d < 0:
            neg += 1
    ident = (n1 == n2 and m1 == m2 and m1 > 0)
    if neg == 0:
        return 'pd'
    if neg == 1 and ident:
        return 'kernel1'
    return f'FAIL neg={neg} ident={ident}'


def pairs_upto(nmax2):
    """pairs (2ja <= 2jb) with n = ja+jb+1 <= nmax2/2; returns list of (n, m) as Fractions plus spins."""
    out = []
    for a in range(1, nmax2):
        for b in range(a, nmax2):
            if Fr(a + b, 2) + 1 <= Fr(nmax2, 2):
                out.append((Fr(a + b, 2) + 1, Fr(b - a, 2), a, b))
    return out


def F_a(n, m, nu):
    """analytic bound (case nu > n) used in Theorem P: tau'*M' <= F."""
    N = n - m
    H = lambda k: sum(1.0 / i for i in range(1, int(k) + 1))
    Mh = 4 + 2 * math.log(2 * n)
    if m > n / 2:
        Mh = min(Mh, 4 * N / n)
    t1 = (4 * H(N - 1) + 4) / ((n - m) * (n + m))
    t2 = 8 * H(N) / (nu * nu - n * n)
    return min(t1, t2) * Mh


if __name__ == '__main__':
    t0 = time.time()
    # (1) reduction check
    worst = 0.0; cnt = 0
    for tj in itertools.product([2], *[range(1, 17)] * 4):
        if sum(tj) % 2 or tj[1] > tj[2] or tj[3] > tj[4]:
            continue
        if L.dim_inv(tj) < 2:
            continue
        q, _ = L.qmin(tj)
        s2, a, b, c, d = tj
        qp = q_from_pencil((a + b) / 2 + 1, abs(a - b) / 2, (c + d) / 2 + 1, abs(c - d) / 2)
        if q is None and qp is None:
            continue
        cnt += 1
        worst = max(worst, abs(q[0] - qp) / q[0])
    print(f"[R] pencil vs full operator: {cnt} tuples with j1=1, 2j_i<=16: max rel diff {worst:.2e}  ({time.time()-t0:.0f}s)", flush=True)

    # (2) exact certificate for max(n1,n2) <= 32
    t1 = time.time()
    prs = pairs_upto(64)
    stats = {}; fails = []; nt = 0; fmin = (1e9, None)
    for i in range(len(prs)):
        for k in range(i, len(prs)):
            n1, m1, a, b = prs[i]; n2, m2, c, d = prs[k]
            if (n1 - n2).denominator != 1:
                continue
            r = certify(n1, m1, n2, m2)
            if r == 'empty':
                continue
            nt += 1
            stats[r] = stats.get(r, 0) + 1
            if r not in ('pd', 'kernel1'):
                fails.append(((a, b, c, d), r))
            if nt % 7 == 0 or max(n1, n2) <= 12:
                qf = q_from_pencil(float(n1), float(m1), float(n2), float(m2))
                if qf is not None and qf < fmin[0]:
                    fmin = (qf, (2, a, b, c, d))
    print(f"[C] exact certificate q >= 4.558 for all j1=1 tuples with max(n1,n2) <= 32: {nt} pair-pairs, "
          f"outcomes {stats}, failures {len(fails)} {fails[:5]}  ({time.time()-t1:.0f}s)")
    print(f"    float min q over sampled pair-pairs: {fmin[0]:.6f} at doubled tuple {fmin[1]}", flush=True)

    # (3) analytic bounds: check thresholds and compare with the actual Hardy quantity on a grid
    H = lambda k: sum(1.0 / i for i in range(1, int(k) + 1))
    g = lambda n, c: 8 * (H(n) + c) * (4 + 2 * math.log(2 * n)) / n ** 2
    print(f"[A] case nu>n, n>=25, m<=n/2: F <= 8(H_n+1)(4+2ln2n)/n^2 at n=24,25: {g(24,1):.4f}, {g(25,1):.4f} (need <= {(4/4.558)**2:.4f})")
    print(f"[A] case nu>n, n<=24, nu>=33: F <= 8 H_24 (4+2ln48)/(33^2-24^2) = {8*H(24)*(4+2*math.log(48))/(33**2-24**2):.4f}")
    print(f"[A] case nu=n (not identical), n>=28: 8(H_n+2)(4+2ln2n)/n^2 at n=27,28: {g(27,2):.4f}, {g(28,2):.4f}")
    print(f"[A] identical pairs, n>=28: tau*M <= 8H_n(4+2ln2n)/n^2 = {g(28,0):.4f} (need <= {2*(4/4.558)**2:.4f})")
    # sanity: actual Hardy product tau'*M' <= analytic F on a grid, case nu > n
    bad = 0; chk = 0
    for n2x in range(6, 120):
        n = n2x / 2
        for m2x in range(0, n2x - 2, 1):
            m = m2x / 2
            if (n - m) != int(n - m):
                continue
            for dnu in (1, 2, 5, 20):
                nu = n + dnu
                for (m1, m2) in ((m, max(m - 1, 0) if m >= 1 else m), (m, m)):
                    lo = max(m1, m2); lo = 1.0 if lo == 0 else lo
                    if lo > n - 1:
                        continue
                    E = np.arange(lo + 1, n + 1)
                    P1 = np.maximum((n * n - E * E) * (E * E - m1 * m1), 0); P2 = np.maximum((nu * nu - E * E) * (E * E - m2 * m2), 0)
                    A = (np.sqrt(P1) + np.sqrt(P2)) ** 2 / (8 * E)
                    tau = np.sum(1 / A)
                    ells = np.arange(lo, n)
                    Mt = np.sum((2 * ells + 1) / (ells * (ells + 1)))
                    chk += 1
                    if tau * Mt > F_a(n, m, nu) * (1 + 1e-12):
                        bad += 1
    print(f"[S] grid sanity tau'*M' <= F_a: {chk} cases, {bad} violations")
    print(f"total {time.time()-t0:.0f}s")

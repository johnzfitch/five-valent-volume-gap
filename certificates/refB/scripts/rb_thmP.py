"""Checks of [LS] Theorem P (j1 = 1 closed), written from the statement in THEOREM.md, independent code.

Pencil of Prop. R for pairs (n1,m1 | n2,m2):  sites l = lo..hi, lo = max(m1,m2) (1 if both 0), hi = min(n1,n2)-1,
edges e = lo..hi+1, P_i(e) = max(0,(n_i^2-e^2)(e^2-m_i^2)),
A_e = (sqrtP1+sqrtP2)^2/(8e), M_e = (sqrtP1-sqrtP2)^2/(8e), mu_l = (2l+1)/(l(l+1)), ghosts Phi_{lo-1} = Phi_{hi+1} = 0.
  [R] pencil spectrum vs the Racah operator rb_core on all j1 = 1 tuples with legs 2j <= MAXR (pairing 23|45).
  [C] exact rational LDL^T certificate of E - c mu, c = (4.558/8)^2, for ALL unordered pair-pairs with n1, n2 <= NMAXC
      (half-integers included), reported separately for max(n1,n2) <= 32 (the authors' range), = 32.5 and = 33.
  [A] the analytic bounds of the proof, evaluated with exact harmonic numbers on the grid that the case split leaves,
      including nu = 32.5 (not covered by either part of the written proof).
  [S] float: true pencil q for random pair-pairs with max(n1,n2) >= 33 up to n = 3000.
Usage: python rb_thmP.py MAXR NMAXC > ../outputs/rb_thmP_out.txt
"""
import sys, os, math, time, random, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C

QC = Fr(4558, 1000)
CC = (QC / 8) ** 2


def P(n, m, e):
    v = (n * n - e * e) * (e * e - m * m)
    return v if v > 0 else 0 * v


def pencil_float(n1, m1, n2, m2):
    lo = max(m1, m2); lo = 1.0 if lo == 0 else lo
    hi = min(n1, n2) - 1
    if hi < lo:
        return None
    l = np.arange(lo, hi + 0.5)
    K = len(l)
    e = np.arange(lo, hi + 1.5)                         # K+1 edges
    p1 = np.maximum((n1 * n1 - e * e) * (e * e - m1 * m1), 0); p2 = np.maximum((n2 * n2 - e * e) * (e * e - m2 * m2), 0)
    S = (p1 + p2) / (4 * e)                             # A+M
    off = -np.sqrt(p1 * p2) / (2 * e)                   # M-A on the interior edges
    diag = S[:-1] + S[1:]
    mu = (2 * l + 1) / (l * (l + 1))
    s = 1 / np.sqrt(mu)
    w = eigvalsh_tridiagonal(diag * s * s, off[1:-1] * s[:-1] * s[1:]) if K > 1 else np.array([diag[0] / mu[0]])
    return w


def q_pencil(n1, m1, n2, m2):
    w = pencil_float(n1, m1, n2, m2)
    if w is None:
        return None
    wp = w[w > 1e-9 * max(1.0, abs(w).max())]
    return 8 * math.sqrt(wp.min()) if wp.size else None


def certify(n1, m1, n2, m2):
    lo = max(m1, m2); lo = Fr(1) if lo == 0 else lo
    hi = min(n1, n2) - 1
    if hi < lo:
        return 'empty'
    K = int(hi - lo) + 1
    d = None; neg = 0
    for k in range(K):
        l = lo + k
        diag = sum(((P(n1, m1, e) + P(n2, m2, e)) / (4 * e) for e in (l, l + 1)), Fr(0)) - CC * (2 * l + 1) / (l * (l + 1))
        if k == 0:
            d = diag
        else:
            off2 = P(n1, m1, l) * P(n2, m2, l) / (4 * l * l)
            d = diag - off2 / d
        if d == 0:
            return 'zero'
        neg += d < 0
    ident = (n1 == n2 and m1 == m2 and m1 > 0)
    if neg == 0:
        return 'pd'
    if neg == 1 and ident:
        return 'kernel1'
    return f'FAIL{neg}'


def Hn(k):
    return sum(Fr(1, i) for i in range(1, int(k) + 1))


if __name__ == '__main__':
    MAXR = int(sys.argv[1]); NMAXC = Fr(sys.argv[2])
    t0 = time.time()
    # [R]
    worst = 0.0; cnt = 0
    for a, b, c, d in itertools.product(range(1, MAXR + 1), repeat=4):
        if a > b or c > d or (a + b + c + d) % 2:
            continue
        tj = (2, a, b, c, d)
        if C.dim_inv(tj) < 2:
            continue
        q, _ = C.qmin(tj)
        qp = q_pencil((a + b) / 2 + 1, abs(a - b) / 2, (c + d) / 2 + 1, abs(c - d) / 2)
        if q is None and qp is None:
            continue
        if (q is None) != (qp is None):
            worst = float('inf'); print('KERNEL MISMATCH', tj); continue
        cnt += 1; worst = max(worst, abs(q - qp) / q)
    print(f"[R] Prop. R pencil vs Racah operator: {cnt} j1=1 tuples (legs 2j<={MAXR}), max rel diff {worst:.2e}  ({time.time()-t0:.0f}s)", flush=True)

    # [C]
    t1 = time.time()
    pairs = []                                           # (n, m) with n = (a+b)/2+1, m = |a-b|/2, a,b >= 1
    for a in range(1, int(2 * NMAXC)):
        for b in range(a, int(2 * NMAXC)):
            n = Fr(a + b, 2) + 1
            if n <= NMAXC:
                pairs.append((n, Fr(b - a, 2)))
    pairs = sorted(set(pairs))
    stats = {}; fails = []
    for i in range(len(pairs)):
        for k in range(i, len(pairs)):
            (n1, m1), (n2, m2) = pairs[i], pairs[k]
            if (n1 - n2).denominator != 1:
                continue
            r = certify(n1, m1, n2, m2)
            if r == 'empty':
                continue
            nu = max(n1, n2)
            band = '<=32' if nu <= 32 else ('=32.5' if nu == Fr(65, 2) else f'={float(nu)}')
            stats.setdefault(band, {}); stats[band][r] = stats[band].get(r, 0) + 1
            if r not in ('pd', 'kernel1'):
                fails.append(((n1, m1, n2, m2), r))
    print(f"[C] exact LDL certificate of q >= 4.558 for all pair-pairs with n1,n2 <= {float(NMAXC)}:", flush=True)
    for band in sorted(stats):
        print(f"     max(n1,n2) {band}: {sum(stats[band].values())} pair-pairs, outcomes {stats[band]}")
    print(f"     failures: {len(fails)} {fails[:5]}  ({time.time()-t1:.0f}s)", flush=True)

    # [A] analytic bounds, as written, on the cases the split must cover
    need = (Fr(4) / QC) ** 2 * 1                         # B <= (4/4.558)^2
    needf = float(need)

    def lnM(n):                                          # the mass bound 4 + 2 ln 2n (and 4N/n if m > n/2)
        return 4 + 2 * math.log(2 * n)

    def F_top(n, m, nu):                                 # case nu > n, reference at the top
        N = n - m
        t1_ = float((4 * Hn(N - 1) + 4) / ((n - m) * (n + m)))
        t2_ = float(8 * Hn(N) / (nu * nu - n * n))
        Mh = lnM(float(n))
        if m > n / 2:
            Mh = min(Mh, float(4 * N / n))
        return min(t1_, t2_) * Mh
    worstA = (0, None)
    for nu2 in range(65, 121):                           # nu = 32.5 .. 60
        nu = Fr(nu2, 2)
        for n2_ in range(4, nu2, 2):                     # n < nu, n = nu - integer
            n = Fr(n2_ + (nu2 % 2), 2)
            if n >= nu:
                continue
            for m2_ in range(0, int(2 * n) - 3, 2):
                m = Fr(m2_ + (nu2 % 2), 2) if (nu2 % 2) else Fr(m2_, 2)
                if m > n - 2:
                    continue
                v = F_top(n, m, nu)
                if v > worstA[0]:
                    worstA = (v, (float(n), float(m), float(nu)))
    print(f"[A] case nu > n, all n < nu, m, with nu = 32.5..60 (half-integer grid): max F = {worstA[0]:.4f} at (n,m,nu) = {worstA[1]}; need <= {needf:.4f}")
    worst325 = (0, None)
    nu = Fr(65, 2)
    for n2_ in range(5, 65, 2):
        n = Fr(n2_, 2)
        for m2_ in range(1, n2_ - 3, 2):
            m = Fr(m2_, 2)
            v = F_top(n, m, nu)
            if v > worst325[0]:
                worst325 = (v, (float(n), float(m)))
    print(f"    nu = 32.5 alone (outside both parts of the written proof): max F = {worst325[0]:.4f} at (n,m) = {worst325[1]}")
    # the second bound alone at nu=32.5, n=24.5 (what the text's 'n <= 24' line would give)
    print(f"    second bound alone at nu=32.5, n=24.5: 8 H_24 (4+2 ln 49)/(32.5^2-24.5^2) = "
          f"{float(8*Hn(24))*(4+2*math.log(49))/(32.5**2-24.5**2):.4f}  (> {needf:.4f}: the n<=24 line does not extend; the min with the first bound is needed)")
    # monotonicity claims for n >= 25 / 28 on a long grid (integer and half-integer n)
    g = lambda n, c: 8 * (float(Hn(int(n))) + c) * (4 + 2 * math.log(2 * n)) / n ** 2
    mono = all(g(n / 2, 1) >= g(n / 2 + 0.5, 1) for n in range(50, 4000))
    print(f"    8(H_n+1)(4+2ln2n)/n^2 nonincreasing on n = 25..2000 (step 1/2): {mono};  value at 25: {g(25,1):.4f}, at 24.5 (H_24): {g(24.5,1):.4f}")
    print(f"    8(H_n+2)(4+2ln2n)/n^2 at n=28: {g(28,2):.4f};  8H_n(4+2ln2n)/n^2 at 28: {g(28,0):.4f} (need <= {2*needf:.4f})")

    # [S] float sanity at large n
    random.seed(7)
    mn = (1e9, None); cnt = 0
    for _ in range(4000):
        nu = random.choice([random.randint(66, 200), random.randint(66, 6000)]) / 2      # nu >= 33
        if random.random() < 0.2:                                                          # nu = n (identical or not)
            n = nu
        else:
            n = nu - random.randint(1, int(nu) - 2)
        if n < 2: continue
        m1 = random.randint(0, int(2 * n) - 4) / 2; m2 = random.randint(0, int(2 * nu) - 4) / 2
        if random.random() < 0.1 and n == nu: m2 = m1
        if ((n - m1) % 1) or ((nu - m2) % 1): continue
        q = q_pencil(n, m1, nu, m2)
        if q is None: continue
        cnt += 1
        if q < mn[0]: mn = (q, (n, m1, nu, m2))
    print(f"[S] float pencil q for {cnt} random pair-pairs with max(n1,n2) >= 33 (n up to 3000): min q = {mn[0]:.4f} at {mn[1]}")
    print(f"total {time.time()-t0:.0f}s")

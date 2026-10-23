"""Exact (rational) rechecks of the SW note [SW] and of Lemma 7 of [CL] (own code).

 (H)  Lemma H: for rational m, H(m+N, m) has spectrum {N-1-2i}: the continuant p_N(lambda) vanishes at those N integers.
 (F2) end-weight ratio formulas (bottom and top), exact, from eigenvectors built by the continuant.
 (SW) sigma_min(B_{a,b})^2 >= s decided exactly.  B = rows [a+1,b+1] x cols [a,b] of H - 1 is upper-triangular banded:
      B_ii = g_{a+1+i}, B_{i,i+1} = -1, B_{i,i+2} = g_{a+2+i}.  With positive diagonals r (rows), c (cols),
      r_0 = 1, r_i = r_{i-1}/g_{a+1+i}, c_0 = 1/g_{a+1}, c_{i+1} = 1/r_i, the matrix Bt = D_r B D_c has entries 1, -1,
      g_{a+2+i}^2 (rational), and  D_c (B^T B - s) D_c = Bt^T D_r^{-2} Bt - s D_c^2  =: N_s is rational with
      the same inertia.  B^T B - s > 0  <=>  all LDL^T pivots of N_s are > 0.
      s = rational lower bound of c*^2 = (15 - sqrt 97)/6 for odd N, s = 1 for even N.
      Sharpness: N = 3, m = 0, s = rational upper bound of c*^2 must FAIL.
 (G)  Lemma Gamma of [SW] sec. 4 in exact rationals for even N <= NG; R_N recomputed from P'(1)/P(0); the analytic
      large-P chain (P >= 25) evaluated numerically for 25 <= P <= 2*10^5.
 (L7) Lemma 7 of [CL]: e^T Q^{-1} e = G_end(2)/2 < 0 for both ends, exact, even N <= 40.
"""
import sys, time, math, random
from fractions import Fraction as Fr
from math import comb, factorial, isqrt


def g2list(N, m):
    n = m + N
    return [None] + [((n * n - (m + k) ** 2) * ((m + k) ** 2 - m * m)) / (4 * (m + k) ** 2 - 1) for k in range(1, N)]


def end_weights(N, m):
    g2 = g2list(N, m)
    lams = [N - 1 - 2 * k for k in range(N)]
    w0, wN = {}, {}
    for lam in lams:
        q = [Fr(1), Fr(lam)]
        for k in range(1, N):
            q.append(lam * q[k] - g2[k] * q[k - 1])
        assert q[N] == 0, ("Lemma H fails", N, m, lam)
        prod = Fr(1); v2 = []
        for k in range(N):
            if k >= 1: prod *= g2[k]
            v2.append(q[k] * q[k] / prod)
        tot = sum(v2)
        w0[lam] = 1 / tot; wN[lam] = v2[N - 1] / tot
    return lams, w0, wN


def check_F2(N, m, lams, w0, wN):
    for k in range(1, N):
        rb = Fr(k) * (N + m - k) / ((m + k) * (N - k))
        rt = Fr(k) * (m + k) / ((N + m - k) * (N - k))
        if w0[lams[k - 1]] / w0[lams[k]] != rb or wN[lams[k - 1]] / wN[lams[k]] != rt:
            return False
    return True


def window_pd(N, m, a, b, s, g2=None):
    """True iff B_{a,b}^T B_{a,b} - s is positive definite (exact)."""
    if g2 is None: g2 = g2list(N, m)
    n = b - a + 1
    G = lambda r: g2[r]
    # squared scalings
    r2 = [Fr(1)]
    for i in range(1, n): r2.append(r2[-1] / G(a + 1 + i))
    c2 = [1 / G(a + 1)] + [1 / r2[i] for i in range(n - 1)]
    # Bt columns: Bt[i][j]
    Bt = {}
    for i in range(n):
        Bt[(i, i)] = Fr(1)
        if i + 1 < n: Bt[(i, i + 1)] = Fr(-1)
        if i + 2 < n: Bt[(i, i + 2)] = G(a + 2 + i)
    inv_r2 = [1 / x for x in r2]
    # N_s = Bt^T diag(inv_r2) Bt - s diag(c2): pentadiagonal
    Nm = {}
    for (i, j), v in Bt.items():
        for (i2, k), v2 in Bt.items():
            if i2 != i or k < j: continue
            Nm[(j, k)] = Nm.get((j, k), Fr(0)) + inv_r2[i] * v * v2
    for j in range(n):
        Nm[(j, j)] = Nm.get((j, j), Fr(0)) - s * c2[j]
    # banded LDL^T (bandwidth 2)
    A = [dict() for _ in range(n)]
    for (j, k), v in Nm.items(): A[j][k] = v
    for k in range(n):
        piv = A[k].get(k, Fr(0))
        if piv <= 0: return False
        row = sorted((j, v) for j, v in A[k].items() if j > k and v != 0)
        for t, (j, vj) in enumerate(row):
            f = vj / piv
            for (l, vl) in row[t:]:
                A[j][l] = A[j].get(l, Fr(0)) - f * vl
    return True


def cstar2_bounds(digits=40):
    S = 10 ** digits
    r = isqrt(97 * S * S)          # floor(sqrt(97) * S)
    lo = Fr(15 * S - (r + 1), 6 * S)   # (15 - sqrt97)/6 > lo
    hi = Fr(15 * S - r, 6 * S)         # < hi
    return lo, hi


def RN_formula(K):
    df = 1
    for i in range(1, 2 * K + 2, 2): df *= i
    return Fr(2 ** (2 * K + 1) * factorial(K) * factorial(K + 1), df * df)


def RN_direct(N):
    lams = [N - 1 - 2 * k for k in range(N)]
    Pp1 = Fr(1); P0 = Fr(1)
    for l in lams:
        if l != 1: Pp1 *= (1 - l)
        P0 *= (0 - l)
    return abs(Pp1 / P0)


def gamma_lemma(N):
    P = N // 2; K = P - 1
    Gam = sum(Fr(4) * (1 - Fr(comb(N - 1, K - i), comb(N - 1, K))) / ((2 * i - 1) * (2 * i + 3)) for i in range(1, K + 1)) + Fr(2 * N, N * N - 1)
    R = RN_formula(K)
    assert R == RN_direct(N)
    rhs = 2 * R * R * (N - 1) / (N * (N + 1))
    return Gam, rhs


if __name__ == '__main__':
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    t0 = time.time()
    lo, hi = cstar2_bounds()
    print(f"c*^2 in ({float(lo)!r}, {float(hi)!r}), rational bounds with 40 digits")
    ms = [Fr(0), Fr(1, 2), Fr(1), Fr(3, 2), Fr(2), Fr(5, 2), Fr(7, 2), Fr(6), Fr(19, 2), Fr(20), Fr(1, 1000), Fr(1, 3),
          Fr(37, 7), Fr(1000), Fr(10 ** 6)]
    # (H), (F2)
    nH = 0
    for N in range(2, 31):
        for m in ms:
            lams, w0, wN = end_weights(N, m)
            assert check_F2(N, m, lams, w0, wN), ("F2 fails", N, m)
            nH += 1
    print(f"(H),(F2) Lemma H and both ratio formulas exact for N = 2..30 and {len(ms)} values of m (incl. 1/1000, 1/3, 37/7, 10^6): {nH} cases, 0 failures")
    # (L7)
    nL = 0
    for N in range(2, 41, 2):
        for m in ms[:10]:
            lams, w0, wN = end_weights(N, m)
            for w in (w0, wN):
                G2 = sum(w[l] / (l - 2) for l in lams)
                assert G2 < 0, ("Lemma 7 criterion fails", N, m)
                nL += 1
    print(f"(L7) e^T Q^-1 e = G(2)/2 < 0 at both ends, even N <= 40, 10 values of m: {nL} cases, 0 failures")
    # (SW) a = 0 window, all N <= NMAX, all m in sample
    fails = []; cnt = 0
    for N in range(2, NMAX + 1):
        for m in ms:
            s = Fr(1) if N % 2 == 0 else lo
            ok = window_pd(N, m, 0, N - 2, s)
            cnt += 1
            if not ok:
                eq = N % 2 == 0 and window_pd(N, m, 0, N - 2, 1 - Fr(1, 10 ** 30))
                fails.append((N, str(m), 'sigma=1 boundary (PD at 1-1e-30)' if eq else 'FAIL'))
    print(f"(SW) window [0, N-2], N = 2..{NMAX}, {len(ms)} m-values: {cnt} cases; B^T B - s positive definite except {fails}")
    print(f"     (N = 2 is the equality case sigma = 1 of the even-N theorem: PD fails there by design)")
    extra = []
    for N in (51, 75, 101, 149, 151, 201):
        for m in (Fr(0), Fr(1, 2), Fr(7), Fr(100)):
            s = Fr(1) if N % 2 == 0 else lo
            extra.append((N, m, window_pd(N, m, 0, N - 2, s)))
    print(f"(SW) larger N: " + ", ".join(f"N={N},m={m}:{'PD' if ok else 'FAIL'}" for N, m, ok in extra))
    # all windows, small N
    cntw = 0; fw = []
    for N in range(3, 15):
        for m in ms[:9]:
            g2 = g2list(N, m)
            for a in range(0, N - 1):
                for b in range(a, N - 1):
                    s = 1 - Fr(1, 10 ** 30) if N % 2 == 0 else lo
                    cntw += 1
                    if not window_pd(N, m, a, b, s, g2): fw.append((N, m, a, b))
    print(f"(SW) every window [a,b], 3 <= N <= 14, 9 m-values: {cntw} windows, failures: {fw[:10]} (count {len(fw)})")
    sharp = window_pd(3, Fr(0), 0, 1, hi)
    sharp2 = window_pd(3, Fr(0), 0, 1, lo)
    print(f"(SW) sharpness N=3, m=0, window [0,1]: PD at s=lo(c*^2): {sharp2};  PD at s=hi(c*^2): {sharp} (must be False)")
    # random rational m
    random.seed(11); rr = 0; rf = []
    for _ in range(300):
        N = random.randint(3, 40); m = Fr(random.randint(0, 4000), random.randint(1, 97))
        s = Fr(1) if N % 2 == 0 else lo
        rr += 1
        if not window_pd(N, m, 0, N - 2, s): rf.append((N, m))
    print(f"(SW) 300 random (N <= 40, rational m <= 4000): failures {rf}")
    # (G)
    gfail = []; ratios = {}
    for N in range(2, 201, 2):
        Gam, rhs = gamma_lemma(N)
        if Gam < rhs: gfail.append(N)
        if N in (2, 4, 6, 10, 50, 100, 200): ratios[N] = float(Gam / rhs)
    print(f"(G) Lemma Gamma exact, even N <= 200: failures {gfail}; Gamma_N/rhs at N=2,4,6,10,50,100,200: {ratios}")
    worstP = None
    for P in range(25, 200001):
        lhs = (math.atan(math.sqrt(P)) - 2 / math.sqrt(P - 1)) / math.sqrt(P)
        rhsP = 4 / P
        r = lhs / rhsP
        if worstP is None or r < worstP[0]: worstP = (r, P)
    print(f"(G) analytic chain: min over 25 <= P <= 2e5 of [(atan sqrt P - 2/sqrt(P-1))/sqrt P] / (4/P) = {worstP[0]:.4f} at P = {worstP[1]} (> 1 needed)")
    # large-P chain ingredients spot-checked exactly: rho_i <= exp(-i(i+1)/(P+i)), each summand >= 1/(P+i^2+2i)
    bad = 0
    for P in (25, 30, 60, 100):
        N = 2 * P; K = P - 1
        for i in range(1, K + 1):
            rho = Fr(comb(N - 1, K - i), comb(N - 1, K))
            if float(rho) > math.exp(-i * (i + 1) / (P + i)) * (1 + 1e-12): bad += 1
            term = 4 * (1 - rho) / ((2 * i - 1) * (2 * i + 3))
            if term < Fr(1, P + i * i + 2 * i): bad += 1
    print(f"(G) chain inequalities rho_i <= exp(-i(i+1)/(P+i)) and summand >= 1/(P+i^2+2i) at P = 25, 30, 60, 100: {bad} violations")
    print(f"total {time.time()-t0:.0f}s")

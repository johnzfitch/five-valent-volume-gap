"""Referee: high-precision q_min for the Theorem-1 chain, two independent methods.
Method S: Sturm (LDL^T inertia) bisection on the 2N x 2N zero-diagonal Jacobi matrix.
Method P: power iteration for ||B^{-1}||_2 where B^{-1} is obtained by forward
substitution on the bidiagonal block (not from the paper's closed form).
Also checks the paper's closed-form P_N, the ratio inequalities, and kappa."""
import sys, time
import mpmath as mp

def couplings(N):
    D = [2 * l * mp.sqrt((N + 1) ** 2 - l ** 2) for l in range(1, N + 1)]          # odd links
    X = [4 * mp.sqrt((N - l) * (N + l + 1) * l * (l + 1)) for l in range(1, N)]    # even links
    c = []
    for l in range(N):
        c.append(D[l])
        if l < N - 1: c.append(X[l])
    return c, D, X

def count_below(c, x):
    """# eigenvalues < x of zero-diagonal Jacobi with couplings c."""
    n = len(c) + 1; cnt = 0; d = -x
    if d < 0: cnt += 1
    for k in range(1, n):
        if d == 0: d = mp.mpf(10) ** (-mp.mp.dps)
        d = -x - c[k - 1] ** 2 / d
        if d < 0: cnt += 1
    return cnt

def qmin_sturm(c, N):
    lo, hi = mp.mpf(0), mp.mpf(1)
    while count_below(c, hi) <= N: hi *= 2
    # log bisection then linear
    lo = hi / mp.mpf(2) ** (3 * N + 20)
    assert count_below(c, lo) == N
    for _ in range(400):
        mid = mp.sqrt(lo * hi) if hi / lo > 2 else (lo + hi) / 2
        if count_below(c, mid) > N: hi = mid
        else: lo = mid
        if (hi - lo) / hi < mp.mpf(10) ** (-30): break
    return (lo + hi) / 2

def qmin_power(D, X, N):
    # B lower bidiagonal: B[i][i]=D_i, B[i+1][i]=X_i ; columns of B^{-1} by forward substitution
    Binv = [[mp.mpf(0)] * N for _ in range(N)]
    for j in range(N):
        v = [mp.mpf(0)] * N
        v[j] = 1 / D[j]
        for i in range(j + 1, N):
            v[i] = -X[i - 1] * v[i - 1] / D[i]
        for i in range(N): Binv[i][j] = v[i]
    x = [mp.mpf(1)] * N; s_old = 0
    for it in range(500):
        y = [mp.fsum(Binv[i][j] * x[j] for j in range(N)) for i in range(N)]
        z = [mp.fsum(Binv[i][j] * y[i] for i in range(N)) for j in range(N)]
        nz = mp.sqrt(mp.fsum(t * t for t in z)); x = [t / nz for t in z]
        s = mp.sqrt(nz)
        if it > 5 and abs(s - s_old) / s < mp.mpf(10) ** (-30): break
        s_old = s
    return 1 / s

if __name__ == "__main__":
    Nmax = int(sys.argv[1]); outp = sys.argv[2]
    lines = []
    def log(s): print(s, flush=True); lines.append(s)
    kappa = None
    mp.mp.dps = 50
    kappa = 1 / mp.sqrt(mp.mpf(16) / 3 * mp.log(mp.mpf(4) / 3))
    log(f"kappa = {mp.nstr(kappa, 20)}")
    Ns = list(range(2, 101)) + list(range(110, Nmax + 1, 10))
    worst_lo, worst_hi = 10, 0; ratio_fail = 0; t0 = time.time()
    for N in Ns:
        mp.mp.dps = int(0.32 * N) + 45
        c, D, X = couplings(N)
        P = 4 * N * mp.sqrt(2 * N + 1) / mp.mpf(2) ** N
        E1 = mp.fprod(X) / mp.fprod(D)
        idP = abs(1 / E1 - P) / P
        # ratio inequalities used in the lower bound
        r1 = min((X[l - 1] / D[l - 1]) ** 2 for l in range(1, N)) if N > 1 else mp.inf
        r2 = min((X[l - 1] / D[l]) ** 2 for l in range(1, N)) if N > 1 else mp.inf
        if r1 < 2 * (1 - mp.mpf(10)**-40) or r2 < mp.mpf(8) / 5 * (1 - mp.mpf(10)**-40): ratio_fail += 1
        qs = qmin_sturm(c, N)
        qp = qmin_power(D, X, N) if (N <= 120 or N % 50 == 0) else None
        r = qs / P
        worst_lo = min(worst_lo, r); worst_hi = max(worst_hi, r)
        if N <= 12 or N % 10 == 0:
            log(f"N={N:4d} q_min(Sturm)={mp.nstr(qs, 15):>24} "
                + (f"q_min(power)rel.diff={mp.nstr(abs(qp-qs)/qs, 3):>9} " if qp is not None else " " * 34)
                + f"q/P={mp.nstr(r, 12)}  kappa-q/P={mp.nstr(kappa - r, 4)}  |1/E_N1-P|/P={mp.nstr(idP, 3)} "
                + f"min(X/D)^2={mp.nstr(r1, 5)} min(X_l/D_l+1)^2={mp.nstr(r2, 5)}")
    log(f"over N in {Ns[0]}..{Ns[-1]} ({len(Ns)} values): min q/P = {mp.nstr(worst_lo, 12)}, max q/P = {mp.nstr(worst_hi, 12)}; "
        f"ratio-inequality failures = {ratio_fail}; time {time.time()-t0:.0f}s")
    open(outp, "w").write("\n".join(lines) + "\n")

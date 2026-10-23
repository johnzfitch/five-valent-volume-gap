"""Referee check R4: adversarial attack on Lemma 1 (abstract zero-diagonal Jacobi lemma).

Objective (i):  f = lambda_min^+ * S * sqrt(rho+ rho-)       (Lemma 1(i) claims f >= 1)
Objective (ii): g = lambda_min^+ * S_odd / 2   (even n)      (Lemma 1(ii) claims g <= 1)
over log-concave positive chains c_1..c_{n-1}.

Search: structured families + local optimisation (Nelder-Mead / Powell) from many random
starts over a parametrisation that is log-concave by construction:
  x_1 = 0, Delta_1 = p_0, Delta_s = Delta_{s-1} - exp(p_s) (strictly decreasing increments),
and also chains with exactly constant increments (the log-concavity boundary).
Best points are re-evaluated with 50-digit mpmath eigenvalues.
"""
import sys, math, time
import numpy as np
import mpmath as mp
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize

rng = np.random.default_rng(7)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

def lam(c):
    n = len(c)+1
    w = np.sort(eigvalsh_tridiagonal(np.zeros(n), c))
    return w[(n+1)//2]

def stats(c):
    r = c[1:]/c[:-1] if len(c) > 1 else np.array([1.0])
    rp = max(1.0, r.max()); rm = max(1.0, (1/r).max())
    S = np.sum(1/c); So = np.sum(1/c[0::2])
    return S, So, rp, rm

def f_i(c):
    S, So, rp, rm = stats(c); return lam(c)*S*math.sqrt(rp*rm)
def g_ii(c):
    S, So, rp, rm = stats(c); return lam(c)*So/2

def chain_from(p, n):
    m = n-1
    inc = [p[0]]
    for s in range(1, m-1):
        inc.append(inc[-1]-math.exp(min(p[s], 30)))
    x = np.concatenate([[0.0], np.cumsum(inc)]) if m > 1 else np.array([0.0])
    x = x - x.max()
    return np.exp(np.clip(x, -300, 0))

def lam_mp(c, dps=60, iters=260):
    """smallest positive eigenvalue by Sturm counting (LDL^T pivots) in mpmath, geometric bisection."""
    mp.mp.dps = dps
    n = len(c)+1; n0 = (n+1)//2   # number of eigenvalues <= 0 (incl. zero for odd n)
    c2 = [mp.mpf(float(v))**2 for v in c]
    def count_below(x):
        q = -x; cnt = 1 if q < 0 else 0
        for v in c2:
            if q == 0: q = mp.mpf(10)**(-dps*3)
            q = -x - v/q; cnt += q < 0
        return cnt
    lo = mp.log(mp.mpf(float(min(c))))-mp.mpf(400); hi = mp.log(2*mp.mpf(float(max(c))))
    for _ in range(iters):
        mid = (lo+hi)/2
        if count_below(mp.exp(mid)) > n0: hi = mid
        else: lo = mid
    return mp.exp((lo+hi)/2)

def f_mp(c):
    cm = [mp.mpf(float(v)) for v in c]
    S = sum(1/v for v in cm)
    r = [cm[i+1]/cm[i] for i in range(len(cm)-1)] or [mp.mpf(1)]
    rp = max([mp.mpf(1)]+r); rm = max([mp.mpf(1)]+[1/v for v in r])
    return lam_mp(c)*S*mp.sqrt(rp*rm)

def g_mp(c):
    cm = [mp.mpf(float(v)) for v in c]
    return lam_mp(c)*sum(1/v for v in cm[0::2])/2

def mp_check(c, which):
    mp.mp.dps = 50
    n = len(c)+1
    A = mp.zeros(n, n)
    for i, v in enumerate(c):
        A[i, i+1] = A[i+1, i] = mp.mpf(float(v))
    ev = sorted(mp.eigsy(A, eigvals_only=True))
    l = ev[(n+1)//2]
    cm = [mp.mpf(float(v)) for v in c]
    S = sum(1/v for v in cm); So = sum(1/v for v in cm[0::2])
    r = [cm[i+1]/cm[i] for i in range(len(cm)-1)] or [mp.mpf(1)]
    rp = max([mp.mpf(1)]+r); rm = max([mp.mpf(1)]+[1/v for v in r])
    return l*S*mp.sqrt(rp*rm) if which == "i" else l*So/2

t0 = time.time()
# ---------------- structured families
fam = []
for n in (2, 3, 4, 5, 6, 8, 10, 20, 50, 100, 200):
    m = n-1
    for r in (1.0, 1.01, 1.1, 1.5, 2, 3, 10, 100):
        for kind in ("geom_up", "geom_down", "tent", "vee_flat"):
            s = np.arange(m)
            if kind == "geom_up": x = s*math.log(r)
            elif kind == "geom_down": x = -s*math.log(r)
            elif kind == "tent": x = -np.abs(s-(m-1)/2)*math.log(r)
            else: x = -np.maximum(0, s-m//2)*math.log(r)   # flat then decreasing (log-concave)
            if x.max()-x.min() > 600: continue
            c = np.exp(x-x.max())
            if c.min() < 1e-6:
                fam.append((float(f_mp(c)), float(g_mp(c)) if n % 2 == 0 else 0, n, r, kind))
            else:
                fam.append((f_i(c), g_ii(c) if n % 2 == 0 else 0, n, r, kind))
fmin = min(fam, key=lambda z: z[0]); gmax = max(fam, key=lambda z: z[1])
log(f"R4 structured families ({len(fam)} chains): min f = {fmin[0]:.6f} at n={fmin[2]}, r={fmin[3]}, {fmin[4]}; max g (even n) = {gmax[1]:.6f} at n={gmax[2]}, r={gmax[3]}, {gmax[4]}")
for n in (2, 4, 10, 50, 200):
    c = np.ones(n-1)
    log(f"   constant chain n={n}: lambda*S = {lam(c)*np.sum(1/c):.6f}, f = {f_i(c):.6f}")
# ---------------- optimisation
best_f = (np.inf, None); best_g = (-np.inf, None)
for trial in range(1500):
    n = int(rng.integers(3, 31))
    m = n-1
    p0 = np.concatenate([[rng.normal(0, 2)], rng.normal(-1, 2, size=max(m-2, 0))])
    if len(p0) == 0: continue
    def obj(p):
        c = chain_from(p, n)
        if c.min() < 1e-12: return 1e6
        return f_i(c)
    res = minimize(obj, p0, method="Nelder-Mead" if trial % 2 else "Powell", options={"maxiter": 4000})
    if res.fun < best_f[0]: best_f = (res.fun, chain_from(res.x, n))
    if n % 2 == 0:
        def objg(p):
            c = chain_from(p, n)
            if c.min() < 1e-12: return 1e6
            return -g_ii(c)
        res = minimize(objg, p0, method="Nelder-Mead" if trial % 2 else "Powell", options={"maxiter": 4000})
        if -res.fun > best_g[0]: best_g = (-res.fun, chain_from(res.x, n))
log(f"R4 optimisation, 1500 starts, 3<=n<=30: min f found = {best_f[0]:.9f} (n={len(best_f[1])+1}); max g found = {best_g[0]:.6f} (n={len(best_g[1])+1})  ({time.time()-t0:.0f}s)")
log(f"   best-f chain (rounded): {np.round(best_f[1], 6).tolist()[:12]}")
log(f"   mpmath 50-digit re-evaluation: f = {mp.nstr(mp_check(best_f[1], 'i'), 15)}, g = {mp.nstr(mp_check(best_g[1], 'ii'), 15)}")
# ---------------- random log-concave, large dynamic range, checked in mpmath when float is doubtful
worst = np.inf; worstc = None; N = 0
for trial in range(20000):
    n = int(rng.integers(2, 41))
    m = n-1
    inc = np.sort(rng.normal(0, rng.uniform(0.01, 3.0), size=max(m-1, 0)))[::-1]
    x = np.concatenate([[0.0], np.cumsum(inc)]); x -= x.max()
    if x.min() < -60: continue
    c = np.exp(x); N += 1
    v = f_i(c) if x.min() > -np.log(1e6) else float(f_mp(c))
    if v < worst: worst, worstc = v, c
log(f"R4 random log-concave chains, dynamic range up to e^60: {N} chains, min f = {worst:.6f} (n={len(worstc)+1}); mpmath: {mp.nstr(mp_check(worstc,'i'),12)}")
# ---------------- the boundary: what fails without log-concavity, and without bounded ratios
c = np.array([1.0 if k % 2 == 0 else 1e3 for k in range(9)])
log(f"R4 non-log-concave dimerised (1,1e3,1,...,1), n=10: f = {mp.nstr(f_mp(c),4)}  (hypothesis violated; f<1 shows log-concavity is needed)")
open(sys.argv[1] if len(sys.argv) > 1 else "r4_out.txt", "w").write("\n".join(out)+"\n")

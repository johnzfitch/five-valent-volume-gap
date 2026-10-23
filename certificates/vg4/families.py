"""Targeted extreme families to large spin (CPU; single tridiagonal matrices).

For each family j -> (j1,j2,j3,j4) we report d, lam = smallest positive eigenvalue of Q,
lam / (sqrt(j_max(j_max+1))/2), lam*S (S = sum 1/a_k), and lam / WKB prediction.
LAPACK (scipy eigvalsh_tridiagonal, index-selected) is cross-checked against a 60-digit
mpmath Sturm bisection on a subset.
Run: python families.py  -> families_out.txt
"""
import sys, os, math
import numpy as np
import mpmath as mp
from scipy.linalg import eigvalsh_tridiagonal
from scipy.special import ellipk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vg_core as vg

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def chain(T):
    J1, J2, J3, J4 = T
    kmin2, kmax2 = vg.krange2(*T)
    K = np.arange(kmin2 + 2, kmax2 + 1, 2, dtype=np.float64) / 2
    j1, j2, j3, j4 = (x / 2 for x in T)
    num = (((j1 + j2 + 1) ** 2 - K * K) * (K * K - (j1 - j2) ** 2) * ((j3 + j4 + 1) ** 2 - K * K) * (K * K - (j3 - j4) ** 2))
    return np.sqrt(num / (16 * (4 * K * K - 1)))


def lam_lapack(c):
    d = len(c) + 1
    n0 = (d + 1) // 2
    w = eigvalsh_tridiagonal(np.zeros(d), c, select="i", select_range=(n0, n0))
    return float(w[0])


def lam_mp(T, dps=60):
    mp.mp.dps = dps
    J1, J2, J3, J4 = T
    kmin2, kmax2 = vg.krange2(*T)
    j1, j2, j3, j4 = (mp.mpf(x) / 2 for x in T)
    c2 = []
    for K2 in range(kmin2 + 2, kmax2 + 1, 2):
        k = mp.mpf(K2) / 2
        c2.append(((j1 + j2 + 1) ** 2 - k * k) * (k * k - (j1 - j2) ** 2) * ((j3 + j4 + 1) ** 2 - k * k) * (k * k - (j3 - j4) ** 2) / (16 * (4 * k * k - 1)))
    d = len(c2) + 1; n0 = (d + 1) // 2
    lo, hi = mp.mpf(0), 2 * mp.sqrt(max(c2))
    for _ in range(260):
        x = (lo + hi) / 2
        q = -x; cnt = 1 if q < 0 else 0
        for v in c2:
            q = -x - v / q
            cnt += 1 if q < 0 else 0
        if cnt > n0:
            hi = x
        else:
            lo = x
    return (lo + hi) / 2


def wkb(T, d):
    A = np.array(T, dtype=np.float64) / 2 + 0.5
    A1, A2, A3, A4 = A
    r = np.sort([(A1 - A2) ** 2, (A3 - A4) ** 2, (A1 + A2) ** 2, (A3 + A4) ** 2])
    den = (r[3] - r[1]) * (r[2] - r[0])
    m = (r[2] - r[1]) * (r[3] - r[0]) / den
    if m >= 1 - 1e-15:
        return float("nan"), m
    T0 = 8 * ellipk(m) / math.sqrt(den)
    return (2.0 if d % 2 else 1.0) * math.pi / T0, m


fams = {
    "(1/2,1/2,j,j)": lambda J: (1, 1, J, J),
    "(1,1,j,j)": lambda J: (2, 2, J, J),
    "(3/2,3/2,j,j)": lambda J: (3, 3, J, J),
    "(1/2,j,j,j) [d=2]": lambda J: (1, J + 1, J + 1, J + 1),
    "(j,j,j,j) monochromatic": lambda J: (J, J, J, J),
    "(j,j,j,j+1)": lambda J: (J, J, J, J + 2),
    "(j,j,2j,2j)": lambda J: (J, J, 2 * J, 2 * J),
    "(j,2j,3j,4j)": lambda J: (J, 2 * J, 3 * J, 4 * J),
    "(j,j,j,3j-1) near-flat": lambda J: (J, J, J, 3 * J - 2),
    "(1,j,j,2j-1)": lambda J: (2, J, J, 2 * J - 2),
    "(j,j,j^2/4,j^2/4) two scales": lambda J: (J, J, (J * J) // 4 * 2 // 2, (J * J) // 4 * 2 // 2),
}
Js = [8, 20, 50, 100, 200, 400, 1000, 2000, 4000]
for name, f in fams.items():
    log(f"--- family {name}")
    for J in Js:
        T = f(J)
        if T is None or not vg.admissible(*T) or T[3] > T[0] + T[1] + T[2] + 0:
            continue
        T = tuple(int(x) for x in T)
        if vg.dim(*T) < 2 or max(T) > 400000:
            continue
        c = chain(T)
        d = len(c) + 1
        lam = lam_lapack(c)
        S = float(np.sum(1 / c))
        jm = max(T) / 2
        p, m = wkb(T, d)
        log(f"  j={J/2:>7} spins={tuple(x/2 for x in T)} d={d} lam={lam:.10g} lam/(sqrt(jm(jm+1))/2)={lam/(0.5*math.sqrt(jm*(jm+1))):.6f} lam*S={lam*S:.6f} lam/wkb={lam/p if p==p else float('nan'):.6f} m={m:.6f}")

log("--- 60-digit mpmath Sturm cross-check of LAPACK")
for T in [(1, 1, 400, 400), (100, 100, 100, 100), (400, 400, 400, 400), (40, 80, 120, 160), (200, 200, 200, 202), (2, 2, 1000, 1000), (60, 60, 60, 178)]:
    c = chain(T)
    a = lam_lapack(c); b = lam_mp(T)
    log(f"  {tuple(x/2 for x in T)}: lapack {a:.15g}  mpmath {mp.nstr(b, 20)}  rel diff {float(abs(a-b)/b):.2e}")

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "families_out.txt"), "w").write("\n".join(out) + "\n")

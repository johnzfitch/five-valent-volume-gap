"""Data files for the paper's plots (pgfplots reads outputs/*.dat).

fig_t.dat     log10 q_min(N) on the spin-1/2 chain of A = -4i[J1.(J2xJ3) + t J1.(J4xJ5)],
              spins (1/2; j-1/2, j; j, j), N = 2j, for t = 1/4, 1/2, 3/4, 9/10, 1 (mpmath).
fig_growth.dat  t = 1: q_min along (1/2; j-1/2, j, j, j) against M = 2j_max (scipy tridiagonal),
              with pi M / ln M; and q_min along (s; 1/2, 1, 1, s-1/2) against 2s.
"""
import sys
import numpy as np
import mpmath as mp
from scipy.linalg import eigvalsh_tridiagonal
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
import grid

mp.mp.dps = 60


def chain(N, al, be):
    ch = []
    for l in range(1, N + 1):
        ch.append(be * l * mp.sqrt((N + 1) ** 2 - l ** 2))
        if l < N:
            ch.append(al * mp.sqrt((N - l) * (N + l + 1) * l * (l + 1)))
    return ch


def qmin_mp(ch):
    # smallest singular value of the bidiagonal block via Sturm bisection on J^2 is overkill;
    # the 2N x 2N Jacobi matrix is small enough for mp.eigsy at N <= 40
    n = len(ch) + 1
    J = mp.zeros(n, n)
    for i, w in enumerate(ch):
        J[i, i + 1] = J[i + 1, i] = w
    return min(abs(e) for e in mp.eigsy(J, eigvals_only=True))


ts = [mp.mpf(1) / 4, mp.mpf(1) / 2, mp.mpf(3) / 4, mp.mpf(9) / 10, mp.mpf(1)]
with open(sys.argv[1] + "/fig_t.dat", "w") as f:
    f.write("N " + " ".join(f"t{int(t * 100)}" for t in ts) + "\n")
    for N in range(2, 41, 2):
        vals = [mp.log10(qmin_mp(chain(N, 1, t))) for t in ts]
        f.write(f"{N} " + " ".join(mp.nstr(v, 8) for v in vals) + "\n")

with open(sys.argv[1] + "/fig_growth.dat", "w") as f:
    f.write("M q_G1 piMlnM\n")
    for k in range(2, 13):
        N = 2 ** k          # N = 2j = M
        ch = np.array([float(w) for w in chain(N, 2, 2)])
        ev = eigvalsh_tridiagonal(np.zeros(len(ch) + 1), ch, select='a')
        q = np.min(np.abs(ev))
        f.write(f"{N} {q:.8e} {np.pi * N / np.log(N):.8e}\n")
with open(sys.argv[1] + "/fig_slope.dat", "w") as f:
    f.write("twos q slope\n")
    for ts_ in range(3, 81, 2):
        q, _ = grid.qmin((ts_, 1, 2, 2, ts_ - 1), 2, 2)
        f.write(f"{ts_} {q:.8f} {4 * np.sqrt(10 - 6 * np.sqrt(2)) * ts_ / 2:.8f}\n")

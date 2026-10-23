"""Checks of the identities and constants used in the paper, in the paper's notation.

All exact where stated (sympy / Fractions), otherwise float64 or mpmath.
Output: outputs/checks_out.txt
"""
import itertools, sys
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
import sympy as S
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
import su2, grid

out = open(sys.argv[1] if len(sys.argv) > 1 else "outputs/checks_out.txt", "w")


def say(*a):
    print(*a, file=out); out.flush()


# C1 Stark matrix: integer spectrum N-1, N-3, ..., at integer and half-integer and irrational m
worst = 0
for N in range(1, 25):
    for m in [0, 0.5, 1, 1.5, 3, 7.5, 0.3141, 12.77]:
        ev = np.sort(np.linalg.eigvalsh(su2.stark(m + N, m)))
        worst = max(worst, np.max(np.abs(ev - np.arange(-(N - 1), N, 2))))
say(f"C1 Stark matrix H(n,m) has spectrum N-1,N-3,...: N<=24, 8 values of m, max dev {worst:.1e}")
# symbolic in m for small N
m = S.symbols('m', nonnegative=True)
x = S.symbols('x')
okH = True
for N in range(1, 9):
    M = S.zeros(N, N)
    for s_ in range(1, N):
        L = m + s_
        h2 = ((m + N) ** 2 - L ** 2) * (L ** 2 - m ** 2) / (4 * L ** 2 - 1)
        M[s_ - 1, s_] = M[s_, s_ - 1] = S.sqrt(h2)
    cp = S.simplify(S.expand((M - x * S.eye(N)).det()))
    target = S.expand(S.prod([(-1) * (x - (N - 1 - 2 * k)) for k in range(N)]))
    okH &= S.simplify(cp - target) == 0
say(f"C1' characteristic polynomial of H(m+N,m) is identically prod(x-(N-1-2k)) in m, N<=8: {okH}")

# C2 Heron-Stark: P(b;s,c) = F(d-1/2) G(u), and the D-link companion
b, c, s = S.symbols('b c s')
P = ((s + c + 1) ** 2 - b ** 2) * (b ** 2 - (s - c) ** 2)
F = lambda t: (s + S.Rational(1, 2)) ** 2 - t ** 2
G = (b + c + s + 1) * (b + c - s)
P2 = ((s + b + 1) ** 2 - c ** 2) * (c ** 2 - (s - b) ** 2)
say(f"C2 Heron-Stark factorisation P(b;s,c)=F(d-1/2)G(u): {S.expand(P - F(b - c - S.Rational(1, 2)) * G) == 0}; "
    f"P(c;s,b)=F(d+1/2)G(u): {S.expand(P2 - F(b - c + S.Rational(1, 2)) * G) == 0}")

# C3 the spin-1/2 chain of the normal form; telescoping product; Proposition T bound
mp.mp.dps = 80


def chain_links(N, al, be):
    """Spin (1/2; j-1/2, j; j, j), N=2j: D_l, l=1..N and X_l, l=1..N-1 (links of iA)."""
    D = [be * l * mp.sqrt((N + 1) ** 2 - l ** 2) for l in range(1, N + 1)]
    X = [al * mp.sqrt((N - l) * (N + l + 1) * l * (l + 1)) for l in range(1, N)]
    ch = []
    for l in range(N):
        ch.append(D[l])
        if l < N - 1:
            ch.append(X[l])
    return ch


def qmin_chain(ch):
    n = len(ch) + 1
    J = mp.zeros(n, n)
    for i, w in enumerate(ch):
        J[i, i + 1] = J[i + 1, i] = w
    ev = mp.eigsy(J, eigvals_only=True)
    return min(abs(e) for e in ev)


worst = 0
for N in range(2, 9):
    sp = (1, N - 1, N, N, N)
    q1, _ = grid.qmin(sp, 4, 2)
    q2 = qmin_chain(chain_links(N, 4, 2))
    worst = max(worst, abs(q1 - float(q2)) / q1)
say(f"C3a grid with s=1/2 equals the explicit chain D_l, X_l (alpha,beta)=(4,2), N=2..8: max rel diff {worst:.1e}")
rows = []
for N in [2, 3, 5, 10, 20, 40, 80]:
    ch = chain_links(N, 4, 2)
    PN = 4 * N * mp.sqrt(2 * N + 1) * mp.mpf(2) ** (-N)
    prodD = mp.fprod(ch[0::2]); prodX = mp.fprod(ch[1::2])
    q = qmin_chain(ch)
    rows.append(f"N={N}: q={mp.nstr(q, 10)}, P_N={mp.nstr(PN, 10)}, prodD/prodX/P_N={mp.nstr(prodD / prodX / PN, 12)}, q/P_N={mp.nstr(q / PN, 6)}")
say("C3b telescoping and bounds (3/8)P_N <= q <= P_N, kappa = " + mp.nstr(1 / mp.sqrt(mp.mpf(16) / 3 * mp.log(mp.mpf(4) / 3)), 10))
for r in rows:
    say("   " + r)
for t in [mp.mpf('0.25'), mp.mpf('0.5'), mp.mpf('0.75'), mp.mpf('0.9')]:
    vals = []
    for N in [10, 20, 40]:
        ch = chain_links(N, 1, t)
        q = qmin_chain(ch)
        bound = N * mp.sqrt(2 * N + 1) * t ** N
        vals.append(f"N={N}: q={mp.nstr(q, 6)} <= {mp.nstr(bound, 6)}: {q <= bound}")
    say(f"C3c Proposition T, alpha=1, beta=t={t}: " + "; ".join(vals))

# C4 the conjectured sharp t=1 constant at (1/2; 1/2,1; 1,1)
sp = (1, 1, 2, 2, 2)
S_, L_ = grid.links(sp, 2, 2)
idx = {xx: i for i, xx in enumerate(sorted(S_))}
W = S.zeros(len(S_), len(S_))
for (p_, q_), w2 in L_.items():
    W[idx[p_], idx[q_]] = W[idx[q_], idx[p_]] = S.sqrt(S.Rational(w2.numerator, w2.denominator))
cp = S.factor((W - x * S.eye(len(S_))).det())
qstar = 2 * S.sqrt(18 - 2 * S.sqrt(41))
say(f"C4 t=1 grid at (1/2;1/2,1;1,1): squared links {sorted(str(v) for v in L_.values())}, char. poly {cp}, "
    f"smallest positive root = {S.N(min(r for r in S.solve(S.expand(cp), x) if S.N(r) > 0), 15)}, 2sqrt(18-2sqrt41) = {S.N(qstar, 15)}")

# C5 the sharp slope: closed form along (s; 1/2, 1, 1, s-1/2), legs placed (s; 1/2,1 | 1, s-1/2)
worst = 0
for ts in range(3, 61, 2 if False else 1):
    sp = (ts, 1, 2, 2, ts - 1)
    q, _ = grid.qmin(sp, 2, 2)
    s_ = ts / 2
    xm2 = 5 * s_ ** 2 / 2 + 3 * s_ / 2 - 9 / 4 - (3 * s_ + 4) * np.sqrt(8 * s_ ** 2 - 12 * s_ + 5) / 4
    worst = max(worst, abs(q - 8 * np.sqrt(xm2)) / q)
nu0 = np.sqrt((5 - 3 * np.sqrt(2)) / 2)
say(f"C5 closed form q_min = 8 x_-(s) on (s;1/2,1,1,s-1/2), 2s=3..60: max rel diff {worst:.1e}; slope 8 nu0 = 4 sqrt(10-6 sqrt2) = {8 * nu0:.12f}")
say(f"   q at 2s=3: {grid.qmin((3, 1, 2, 2, 2), 2, 2)[0]:.10f}; at 2s=4: {grid.qmin((4, 1, 2, 2, 3), 2, 2)[0]:.10f}")

# C6 t = 0: 2sqrt3 (labelled), and 4 sqrt2 when the triple contains the largest sorted spin
best = 1e9
for t in itertools.product(range(1, 9), repeat=5):
    if sum(t) % 2:
        continue
    q, _ = grid.qmin(t, 2, 0)
    if q is not None:
        best = min(best, q)
say(f"C6 t=0 (A = -8i J1.(J2xJ3)), all labelled 2j_i <= 8: min q = {best:.12f}, 2 sqrt3 = {2 * np.sqrt(3):.12f}")

# C7 SW constant at N = 3, m = 0
H = su2.stark(3, 0) - np.eye(3)
Bw = H[1:3, 0:2]
say(f"C7 window of H(3,0)-1, rows 1..2 x cols 0..1: sigma_min = {np.linalg.svd(Bw, compute_uv=False).min():.12f}, "
    f"sqrt((15-sqrt97)/6) = {np.sqrt((15 - np.sqrt(97)) / 6):.12f}")

# C8 small-j1 minima of the t=1 operator over legs 2..5 (multisets, S4 symmetry), 2j <= 16
for ts in (1, 2, 3, 4, 5):
    best = (1e9, None)
    for legs in itertools.combinations_with_replacement(range(1, 17), 4):
        if (ts + sum(legs)) % 2:
            continue
        sp = (ts,) + legs
        q, _ = grid.qmin(sp, 2, 2)
        if q is not None and q < best[0]:
            best = (q, sp)
    say(f"C8 t=1, 2j1={ts}, legs 2j<=16: min q = {best[0]:.10f} at 2j = {best[1]}")
out.close()

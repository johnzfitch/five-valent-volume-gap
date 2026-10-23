"""Referee: Theorem 2 grid. Couplings come from Proposition 1 (referee implementation),
compared exactly with the paper's closed forms D_m, X_m; product identity exact;
q_min/P by Sturm bisection in mpmath; Theorem-1 closed forms vs Prop-1 chain to N=60."""
import sys
from fractions import Fraction as Fr
import mpmath as mp
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from ref_exact import chain_sq_couplings
from ref_bounds import count_below, qmin_sturm
out = []
def log(s): print(s, flush=True); out.append(s)
H = Fr(1, 2)
# Theorem 1 closed forms vs Prop 1
bad = 0
for N in range(2, 61):
    j = Fr(N, 2)
    sites, sq = chain_sq_couplings(j - H, j, j, j, 4, 2)
    D2 = [4 * l * l * ((N + 1) ** 2 - l * l) for l in range(1, N + 1)]
    X2 = [16 * (N - l) * (N + l + 1) * l * (l + 1) for l in range(1, N)]
    want = []
    for l in range(N):
        want.append(D2[l])
        if l < N - 1: want.append(X2[l])
    bad += (sq != want) or len(sites) != 2 * N
log(f"Theorem 1: closed-form D_l, X_l equal Prop-1 couplings exactly for N=2..60: {bad == 0}")
# Theorem 2 grid
Mmax, Dmax = int(sys.argv[1]), int(sys.argv[2])
nform = nprod = nratio = 0; lo, hi = 10, 0; cnt = 0
for M in range(1, Mmax + 1):
    for tD in range(1, Dmax + 1):
        D = Fr(tD, 2)
        jj = [Fr(M, 2), Fr(M, 2) + D, Fr(M + 1, 2), Fr(M, 2) + D]
        sites, sq = chain_sq_couplings(*jj, 4, 2)
        Dm = [4 * (M + 1 - m) * (M + 2 * D + 2 + m) * (m + 1) * (2 * D + m) for m in range(M + 1)]
        Xm = [16 * (M - m) * (M + 2 * D + 2 + m) * (m + 1) * (2 * D + 1 + m) for m in range(M)]
        want = []
        for m in range(M + 1):
            want.append(Dm[m])
            if m < M: want.append(Xm[m])
        nform += (sq == want and len(sites) == 2 * (M + 1))
        num = Fr(1); den = Fr(1)
        for s in sq[0::2]: num *= s
        for s in sq[1::2]: den *= s
        P2 = Fr(4) ** (1 - M) * (M + 1) ** 2 * 4 * D * (M + D + 1)
        nprod += (num / den == P2)
        r1 = min((Xm[m] / Fr(Dm[m]) for m in range(M)), default=99)
        r2 = min((Xm[m] / Fr(Dm[m + 1]) for m in range(M)), default=99)
        nratio += (r1 >= 2 and r2 >= Fr(8, 5))
        mp.mp.dps = int(0.32 * M) + 40
        c = [mp.sqrt(mp.mpf(s.numerator) / s.denominator) for s in sq]
        q = qmin_sturm(c, M + 1)
        P = mp.sqrt(mp.mpf(P2.numerator) / P2.denominator)
        r = q / P; lo = min(lo, r); hi = max(hi, r); cnt += 1
log(f"Theorem 2 grid M<= {Mmax}, 2Delta <= {Dmax} ({cnt} pairs): closed forms match Prop 1: {nform}/{cnt}; "
    f"product identity exact: {nprod}/{cnt}; ratio inequalities hold: {nratio}/{cnt}; q/P in [{mp.nstr(lo,6)}, {mp.nstr(hi,6)}]")
open(sys.argv[3], "w").write("\n".join(out) + "\n")

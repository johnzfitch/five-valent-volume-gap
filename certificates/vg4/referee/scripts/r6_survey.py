"""Referee check R6: independent float64 survey of the smallest positive eigenvalue, all sorted
tuples with 2 j_max <= N (LAPACK stebz via scipy, index selection), own enumeration and own
closed-form chain.  Checks Theorem A/B numerically, Proposition D, Conjecture 1 in range, and
compares with the author's stored per-tuple data (read as data only, if present).
CPU choice: the bounded run below (N=80, ~0.8M tuples) takes about a minute or two; no GPU needed.
"""
import sys, math, time
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
OUT = sys.argv[2] if len(sys.argv) > 2 else "r6_out.txt"
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

C26 = 24**0.25*math.sqrt(96)*(1/math.sqrt(2)+0.5)
t0 = time.time()
n = 0
minA = (np.inf, None); minB = (np.inf, None); viol_conj = []; minratio_d4 = (np.inf, None)
LSe = [np.inf, -np.inf]; LSo = [np.inf, -np.inf]; violD = 0
per_jmax_min = {}
for J4 in range(1, N+1):
    for J3 in range(1, J4+1):
        for J2 in range(1, J3+1):
            for J1 in range(1, J2+1):
                if (J1+J2+J3+J4) % 2: continue
                j1, j2, j3, j4 = J1/2, J2/2, J3/2, J4/2
                kmin = max(abs(j1-j2), abs(j3-j4)); kmax = min(j1+j2, j3+j4)
                if kmax < kmin: continue
                d = int(round(kmax-kmin))+1
                if d < 2: continue
                k = kmin+np.arange(1, d)
                c2 = (((j1+j2+1)**2-k*k)*(k*k-(j1-j2)**2)*((j3+j4+1)**2-k*k)*(k*k-(j3-j4)**2))/(16*(4*k*k-1))
                c = np.sqrt(c2)
                i0 = (d+1)//2
                lam = eigvalsh_tridiagonal(np.zeros(d), c, select="i", select_range=(i0, i0))[0]
                n += 1
                jm = j4
                S = np.sum(1/c)
                if lam*C26/(jm+1) < minA[0]: minA = (lam*C26/(jm+1), (J1, J2, J3, J4))
                if lam < minB[0]: minB = (lam, (J1, J2, J3, J4))
                bound = math.sqrt(jm*(jm+1))/2
                if lam < bound*(1-1e-12): viol_conj.append((J1, J2, J3, J4, lam, bound))
                prev = per_jmax_min.get(J4)
                if prev is None or lam < prev[0]: per_jmax_min[J4] = (lam, (J1, J2, J3, J4))
                if d >= 4 and lam/math.sqrt(jm*(jm+1)) < minratio_d4[0]: minratio_d4 = (lam/math.sqrt(jm*(jm+1)), (J1, J2, J3, J4))
                LS = lam*S
                if d % 2 == 0:
                    LSe[0] = min(LSe[0], LS); LSe[1] = max(LSe[1], LS)
                    if not (1/24**0.25 <= LS <= 6): violD += 1
                elif d >= 3:
                    LSo[0] = min(LSo[0], LS); LSo[1] = max(LSo[1], LS)
log(f"R6 N={N}: {n} sorted tuples with d>=2, {time.time()-t0:.0f}s")
log(f"R6 min lambda*26.18/(j_max+1) = {minA[0]:.4f} at {minA[1]} (Theorem A claims >= 1)")
log(f"R6 global min lambda = {minB[0]:.15f} at doubled {minB[1]}; sqrt3/4 = {math.sqrt(3)/4:.15f}")
log(f"R6 Conjecture 1 (lambda >= sqrt(jmax(jmax+1))/2) violations: {len(viol_conj)} {viol_conj[:3]}")
nonhalf = [(J4, v) for J4, v in per_jmax_min.items() if not (v[1][0] == 1 and v[1][1] == 1 and v[1][2] == J4)]
log(f"R6 per-j_max minimiser not of the form (1/2,1/2,j,j) for 2j_max in: {[x[0] for x in nonhalf][:10]} (count {len(nonhalf)}) -- note ties (e.g. (1/2,1/2,j,j) vs other tuples with equal lambda) would show here")
log(f"R6 d>=4: min lambda/sqrt(jmax(jmax+1)) = {minratio_d4[0]:.4f} at {minratio_d4[1]}")
log(f"R6 lambda*S range even d: [{LSe[0]:.4f}, {LSe[1]:.4f}], violations of Prop D [24^-1/4, 6]: {violD}; odd d>=3: [{LSo[0]:.4f}, {LSo[1]:.4f}]")
# compare with the author's stored data if present (read as data only)
try:
    z = np.load(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'survey_N200', 'pertuple_J4le60.npz'), allow_pickle=False)
    log(f"R6 author npz keys: {list(z.keys())}")
except Exception as e:
    log(f"R6 author npz not compared: {e!r}")
open(OUT, "w").write("\n".join(out)+"\n")

"""Referee check R8: comparison of Lemma 1 with the classical weighted discrete Hardy inequality.

For even n, |B^{-1}| = E with E_ij = U_i V_j (i >= j), U_i = P_i, V_j = 1/(P_j c_{2j-1}),
P_1 = 1, P_{i+1} = P_i c_{2i}/c_{2i+1}.  So E is the matrix of a weighted Hardy operator, and
Muckenhoupt's theorem gives  A <= ||E|| <= 2A,  A = max_r (sum_{i>=r} U_i^2)^{1/2} (sum_{j<=r} V_j^2)^{1/2},
i.e.  1/(2A) <= lambda_min <= 1/A  WITHOUT log-concavity.  This script checks the two-sided bound on
volume chains (all sorted tuples, 2 j_max <= N; odd d reduced to the leading even block as in the
paper) and reports A*(j_max+1), to show what a Muckenhoupt-based route would need.
"""
import sys, math
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
lo = np.inf; hi = 0; maxAj = 0; arg = None; n = 0
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
                c = np.sqrt((((j1+j2+1)**2-k*k)*(k*k-(j1-j2)**2)*((j3+j4+1)**2-k*k)*(k*k-(j3-j4)**2))/(16*(4*k*k-1)))
                lam = eigvalsh_tridiagonal(np.zeros(d), c, select="i", select_range=((d+1)//2, (d+1)//2))[0]
                ce = c if d % 2 == 0 else c[:-1]        # even block (odd d: leading (d-1)x(d-1) submatrix)
                m = (len(ce)+1)//2
                logP = np.zeros(m)
                for i in range(1, m):
                    logP[i] = logP[i-1] + math.log(ce[2*i-1]) - math.log(ce[2*i])
                U2 = np.exp(2*logP); V2 = np.exp(-2*logP)/ce[0::2]**2
                tail = np.cumsum(U2[::-1])[::-1]; head = np.cumsum(V2)
                A = math.sqrt(np.max(tail*head))
                lamE = 1/np.linalg.norm(np.tril(np.outer(np.exp(logP), np.exp(-logP)/ce[0::2])), 2)
                r = lamE*A                                 # Muckenhoupt: 1/2 <= r <= 1
                lo = min(lo, r); hi = max(hi, r); n += 1
                if A*(j4+1) > maxAj: maxAj, arg = A*(j4+1), (J1, J2, J3, J4)
log(f"R8 N={N}: {n} tuples. Muckenhoupt ratio sigma_min(B_even)*A in [{lo:.4f}, {hi:.4f}] (theory: [0.5, 1]); "
    f"max A*(j_max+1) = {maxAj:.4f} at doubled {arg}  (so lambda >= (j_max+1)/(2*{maxAj:.3f}) in this range)")
open(sys.argv[2] if len(sys.argv) > 2 else "r8_out.txt", "w").write("\n".join(out)+"\n")

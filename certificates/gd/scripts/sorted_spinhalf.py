"""Sorted convention, orbit 3, distinguished edge at sorted position 1 with spin 1/2 (chain, Prop. 1 of the
closed theorem).  For each alpha-pair pattern P in {(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)} (sorted positions of the
|alpha|=4 pair), minimise the chain's q_min over sorted (s2<=s3<=s4<=s5), 2s5 = m, for m up to M (float64 Jacobi
eigenvalues; results below ~1e-13 relative are flagged).  Shows which patterns decay with a spin-1/2 edge."""
import sys, itertools, numpy as np
M = int(sys.argv[1]) if len(sys.argv) > 1 else 60
def chain_q(J2, J3, J4, J5, al, be):
    # X-pair (J2,J3) weight alpha, D-pair (J4,J5) weight beta; doubled spins
    mX, nX, mD, nD = abs(J2-J3), J2+J3+2, abs(J4-J5), J4+J5+2
    m, n = max(mX, mD), min(nX, nD)
    if n - m < 2: return None
    own_m = 'X' if mX > mD else 'D'
    c = []
    typ = 'D' if own_m == 'X' else 'X'
    for l in range(m+1, n):
        mm, nn, w = (mX, nX, al) if typ == 'X' else (mD, nD, be)
        c.append(w/4*np.sqrt((nn*nn-l*l)*(l*l-mm*mm)))
        typ = 'X' if typ == 'D' else 'D'
    N = len(c)+1
    T = np.zeros((N, N))
    for i, x in enumerate(c): T[i, i+1] = T[i+1, i] = x
    e = np.sort(np.abs(np.linalg.eigvalsh(T)))
    e = e[e > 1e-12*e[-1]]
    return e[0]
pats = [(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
print("m  " + "  ".join(f"{p}" for p in pats))
for m in range(4, M+1, 4):
    row = []
    for p in pats:
        best = None
        for s2 in range(1, m+1):
            for s3 in range(s2, m+1):
                for s4 in range(s3, m+1):
                    s = {2: s2, 3: s3, 4: s4, 5: m}
                    if (1+s2+s3+s4+m) % 2: continue
                    q_ = [x for x in (2,3,4,5) if x not in p]
                    q = chain_q(s[p[0]], s[p[1]], s[q_[0]], s[q_[1]], 4, 2)
                    if q is not None and (best is None or q < best[0]): best = (q, (1, s2, s3, s4, m))
        row.append(f"{best[0]:.2e}{best[1]}")
    print(m, "  ".join(row), flush=True)

"""Checks of Theorem 1 (spin-1/2 edge) on all valence-5 tuples with j1 = 1/2 and 2 j_max <= M (8 CPU processes):
 (a) each leg-1 triple qhat_{1JK} has kernel dim <= 2, and every nonzero eigenvalue of |qhat_{1JK}| is >= sqrt(3);
 (b) kernels of complementary leg-1 triples {1,J,K}, {1,L,M} are orthogonal when dim Inv >= 2;
 (c) Q_RS >= sum_{T contains 1} |qhat_T| >= 3 sqrt(3) on Inv when dim Inv >= 2 (min eigenvalues reported);
 (d) the contrast family (1/2, j-1/2, j, j, j) to 2j = M: lambda_min(Q_RS) next to the AL value P_N-law of [HV] Thm 1.
"""
import numpy as np, itertools, sys, time, math
from multiprocessing import Pool
from rs_core import *

def kerproj(A):
    w, U = np.linalg.eigh(1j * A); tol = 1e-9 * max(1, np.abs(w).max())
    Uk = U[:, np.abs(w) <= tol]; return (Uk @ Uk.conj().T).real, np.abs(w)[np.abs(w) > tol]

MATCH = [(((0, 1, 2), (0, 3, 4))), (((0, 1, 3), (0, 2, 4))), (((0, 1, 4), (0, 2, 3)))]

def work(Jd):
    V = Vertex(Jd)
    if V.dim == 0: return None
    Q = qhats(V)
    res = {"Jd": Jd, "dim": V.dim}
    kd = []; gmin = 1e9; P = {}
    for T in [t for t in Q if 0 in t]:
        K, nz = kerproj(Q[T]); P[T] = K
        kd.append(int(round(np.trace(K))))
        if len(nz): gmin = min(gmin, nz.min())
    res["maxker"] = max(kd); res["gmin"] = gmin
    res["orth"] = max(np.linalg.norm(P[a] @ P[b], 2) for a, b in MATCH)
    S1 = sum(absm(Q[T]) for T in Q if 0 in T)
    full = sum(absm(M) for M in Q.values())
    res["lam_leg1"] = float(np.linalg.eigvalsh(S1).min()); res["lam_full"] = float(np.linalg.eigvalsh(full).min())
    return res

if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    t0 = time.time()
    tups = [(1, b, c, d, e) for e in range(1, M + 1) for d in range(1, e + 1) for c in range(1, d + 1) for b in range(1, c + 1)
            if (1 + b + c + d + e) % 2 == 0]
    with Pool(8) as p:
        rows = [r for r in p.map(work, tups, chunksize=10) if r]
    big = [r for r in rows if r["dim"] >= 2]
    lines = [f"tuples with j1=1/2, 2jmax<={M}: {len(rows)} (dim>=2: {len(big)})",
             f"(a) max kernel dim of a leg-1 triple: {max(r['maxker'] for r in rows)}; min nonzero |qhat| eigenvalue: {min(r['gmin'] for r in rows if r['gmin']<1e9):.6f} (sqrt3 = {math.sqrt(3):.6f})",
             f"(b) max ||Pi_ker(1JK) Pi_ker(1LM)|| over complementary pairs, dim>=2: {max(r['orth'] for r in big):.2e}",
             f"(c) min eig of sum over leg-1 triples (dim>=2): {min(r['lam_leg1'] for r in big):.6f}; min eig of Q_RS (dim>=2): {min(r['lam_full'] for r in big):.6f}; 3 sqrt3 = {3*math.sqrt(3):.6f}",
             f"    dim-1 tuples: Q_RS = 0 in all: {all(abs(r['lam_full'])<1e-9 for r in rows if r['dim']==1)}"]
    lines.append("(d) contrast family (1/2, j-1/2, j, j, j): 2j, dim, lambda_min(Q_RS), AL q_min ~ kappa*4N sqrt(2N+1) 2^-N ([HV] Thm 1)")
    for N in [2, 4, 6, 10, 20, 30, 40, M]:
        r = work((1, N - 1, N, N, N))
        Pn = 4 * N * math.sqrt(2 * N + 1) * 2.0 ** (-N)
        lines.append(f"    2j={N:3d}: dim {r['dim']:3d}, lambda_min(Q_RS) = {r['lam_full']:.4f}, AL q_min in [{0.375*Pn:.3e}, {Pn:.3e}]")
    lines.append(f"wall {time.time()-t0:.0f}s")
    print("\n".join(lines)); open(f"../outputs/rs_verify_M{M}.txt", "w").write("\n".join(lines) + "\n")

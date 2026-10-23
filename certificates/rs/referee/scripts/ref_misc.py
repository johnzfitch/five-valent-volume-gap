"""Miscellaneous referee checks.

1. Literature cross-check of normalisation and code: Kisielowski 2104.11010 Table 2
   gives RS volume eigenvalues V = sqrt(q_K) with q_K = (1/8) sum_{I<J<K} |eps J J J|
   (J = -i x spin).  With q_hat = [(J_IJ)^2,(J_JK)^2] = -4i J_I.(J_J x J_K),
   q_K = Q_RS/32.  Compare 4-valent j=1/2 and 5-valent j=1, j=2 monochromatic spectra.
2. Q_RS on the AL no-gap family (1/2, j-1/2, j, j, j) for 2j = 2..40.
3. mpmath (50 digits) recomputation of lambda_min(Q_RS) and of
   lambda_min(sum_{T contains leg 1}|q_T|) on the extremal tuples, from exact entries.
4. Brute-force tensor-product spot checks at larger spins with a spin-1/2 leg.
"""
import itertools, math
import numpy as np
import mpmath as mp
import sympy as sp
from ref_core import q_triples, absq, dimInv
from ref_brute import brute


def QRS(js5):
    qs, _ = q_triples(js5)
    return sum(absq(q)[0] for q in qs.values()), qs


def part1():
    print("1. Kisielowski Table 2 cross-check (V = sqrt(Q_RS/32))")
    # 5-valent j = 1 and j = 2
    for tj, ref in [(2, [(1.201406, 6)]), (4, [(0.0, 1), (2.293525, 5), (2.754087, 5), (2.911094, 4), (3.095988, 1)])]:
        Q, _ = QRS((tj,) * 5)
        w = np.linalg.eigvalsh(Q)
        V = np.sqrt(np.clip(w, 0, None) / 32)
        vals = []
        for v in V:
            if vals and abs(v - vals[-1][0]) < 1e-6:
                vals[-1][1] += 1
            else:
                vals.append([v, 1])
        print(f"   5-valent j={tj/2}: ours {[(round(a,6),b) for a,b in vals]}  Kisielowski {ref}")
    print("   (4-valent j=1/2: sqrt(4*sqrt3/32) =", round(math.sqrt(4 * math.sqrt(3) / 32), 6), "; Kisielowski 0.465302)")


def part2():
    print("2. Q_RS on (1/2, j-1/2, j, j, j):")
    out = []
    for t in range(2, 41):
        js5 = (1, t - 1, t, t, t)
        if dimInv(js5) == 0:
            continue
        Q, _ = QRS(js5)
        out.append((t, float(np.linalg.eigvalsh(Q)[0])))
    print("   " + "; ".join(f"2j={t}: {v:.2f}" for t, v in out))


def mp_absq(qm):
    # qm: mpmath real antisymmetric matrix; |q| = sqrt(-q^2) via eigh of Hermitian i q
    H = mp.matrix(qm.rows, qm.cols)
    for i in range(qm.rows):
        for j in range(qm.cols):
            H[i, j] = mp.mpc(0, 1) * qm[i, j]
    E, U = mp.eighe(H) if hasattr(mp, 'eighe') else mp.eigh(H)
    D = mp.diag([abs(e) for e in E])
    return U * D * U.transpose_conj()


def part3(tuples):
    mp.mp.dps = 50
    print("3. 50-digit recomputation from exact entries:")
    for js5 in tuples:
        qs, _ = q_triples(js5, exact=True)
        mats = {}
        for T, q in qs.items():
            mats[T] = mp.matrix([[mp.mpf(sp.N(q[i, j], 60)) for j in range(q.cols)] for i in range(q.rows)])
        A = {T: mp_absq(m) for T, m in mats.items()}
        Q = sum((A[T] for T in A), mp.zeros(len(list(A.values())[0]), len(list(A.values())[0])))
        S1 = sum((A[T] for T in A if 0 in T), mp.zeros(Q.rows, Q.cols))
        eQ = min(mp.re(e) for e in mp.eigh(Q)[0]); eS = min(mp.re(e) for e in mp.eigh(S1)[0])
        print(f"   {js5}: lambda_min(Q_RS) = {mp.nstr(eQ, 20)}; lambda_min(sum_(T contains 1)|q_T|) = {mp.nstr(eS, 20)}")


def part4(tuples):
    print("4. brute-force tensor-product spot checks (spectrum of Q_RS and of sum c_T q_T):")
    rng = np.random.default_rng(7)
    for js5 in tuples:
        qb, db = brute(js5); qr, _ = q_triples(js5)
        c = {T: rng.normal() for T in qr}
        def inv(qs):
            Q = sum(absq(q)[0] for q in qs.values())
            C = sum(c[T] * qs[T] for T in qs)
            return np.sort(np.linalg.eigvalsh(Q)), np.sort(np.linalg.eigvalsh(1j * C))
        a = inv(qb); b = inv(qr)
        e = max(np.max(np.abs(a[0] - b[0])), np.max(np.abs(a[1] - b[1])))
        print(f"   {js5}: dim {db}, lambda_min(Q_RS) {a[0][0]:.6f}, max discrepancy {e:.2e}")


if __name__ == "__main__":
    part1(); part2()
    part3([(1, 1, 1, 1, 2), (1, 2, 2, 3, 4), (1, 1, 2, 2, 2), (1, 2, 2, 2, 3)])
    part4([(1, 5, 5, 5, 4), (1, 2, 2, 5, 4), (1, 1, 3, 5, 4), (5, 1, 4, 3, 5)])

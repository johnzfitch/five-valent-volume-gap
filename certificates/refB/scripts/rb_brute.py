"""Brute-force tensor-product check of rb_core (no recoupling theory used here).

Build spin matrices for five legs, restrict to total M = 0, find Inv = ker(J^2) there, compute
T = J1.(J2xJ3 + J4xJ5) on Inv, and compare its spectrum with rb_core.T_matrix.
Also checks antisymmetry of R computed in both directions (exact), and Hermiticity.
Usage: python rb_brute.py MAX2J [MAX2J1] > ../outputs/rb_brute_out.txt
"""
import sys, itertools, time
import numpy as np
import scipy.sparse as sps
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import rb_core as C


def spin_ops(j2):
    d = j2 + 1
    m = np.array([j2 / 2 - k for k in range(d)])
    Jz = np.diag(m)
    Jp = np.zeros((d, d))
    for k in range(1, d):
        mm = m[k]
        Jp[k - 1, k] = np.sqrt(j2 / 2 * (j2 / 2 + 1) - mm * (mm + 1))
    Jx = (Jp + Jp.T) / 2; Jy = (Jp - Jp.T) / 2j
    return [Jx, Jy.astype(complex), Jz], m


def brute_spectrum(tj):
    ops, ms = zip(*[spin_ops(x) for x in tj])
    dims = [x + 1 for x in tj]
    # basis states with total M = 0
    states = [st for st in itertools.product(*[range(d) for d in dims]) if abs(sum(ms[k][st[k]] for k in range(5))) < 1e-9]
    if not states:
        return np.array([]), 0
    idx = {st: i for i, st in enumerate(states)}
    N = len(states)

    def leg_op(k, a):
        """matrix of component a of leg k on the M=0 subspace is not closed (Jx changes M); build full sparse instead"""
        mats = [sps.identity(d, format='csr', dtype=complex) for d in dims]
        mats[k] = sps.csr_matrix(ops[k][a])
        out = mats[0]
        for mm in mats[1:]:
            out = sps.kron(out, mm, format='csr')
        return out

    Dfull = int(np.prod(dims))
    L = [[leg_op(k, a) for a in range(3)] for k in range(5)]
    Jt = [sum(L[k][a] for k in range(5)) for a in range(3)]
    J2 = Jt[0] @ Jt[0] + Jt[1] @ Jt[1] + Jt[2] @ Jt[2]
    # indices of M=0 states in the full product basis
    strides = np.cumprod([1] + dims[::-1])[::-1][1:]
    full_idx = np.array([sum(st[k] * strides[k] for k in range(5)) for st in states])
    J2r = J2[full_idx][:, full_idx].toarray()
    w, v = np.linalg.eigh(J2r)
    inv = v[:, np.abs(w) < 1e-8]
    if inv.shape[1] == 0:
        return np.array([]), 0

    def cross_dot(a, b, c):
        # J_a . (J_b x J_c)
        out = sps.csr_matrix((Dfull, Dfull), dtype=complex)
        for (x, y, z, sg) in ((0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (0, 2, 1, -1), (1, 0, 2, -1), (2, 1, 0, -1)):
            out = out + sg * (L[a][x] @ (L[b][y] @ L[c][z]))
        return out

    T = cross_dot(0, 1, 2) + cross_dot(0, 3, 4)
    Tr = T[full_idx][:, full_idx].toarray()
    Ti = inv.conj().T @ Tr @ inv
    ev = np.linalg.eigvalsh((Ti + Ti.conj().T) / 2)
    herm = np.abs(Ti - Ti.conj().T).max()
    return np.sort(ev), herm


if __name__ == '__main__':
    M = int(sys.argv[1]); M1 = int(sys.argv[2]) if len(sys.argv) > 2 else M
    t0 = time.time()
    worst = 0.0; cnt = 0; dimbad = 0; asym_bad = 0; herm_w = 0.0
    for tj in itertools.product(range(1, M1 + 1), *[range(1, M + 1)] * 4):
        if sum(tj) % 2:
            continue
        S, T = C.T_matrix(tj)
        eb, herm = brute_spectrum(tj)
        if len(S) != len(eb):
            dimbad += 1
            print('DIM MISMATCH', tj, len(S), len(eb)); continue
        if len(S) == 0:
            continue
        herm_w = max(herm_w, herm)
        # antisymmetry of R exactly
        _, e1 = C.R_entries(tj); _, e2 = C.R_entries_reverse(tj)
        for (i, j), (sg, sq) in e1.items():
            sg2, sq2 = e2.get((j, i), (0, 0))
            if not (sq2 == sq and sg2 == -sg):
                asym_bad += 1
        if len(e2) != len(e1):
            asym_bad += 1
        er = np.sort(np.linalg.eigvalsh(T))
        sc = max(1.0, np.abs(eb).max())
        worst = max(worst, np.abs(er - eb).max() / sc)
        cnt += 1
    print(f"brute force vs rb_core: {cnt} tuples (2j1<={M1}, 2j_i<={M}), dim mismatches {dimbad}, "
          f"antisymmetry failures (exact) {asym_bad}, max |spec diff|/norm {worst:.2e}, brute Hermiticity defect {herm_w:.1e} "
          f"({time.time()-t0:.0f}s)")

"""Check: Y = J1.(J2-J3), Z = J1.(J4-J5) on Inv; Y+Z = (J3+J5)^2-(J2+J4)^2;
T = (1/2i)([J2.J3, Y] + [J4.J5, Z]).  Brute-force tensor construction in the invariant space."""
import sys, numpy as np, scipy.sparse as sp
from t1core import spin_mats


def ops_for(spins):
    dims = [t + 1 for t in spins]
    ops = []
    for i, t in enumerate(spins):
        full = []
        for M in spin_mats(t):
            op = sp.csr_matrix(np.array([[1.0 + 0j]]))
            for k in range(len(spins)):
                op = sp.kron(op, sp.csr_matrix(M.astype(complex)) if k == i else sp.identity(dims[k], format='csr'), format='csr')
            full.append(op)
        ops.append(full)
    return ops


def inv_basis(ops):
    Jt = [sum(o[a] for o in ops) for a in range(3)]
    jz = np.real(Jt[2].diagonal()); Z = np.where(np.abs(jz) < 1e-9)[0]
    Jp = (Jt[0] + 1j * Jt[1])[:, Z].toarray()
    u, sv, vh = np.linalg.svd(Jp, full_matrices=True)
    r = int(np.sum(sv > 1e-8)); k = vh[r:].conj().T
    inv = np.zeros((Jt[0].shape[0], k.shape[1]), dtype=complex); inv[Z] = k
    return inv


def dot(A, B, X):   # (A.B) X
    return sum(A[a] @ (B[a] @ X) for a in range(3))


def cross_dot(A, B, C, X):  # A.(B x C) X
    out = 0
    for (i, k, l, sg) in [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]:
        out = out + sg * (A[i] @ (B[k] @ (C[l] @ X)))
    return out


if __name__ == '__main__':
    spins = list(map(int, sys.argv[1:6]))
    ops = ops_for(spins)
    P = inv_basis(ops)
    J1, J2, J3, J4, J5 = ops
    red = lambda Y: P.conj().T @ Y
    T = red(cross_dot(J1, J2, J3, P) + cross_dot(J1, J4, J5, P))
    Y = red(dot(J1, J2, P) - dot(J1, J3, P))
    Zm = red(dot(J1, J4, P) - dot(J1, J5, P))
    b23 = red(dot(J2, J3, P)); g45 = red(dot(J4, J5, P))
    P24 = [J2[a] + J4[a] for a in range(3)]; Q35 = [J3[a] + J5[a] for a in range(3)]
    Om = red(dot(Q35, Q35, P) - dot(P24, P24, P))
    print('dim Inv', P.shape[1])
    print('|Y+Z-Omega| =', np.abs(Y + Zm - Om).max())
    print('|T - (1/2i)([b,Y]+[g,Z])| =', np.abs(T - (b23 @ Y - Y @ b23 + g45 @ Zm - Zm @ g45) / 2j).max())
    ev = np.sort(np.linalg.eigvalsh((Om + Om.conj().T) / 2))
    print('spec Omega:', np.round(ev, 6))
    eT = np.sort(np.abs(np.linalg.eigvalsh((T + T.conj().T) / 2)))
    print('spec |T| :', np.round(eT, 6), ' q_min =', 8 * eT[eT > 1e-9].min() if (eT > 1e-9).any() else None)

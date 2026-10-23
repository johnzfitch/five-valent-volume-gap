"""Check of Theorem 7.1 (three-spin pencil) and of the equivalent short form of its operator A.

For spin tuples (s; j2, j3, j4, j5) (doubled spins on the command line or the built-in list), this script
  1. builds T = J1.(J2xJ3 + J4xJ5) on Inv(j1..j5) by brute force on the tensor product (no recoupling theory);
  2. builds, on the weight space V[M0] of j2 x j3 x j4 with M0 = s - j5, the operators
       G = J2+J3+J4, U = J2xJ3 + J2xJ4 + J3xJ4,
       B  = sum_k c_k G_-^k G_+^k,    c_k  = (-1)^k / (k! (2s+2)_k),
       B' = sum_k c'_k G_-^k G_+^k,   c'_k = (-1)^k / (k! (2s+3)_k),
       X  = s U_z + (U_+ G_- + U_- G_+)/2,
       A  = X^* B - j5/(2s+2) U_- B' G_+          (the form printed in Theorem 7.1)
       A' = (s U_z + U_+ G_- / 2) B                (the short form derived in the appendix)
  3. checks: A = A' (as matrices on V[M0]); A Hermitian; B >= 0; and that the spectrum of -B^{-1/2} A B^{-1/2}
     on range(B) equals the spectrum of T, with multiplicities.
Float64; prints the largest deviations. Usage: python pencil_check.py  (writes pencil_check_out.txt beside it)."""
import os, sys, itertools
import numpy as np
from math import factorial
HERE = os.path.dirname(os.path.abspath(__file__))


def spin(tj):
    j = tj / 2; m = j - np.arange(tj + 1)
    jp = np.zeros((tj + 1, tj + 1))
    for k in range(1, tj + 1):
        jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    jx = (jp + jp.T) / 2; jy = (jp - jp.T) / 2j; jz = np.diag(m)
    return [jx.astype(complex), jy, jz.astype(complex)], m


def legs_ops(tjs):
    mats = [spin(t)[0] for t in tjs]; dims = [t + 1 for t in tjs]
    ops = []
    for i in range(len(tjs)):
        L = []
        for a in range(3):
            M = np.array([[1.0 + 0j]])
            for k, d in enumerate(dims):
                M = np.kron(M, mats[i][a] if k == i else np.eye(d))
            L.append(M)
        ops.append(L)
    return ops


def cross(A, B):
    return [A[1] @ B[2] - A[2] @ B[1], A[2] @ B[0] - A[0] @ B[2], A[0] @ B[1] - A[1] @ B[0]]


def dot(A, B):
    return A[0] @ B[0] + A[1] @ B[1] + A[2] @ B[2]


def T_spectrum(tjs):
    ops = legs_ops(tjs)
    tot = [sum(o[a] for o in ops) for a in range(3)]
    C = dot(tot, tot)
    w, v = np.linalg.eigh(C)
    P = v[:, np.abs(w) < 1e-8]
    if P.shape[1] == 0: return np.array([])
    T = dot(ops[0], [x + y for x, y in zip(cross(ops[1], ops[2]), cross(ops[3], ops[4]))])
    Ti = P.conj().T @ T @ P
    return np.sort(np.linalg.eigvalsh((Ti + Ti.conj().T) / 2))


def pencil(tjs):
    s2, t2, t3, t4, t5 = tjs
    s, j5 = s2 / 2, t5 / 2
    M0 = s - j5
    ops = legs_ops((t2, t3, t4))
    G = [sum(o[a] for o in ops) for a in range(3)]
    U = [x + y + z for x, y, z in zip(cross(ops[0], ops[1]), cross(ops[0], ops[2]), cross(ops[1], ops[2]))]
    Gp, Gm = G[0] + 1j * G[1], G[0] - 1j * G[1]
    Up, Um, Uz = U[0] + 1j * U[1], U[0] - 1j * U[1], U[2]
    gz = np.real(np.diag(G[2]))
    idx = np.where(np.abs(gz - M0) < 1e-9)[0]
    E = np.eye(len(gz))[:, idx]                       # inclusion of V[M0]
    def poch(a, k):
        r = 1.0
        for i in range(k): r *= a + i
        return r
    K = int(round(2 * (s + j5))) + 2
    Bf = np.zeros_like(Gp); Bpf = np.zeros_like(Gp)
    GmkGpk = np.eye(len(gz), dtype=complex)
    for k in range(K + 1):
        if k > 0: GmkGpk = Gm @ GmkGpk @ Gp if False else np.linalg.matrix_power(Gm, k) @ np.linalg.matrix_power(Gp, k)
        Bf = Bf + (-1) ** k / (factorial(k) * poch(2 * s + 2, k)) * GmkGpk
        Bpf = Bpf + (-1) ** k / (factorial(k) * poch(2 * s + 3, k)) * GmkGpk
    X = s * Uz + (Up @ Gm + Um @ Gp) / 2
    Afull = X.conj().T @ Bf - j5 / (2 * s + 2) * Um @ Bpf @ Gp
    Ashort = (s * Uz + Up @ Gm / 2) @ Bf
    B = E.T @ Bf @ E; A = E.T @ Afull @ E; A2 = E.T @ Ashort @ E
    return A, A2, B


def check(tjs):
    A, A2, B = pencil(tjs)
    sc = max(1.0, np.abs(A).max())
    dA = np.abs(A - A2).max() / sc
    herm = np.abs(A - A.conj().T).max() / sc
    wb, vb = np.linalg.eigh((B + B.conj().T) / 2)
    minB = wb.min() if wb.size else 0.0
    keep = wb > 1e-10
    R = vb[:, keep] / np.sqrt(wb[keep])
    Mp = -(R.conj().T @ A @ R)
    ev_p = np.sort(np.linalg.eigvalsh((Mp + Mp.conj().T) / 2)) if Mp.size else np.array([])
    ev_T = T_spectrum(tjs)
    if len(ev_p) != len(ev_T):
        return dA, herm, minB, None, len(ev_p), len(ev_T)
    dev = np.abs(ev_p - ev_T).max() / max(1.0, np.abs(ev_T).max()) if len(ev_T) else 0.0
    return dA, herm, minB, dev, len(ev_p), len(ev_T)


if __name__ == '__main__':
    tuples = [(3, 1, 2, 2, 2), (5, 2, 2, 2, 3), (4, 2, 2, 3, 3), (6, 1, 2, 3, 4), (5, 1, 2, 2, 2), (7, 2, 2, 2, 5),
              (2, 2, 2, 2, 2), (4, 1, 1, 2, 2), (5, 3, 2, 1, 5), (6, 2, 3, 3, 2), (3, 3, 3, 2, 2), (8, 2, 2, 2, 6),
              (5, 2, 3, 2, 4), (4, 3, 3, 2, 4), (6, 1, 1, 2, 6), (7, 3, 2, 2, 4)]
    lines = []; worst = [0, 0, 0]
    for t in tuples:
        if sum(t) % 2: continue
        dA, herm, minB, dev, n1, n2 = check(t)
        line = (f"2j = {t}: |A - A'| = {dA:.1e}, |A - A^*| = {herm:.1e}, min eig B = {minB:.3e}, "
                f"pencil vs T: dims {n1}/{n2}, max dev {dev if dev is None else f'{dev:.1e}'}")
        print(line); lines.append(line)
        worst = [max(worst[0], dA), max(worst[1], herm), max(worst[2], dev if dev is not None else 1e9)]
    lines.append(f"worst: |A - A'| {worst[0]:.1e}, non-Hermiticity {worst[1]:.1e}, spectrum deviation {worst[2]:.1e}")
    print(lines[-1])
    open(os.path.join(HERE, 'pencil_check_out.txt'), 'w').write("\n".join(lines) + "\n")

"""Exact three-spin pencil for T (claimed identity, checked here numerically against lj_core).
Legs 2,3,4 form V = j2 x j3 x j4 with G = J2+J3+J4 and U = J2xJ3 + J2xJ4 + J3xJ4; leg 5 = spin j5;
M = s - j5.  With c_k = (-1)^k / (k! (2s+2)_k):
   B  = sum_k c_k G-^k G+^k         on V[M]
   B' = sum_k c'_k G-^k G+^k        on V[M+1],  c'_k = (-1)^k / (k! (2s+3)_k)
   A  = X^* B - (j5/(2s+2)) U- B' G+,   X = s U_z + (U+ G- + U- G+)/2
Claim: the nonzero spectrum of T equals the nonzero generalized eigenvalues of (A, B) on range(B), up to sign."""
import sys, os, itertools, numpy as np
from math import factorial
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import lj_core as L


def spin(j2):
    j = j2 / 2; m = np.arange(j, -j - 1, -1)
    jp = np.diag(np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1)), 1)
    return jp, jp.T, np.diag(m)


def three(t):
    ops = [spin(x) for x in t]; dims = [x + 1 for x in t]; I = [np.eye(d) for d in dims]
    def emb(k, A):
        mats = list(I); mats[k] = A
        return np.kron(np.kron(mats[0], mats[1]), mats[2])
    Jp = [emb(k, o[0]) for k, o in enumerate(ops)]; Jm = [emb(k, o[1]) for k, o in enumerate(ops)]
    Jz = [emb(k, o[2]) for k, o in enumerate(ops)]
    Jx = [(Jp[k] + Jm[k]) / 2 for k in range(3)]; Jy = [(Jp[k] - Jm[k]) / 2j for k in range(3)]
    Ux = sum(Jy[a] @ Jz[b] - Jz[a] @ Jy[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    Uy = sum(Jz[a] @ Jx[b] - Jx[a] @ Jz[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    Uz = sum(Jx[a] @ Jy[b] - Jy[a] @ Jx[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    Gp, Gm, Gz = sum(Jp), sum(Jm), sum(Jz)
    return dict(Up=Ux + 1j * Uy, Um=Ux - 1j * Uy, Uz=Uz, Gp=Gp, Gm=Gm, Mz=np.real(np.diag(Gz)))


def poch(a, k):
    r = 1.0
    for i in range(k): r *= a + i
    return r


def pencil(t, s, j5):
    o = three(t); n = o['Gp'].shape[0]
    Bfull = np.zeros((n, n)); Bpf = np.zeros((n, n))
    P = np.eye(n)
    for k in range(0, sum(t) + 2):
        Bfull = Bfull + ((-1) ** k / (factorial(k) * poch(2 * s + 2, k))) * P
        Bpf = Bpf + ((-1) ** k / (factorial(k) * poch(2 * s + 3, k))) * P
        P = o['Gm'] @ P @ o['Gp']
    X = s * o['Uz'] + 0.5 * (o['Up'] @ o['Gm'] + o['Um'] @ o['Gp'])
    A = X.conj().T @ Bfull - (j5 / (2 * s + 2)) * o['Um'] @ Bpf @ o['Gp']
    M = s - j5
    idx = np.where(np.abs(o['Mz'] - M) < 1e-9)[0]
    Ar = A[np.ix_(idx, idx)]; Br = np.real(Bfull[np.ix_(idx, idx)])
    return Ar, Br, o['Uz'][np.ix_(idx, idx)]


def pencil_spec(t, s, j5):
    Ar, Br, Uz = pencil(t, s, j5)
    w, V = np.linalg.eigh(Br)
    keep = w > 1e-12 * max(1, w.max())
    Q = V[:, keep] / np.sqrt(w[keep])
    Tm = Q.conj().T @ Ar @ Q
    herm = np.abs(Tm - Tm.conj().T).max()
    ev = np.linalg.eigvalsh((Tm + Tm.conj().T) / 2)
    return ev, herm, int(keep.sum())


if __name__ == '__main__':
    worst = 0; worst_h = 0; n = 0
    for tj in itertools.product(range(1, 6), repeat=5):
        if sum(tj) % 2: continue
        o = L.build(tj)
        if o['nS'] < 1: continue
        W = np.zeros((o['nS'], o['nS'])); W[o['lo'], o['up']] = o['w']; W = W + W.T
        evT = np.sort(np.abs(np.linalg.eigvalsh(W)))
        ev, herm, dim = pencil_spec(tj[1:4], tj[0] / 2, tj[4] / 2)
        evP = np.sort(np.abs(ev))
        assert dim == o['nS'], (tj, dim, o['nS'])
        worst = max(worst, np.abs(evT - evP).max() / max(1, evT.max())); worst_h = max(worst_h, herm)
        n += 1
    print(f"tuples {n} (2j<=5): dims agree; max |spec T - spec pencil| (rel) = {worst:.2e}; max non-Hermiticity of B^-1/2 A B^-1/2 = {worst_h:.2e}")

"""Numerical check of [LJ] Theorem A (three-spin pencil) with an independent implementation of its statement.

V = j2 x j3 x j4 (explicit spin matrices), G = J2+J3+J4, U = J2xJ3 + J2xJ4 + J3xJ4, s = j1, M = s - j5.
  X = s U_z + (U+ G- + U- G+)/2 ;  B = sum_k c_k G-^k G+^k (on V[M]),  B' = sum_k c'_k G-^k G+^k (on V[M+1]),
  c_k = (-1)^k/(k!(2s+2)_k),  c'_k = (-1)^k/(k!(2s+3)_k),  A = X^* B - (j5/(2s+2)) U- B' G+.
Claims checked: A Hermitian; B >= 0 with eigenvalues beta_g = (s+j5+1-g)_{g-M}/(2s+2)_{g-M} on the G^2 = g(g+1) part;
spec(T) = spec(-B^{-1/2} A B^{-1/2}) on range(B), with multiplicity (T from the Racah operator rb_core).
Usage: python rb_pencil.py MAX2J > ../outputs/rb_pencil_out.txt
"""
import sys, os, itertools, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C


def spin(j2):
    d = j2 + 1; m = np.array([j2 / 2 - k for k in range(d)])
    Jp = np.zeros((d, d))
    for k in range(1, d):
        Jp[k - 1, k] = math.sqrt(j2 / 2 * (j2 / 2 + 1) - m[k] * (m[k] + 1))
    return Jp, Jp.T.copy(), np.diag(m)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def poch(a, k):
    r = 1.0
    for i in range(k):
        r *= a + i
    return r


def pencil_spectrum(tj):
    s2, j2, j3, j4, j5 = tj
    s = s2 / 2; J5 = j5 / 2; M = s - J5
    ops = [spin(x) for x in (j2, j3, j4)]
    I = [np.eye(x + 1) for x in (j2, j3, j4)]
    def leg(k, which):
        f = list(I); f[k] = ops[k][which]
        return kron3(*f)
    Jp = [leg(k, 0) for k in range(3)]; Jm = [leg(k, 1) for k in range(3)]; Jz = [leg(k, 2) for k in range(3)]
    Jx = [(Jp[k] + Jm[k]) / 2 for k in range(3)]; Jy = [(Jp[k] - Jm[k]) / 2j for k in range(3)]
    def cross(a, b):
        return [Jy[a] @ Jz[b] - Jz[a] @ Jy[b], Jz[a] @ Jx[b] - Jx[a] @ Jz[b], Jx[a] @ Jy[b] - Jy[a] @ Jx[b]]
    U = [sum(x) for x in zip(cross(0, 1), cross(0, 2), cross(1, 2))]
    Up = U[0] + 1j * U[1]; Um = U[0] - 1j * U[1]; Uz = U[2]
    Gp = sum(Jp); Gm = sum(Jm); Gz = sum(Jz)
    gz = np.real(np.diag(Gz))
    iM = np.where(np.abs(gz - M) < 1e-9)[0]; iM1 = np.where(np.abs(gz - (M + 1)) < 1e-9)[0]
    if len(iM) == 0:
        return None
    X = s * Uz + (Up @ Gm + Um @ Gp) / 2
    def Bop(cfirst):
        out = np.zeros_like(Gp, dtype=complex); Pk = np.eye(Gp.shape[0], dtype=complex)   # G-^k G+^k
        k = 0
        while True:
            c = (-1) ** k / (math.factorial(k) * poch(2 * s + cfirst, k))
            out = out + c * Pk
            k += 1
            Pk = Gm @ Pk @ Gp
            if np.abs(Pk).max() < 1e-300 or k > 60:
                break
        return out
    Bf = Bop(2); Bpf = Bop(3)
    B = Bf[np.ix_(iM, iM)]
    A = (X.conj().T @ Bf)[np.ix_(iM, iM)] - (J5 / (2 * s + 2)) * (Um[np.ix_(iM, iM1)] @ Bpf[np.ix_(iM1, iM1)] @ Gp[np.ix_(iM1, iM)])
    herm = np.abs(A - A.conj().T).max() / max(1, np.abs(A).max())
    w, V = np.linalg.eigh((B + B.conj().T) / 2)
    keep = w > 1e-10
    Bh = V[:, keep] @ np.diag(w[keep] ** -0.5)
    Tm = -(Bh.conj().T @ A @ Bh)
    ev = np.sort(np.linalg.eigvalsh((Tm + Tm.conj().T) / 2))
    # beta_g check: B on the G^2 = g(g+1) part
    G2 = (Gp @ Gm + Gm @ Gp) / 2 + Gz @ Gz
    G2M = G2[np.ix_(iM, iM)]
    gw, gv = np.linalg.eigh((G2M + G2M.conj().T) / 2)
    bad_beta = 0.0
    for idx in range(len(gw)):
        g = (-1 + math.sqrt(1 + 4 * gw[idx])) / 2
        v = gv[:, idx]
        bval = np.real(v.conj() @ B @ v)
        n_ = int(round(g - M))
        beta = poch(s + J5 + 1 - g, n_) / poch(2 * s + 2, n_)
        bad_beta = max(bad_beta, abs(bval - beta))
    return ev, herm, w.min(), bad_beta


if __name__ == '__main__':
    MAX = int(sys.argv[1])
    t0 = time.time()
    worst = 0.0; cnt = 0; hw = 0.0; bmin = 1.0; bb = 0.0; dimbad = 0
    for tj in itertools.product(range(1, MAX + 1), repeat=5):
        if sum(tj) % 2 or C.dim_inv(tj) == 0:
            continue
        r = pencil_spectrum(tj)
        if r is None:
            continue
        ev, herm, bm, bbeta = r
        S, T = C.T_matrix(tj)
        et = np.sort(np.linalg.eigvalsh(T))
        if len(et) != len(ev):
            dimbad += 1; continue
        worst = max(worst, np.abs(et - ev).max() / max(1, np.abs(et).max()))
        hw = max(hw, herm); bmin = min(bmin, bm); bb = max(bb, bbeta); cnt += 1
    print(f"[LJ] Theorem A pencil vs Racah T: {cnt} tuples (all 2j <= {MAX}); dimension mismatches {dimbad}; "
          f"max rel spectral diff {worst:.2e}; max |A - A*| {hw:.1e}; min eig B {bmin:.2e} (>= -1e-12 expected); "
          f"max |B - beta_g| on g-parts {bb:.1e}  ({time.time()-t0:.0f}s)")

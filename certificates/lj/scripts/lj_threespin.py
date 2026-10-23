"""Three-spin chirality operator U_z = (J2xJ3 + J2xJ4 + J3xJ4)_z on weight spaces (j2 x j3 x j4)_{M}.
nu(j2,j3,j4;M) = smallest nonzero |eigenvalue|.  Brute force in the product basis (float64)."""
import numpy as np, itertools, sys


def spin(j2):
    j = j2 / 2; m = np.arange(j, -j - 1, -1)
    jp = np.diag(np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1)), 1)
    return jp, jp.T, np.diag(m)


def Uz_blocks(t):
    ops = [spin(x) for x in t]
    dims = [x + 1 for x in t]
    I = [np.eye(d) for d in dims]

    def emb(k, A):
        mats = [I[0], I[1], I[2]]; mats[k] = A
        return np.kron(np.kron(mats[0], mats[1]), mats[2])
    Jp = [emb(k, ops[k][0]) for k in range(3)]; Jm = [emb(k, ops[k][1]) for k in range(3)]
    Jz = [emb(k, ops[k][2]) for k in range(3)]
    U = np.zeros_like(Jp[0], dtype=complex)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        U += 0.5j * (Jp[a] @ Jm[b] - Jm[a] @ Jp[b])
    Mz = np.real(np.diag(sum(Jz)))
    out = {}
    for M in np.unique(np.round(Mz * 2) / 2):
        idx = np.where(np.abs(Mz - M) < 1e-9)[0]
        out[M] = U[np.ix_(idx, idx)]
    return out


def nu(t, M=None):
    res = {}
    for MM, B in Uz_blocks(t).items():
        if M is not None and abs(MM - M) > 1e-9: continue
        ev = np.linalg.eigvalsh(B)
        nz = np.abs(ev)[np.abs(ev) > 1e-9]
        res[MM] = (nz.min() if nz.size else None, B.shape[0], int(ev.size - nz.size))
    return res


if __name__ == '__main__':
    K = int(sys.argv[1])
    best = (9, None); rows = []
    for t in itertools.combinations_with_replacement(range(1, K + 1), 3):
        for M, (v, dim, ker) in nu(t).items():
            if v is None: continue
            rows.append((v, t, M, dim, ker))
    rows.sort()
    print("smallest nonzero |eig U_z| over weight spaces, triples 2j<=%d (doubled), M>=0 shown" % K)
    seen = 0
    for v, t, M, dim, ker in rows:
        if M < 0: continue
        print(f"nu={v:.10f}  8nu={8*v:.6f}  t={t} M={M} dim={dim} ker={ker}")
        seen += 1
        if seen >= 25: break
    # per max spin minima
    for jm in range(1, K + 1):
        sub = [r for r in rows if max(r[1]) == jm]
        if sub:
            v, t, M, dim, ker = min(sub)
            print(f"max 2j={jm:2d}: min nu={v:.6f} at t={t} M={M}")

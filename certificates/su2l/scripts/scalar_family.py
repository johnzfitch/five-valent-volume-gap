"""Scalar ladder A(kappa): Jacobi matrix on d in {-s..s}, zero diagonal, links l_d^2 = kappa - d(d+1)/4,
d = -s..s-1.  kappa = s(s+1)/4 + gamma, gamma >= 0.  Smallest |eig| (half-integer s) and smallest
nonzero |eig| (integer s).  Float64 scan."""
import numpy as np, sys

def A(S2, kappa):
    s = S2 / 2.0
    ds = -s + np.arange(S2)          # d = -s .. s-1
    l2 = kappa - ds * (ds + 1) / 4.0
    l = np.sqrt(l2)
    return np.diag(l, 1) + np.diag(l, -1)

def smallest(M, integer):
    ev = np.sort(np.abs(np.linalg.eigvalsh(M)))
    if integer:
        return ev[1]   # one kernel vector (odd dimension)
    return ev[0]

if __name__ == '__main__':
    gam = np.concatenate([np.linspace(0, 2, 401), np.geomspace(2, 1e6, 300)])
    out = []
    for S2 in range(1, 82):
        s = S2 / 2
        k0 = s * (s + 1) / 4
        vals = [smallest(A(S2, k0 + g), S2 % 2 == 0) for g in gam]
        i = int(np.argmin(vals))
        mono = all(vals[k + 1] >= vals[k] - 1e-12 for k in range(len(vals) - 1))
        out.append((S2, vals[0], min(vals), gam[i], mono))
        print(f"2s={S2:3d} at gamma=0: {vals[0]:.6f}  min over gamma: {min(vals):.6f} at gamma={gam[i]:.4g}  nondecreasing in gamma: {mono}")

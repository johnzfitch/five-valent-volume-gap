"""Window a=0 of Conjecture SW via the inertia (secular) criteria.
odd N : sigma_min(B)^2 >= s  <=>  s*A(s)*C(s) <= E(s)^2, with A = sum w0/((lam-1)^2-s), C = sum wN/((lam-1)^2-s),
        E = sum (lam-1) v0 vN/((lam-1)^2-s)   (w0,wN: squared end components of the eigenvectors).
even N: sigma_min(B) >= 1  <=>  (X(0)+X(2))^2 <= G0(2) GN(2), X(z) = sum v0 vN/(lam-z), G(z) = sum w/(lam-z);
        and X(2) = -X(0)(N-1)/(N+1), |X(0)| = prod g / ((N-1)!!)^2 exactly.
Checks the criteria against direct SVD, the closed forms, and the bulk limit ratio sAC/E^2 -> sin^2(pi sqrt(s)/2)."""
import numpy as np, math
from stark import stark, Bwin, smin
s = (15 - math.sqrt(97)) / 6
def eig(N, m):
    H, g = stark(N, m); lam, U = np.linalg.eigh(H); return lam, U[0], U[-1], g
out = []
mism = 0; xerr = 0
for N in range(2, 61):
    for m in (0, .5, 1, 2.5, 7, 30):
        lam, v0, vN, g = eig(N, m)
        sv = smin(Bwin(N, m, 0))
        if N % 2:
            r = 1 / ((lam - 1) ** 2 - s)
            A = np.sum(v0**2 * r); C = np.sum(vN**2 * r); E = np.sum((lam - 1) * v0 * vN * r)
            if (s * A * C <= E * E) != (sv**2 >= s - 1e-12): mism += 1
        else:
            G0 = np.sum(v0**2 / (lam - 2)); GN = np.sum(vN**2 / (lam - 2))
            X0 = np.sum(v0 * vN / lam); X2 = np.sum(v0 * vN / (lam - 2))
            if ((X0 + X2)**2 <= G0 * GN) != (sv >= 1 - 1e-12): mism += 1
            df = math.prod(range(1, N, 2))
            xerr = max(xerr, abs(X2 + X0 * (N - 1) / (N + 1)), abs(abs(X0) - np.prod(g[1:]) / df**2) / abs(X0))
out.append(f"criteria vs direct SVD (N<=60, 6 m values): mismatches {mism}")
out.append(f"closed forms X(2) = -X(0)(N-1)/(N+1), |X(0)| = prod g/((N-1)!!)^2: max error {xerr:.2e}")
out.append("bulk ratio s*A*C/E^2 at s = s*, window a=0, odd N (limit sin^2(pi sqrt(s)/2) = %.5f):" % math.sin(math.pi * math.sqrt(s) / 2) ** 2)
for m in (0, 2, 30):
    row = []
    for N in (3, 11, 51, 201, 801):
        lam, v0, vN, g = eig(N, m); r = 1 / ((lam - 1) ** 2 - s)
        A = np.sum(v0**2 * r); C = np.sum(vN**2 * r); E = np.sum((lam - 1) * v0 * vN * r)
        row.append(f"N={N}: {s*A*C/E**2:.5f}")
    out.append(f"  m={m}: " + ", ".join(row))
out.append("even-N criterion ratio (X0+X2)^2/(G0 GN) (must be <= 1):")
for m in (0, .5, 5):
    row = []
    for N in (2, 4, 8, 32, 128, 512):
        lam, v0, vN, g = eig(N, m)
        G0 = np.sum(v0**2 / (lam - 2)); GN = np.sum(vN**2 / (lam - 2)); X0 = np.sum(v0 * vN / lam); X2 = np.sum(v0 * vN / (lam - 2))
        row.append(f"N={N}: {(X0+X2)**2/(G0*GN):.4f}")
    out.append(f"  m={m}: " + ", ".join(row))
print("\n".join(out)); open("../outputs/sw_secular_out.txt", "w").write("\n".join(out) + "\n")

"""Checks of the end-weight formulas used in the SW proof, against eigenvectors of the Stark matrix H(n=m+N, m),
for integer AND half-integer m (where the formulas hold by rational continuation in m at fixed N):
  bottom: w0(k-1)/w0(k) = k(N+m-k)/((m+k)(N-k)),   top: wN(k-1)/wN(k) = k(m+k)/((N+m-k)(N-k)),
with lam_k = N-1-2k; symmetry w(lam) = w(-lam); v0(lam)vN(lam) = prod g / P'(lam) (P = char poly of H)."""
import numpy as np
from stark import stark
err = 0; sym = 0; prod_err = 0; cnt = 0
for N in range(2, 31):
    for m in [x / 2 for x in range(0, 41)]:
        H, g = stark(N, m); lam, U = np.linalg.eigh(H)
        order = np.argsort(-lam); lam = lam[order]; U = U[:, order]   # lam_k = N-1-2k
        w0 = U[0] ** 2; wN = U[-1] ** 2
        for k in range(1, N):
            r0 = k * (N + m - k) / ((m + k) * (N - k)); rN = k * (m + k) / ((N + m - k) * (N - k))
            err = max(err, abs(w0[k-1] / w0[k] - r0) / r0, abs(wN[k-1] / wN[k] - rN) / rN)
        sym = max(sym, np.abs(w0 - w0[::-1]).max(), np.abs(wN - wN[::-1]).max())
        lv = N - 1 - 2 * np.arange(N)
        for k in range(N):
            Pp = np.prod([float(lv[k] - lv[j]) for j in range(N) if j != k])
            if abs(U[0, k] * U[-1, k]) > 1e-8: prod_err = max(prod_err, abs(abs(U[0, k] * U[-1, k]) - np.prod(g[1:]) / abs(Pp)) / abs(U[0, k] * U[-1, k]))
        cnt += 1
lines = [f"(N,m) pairs: {cnt} (N<=30, m in 0..20 step 1/2)",
         f"max relative error of the two ratio formulas (float64 eigenvectors; tiny end weights limit accuracy): {err:.2e}",
         f"max symmetry defect |w(lam)-w(-lam)|: {sym:.2e}",
         f"max relative error of |v0 vN| = prod g/|P'(lam)| (components with |v0 vN| > 1e-8): {prod_err:.2e}"]
print("\n".join(lines)); open("../outputs/weights_check_out.txt", "w").write("\n".join(lines) + "\n")

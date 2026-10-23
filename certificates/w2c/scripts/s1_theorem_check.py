"""Checks for Theorem S1-0 (s = 1, m1 = m2 = 0): with g_b = b,
 (Qg)_b - g_b = b[(h1-h2)^2(b) + (h1-h2)^2(b+1)] + c_b - c_{b+1} - b  and  c_b - c_{b+1} >= b + 1/2,
so lambda_min(Q) >= 1 + 1/(2 hi) by Barta / Collatz-Wielandt.  Here n_i = 2j_i + 1 (pairs (j_i, j_i)), b = 1..min(n)-1.
Also checks the s=1 reduction formula against the 2D block (done in s1.py) and lists lambda for the minimising family."""
import numpy as np
from s1 import Qmat, h2
worst_c = 9e9; worst_b = 9e9; cnt = 0; lam_ok = True
for n1 in range(2, 401):
    for n2 in range(n1, 401, 2):          # N1 + N2 even: n1, n2 same parity (N = n here)
        Q, bs = Qmat(float(n1), 0.0, float(n2), 0.0)
        if len(bs) == 0: continue
        c = lambda L: 2 * np.sqrt(h2(n1, 0, L) * h2(n2, 0, L))
        for b in bs:
            worst_c = min(worst_c, c(b) - c(b + 1) - b - 0.5)
        g = bs.copy(); r = (Q @ g) / g
        worst_b = min(worst_b, r.min() - 1 - 1 / (2 * bs[-1]))
        if n2 <= 120:
            lam = np.linalg.eigvalsh(Q).min()
            if lam < 1 + 1 / (2 * bs[-1]) - 1e-9: lam_ok = False
        cnt += 1
lines = [f"pairs (n1, n2), 2 <= n1 <= n2 <= 400, n1 = n2 mod 2: {cnt}",
         f"min over b of c_b - c_(b+1) - (b + 1/2): {worst_c:.3e}  (>= 0 claimed)",
         f"min over b of (Qg)_b/g_b - (1 + 1/(2 hi)): {worst_b:.3e}  (>= 0 claimed)",
         f"lambda_min(Q) >= 1 + 1/(2 hi) for n2 <= 120: {lam_ok}"]
for n in (3, 5, 9, 15, 31, 61, 121, 241):
    Q, bs = Qmat(float(n), 0.0, float(n), 0.0)
    lines.append(f"identical pairs n = {n}: lambda_min = {np.linalg.eigvalsh(Q).min():.6f}, bound 1 + 1/(2hi) = {1 + 1/(2*bs[-1]):.6f}")
print("\n".join(lines)); open("../outputs/s1_theorem_check_out.txt", "w").write("\n".join(lines) + "\n")

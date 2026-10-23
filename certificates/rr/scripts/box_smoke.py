"""Smoke test: rigorous box evaluation vs the reference float implementation on a few configurations, and an exact
rational evaluation of W at small n1 (Fractions) to confirm the interval upper bound encloses the exact value."""
from fractions import Fraction as Fr
import box_interval as b, ref_s2_proof as r

def W_exact(n1, m1, n2, m2):
    dn = n2 - n1; k0 = max(m1, m2 - 2, 0); c_lo, c_hi = k0 + 1, n2 - 2
    A = {}; acc = Fr(0)
    for k in range(k0, c_hi):
        L = k + 1
        p2 = Fr((n1 * n1 - L * L) * (L * L - m1 * m1) * (L + 2), 2 * (2 * L + 1))
        g = {-1: (n2 - k - 1) * (n2 - k - 2), 0: (n2 - k - 2) * (n2 + k + 2), 1: (n2 + k + 1) * (n2 + k + 2)}[dn]
        acc += Fr(k + 1) / (g * p2); A[k] = acc
    W = Fr(0)
    for c in range(c_lo, c_hi + 1):
        Fc = {-1: (n2 - c) * (n2 - c - 1), 0: (n2 - c - 1) * (n2 + c + 1), 1: (n2 + c) * (n2 + c + 1)}[dn]
        K = c * (2 * c * c + 6 * c + 5)
        N = (n1 * n1 + m1 * m1 - n2 * n2 - m2 * m2) * K + 2 * n1 * n1 * m1 * m1 * (c + 3) \
            - 2 * n2 * n2 * m2 * m2 * (c - 1) - 2 * (n2 * n2 + m2 * m2) + 2 * (2 * c * c + 4 * c + 1) ** 2
        if N > 0: W += Fr(3 * N, 4 * (2 * c + 1) * (2 * c + 3)) * Fr(Fc, c) * A[c - 1]
    return W

if __name__ == '__main__':
    print('T0', b.T0, 'ref', r.T0, ' pi2/6', b.PI2_6)
    for cfg in [(21, 0, 22, 1), (21, 0, 20, 1), (21, 0, 21, 1), (40, 5, 41, 7), (60, 30, 59, 32), (500, 100, 500, 102), (999, 0, 1000, 1)]:
        ok, rr, W = b.config_ok(*cfg)
        line = f"{cfg}: rigorous ok={ok} margin={rr:.6f} W_hi={W:.9f} | ref {r.C3_config_ok(*cfg)}"
        if cfg[0] <= 60:
            We = W_exact(*cfg); line += f" | exact W={float(We):.12f} enclosed={Fr(W) >= We}"
        print(line)

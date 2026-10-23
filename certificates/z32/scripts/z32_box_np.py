"""Box scan (float64 exploration version): for each regime and parameter set with n1 <= NMAX, the best of
two valid bounds for F0, F1, W:
  count form (z32_bounds) and the exact-product ('telescoping') form for CR/RC chains:
     CR:  E_{ck}^2 <= (1/l(c)) prod_{t=c}^{k-1} p_t^2/l(t+1)   (l = worst-case lower bound of q^2)
     RC:  E_{ci}^2 <= (1/p_i^2) prod_{t=i+1}^{c-1} lbar_q(t)/p_t^2   (q upper... exact for Tp/Tm)
Reports the minimum of 8*sqrt(sigma^2 bound) per regime.
Usage: python z32_box_np.py NMAX
"""
import sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z32_bounds_np as zb

K, H = 1.5, 0.5


def CR_tele(lo, hi, ell, pex):
    """returns F, columns c, row norms R (exact-product bounds)."""
    c = zb.ar(lo, hi); L = ell(c); p = pex(c)
    n = len(c); s = np.zeros(n); s[:-1] = p[:-1] / L[1:]
    Phi = 0.0; F = 0.0
    for k in range(n):
        Phi = (s[k - 1] * Phi if k > 0 else 0.0) + 1 / L[k]; F += Phi
    R = np.zeros(n); R[-1] = 1 / L[-1]
    for k in range(n - 2, -1, -1):
        R[k] = 1 / L[k] + s[k] * L[k + 1] / L[k] * R[k + 1]
    return F, c, R


def RC_tele(lo, hi, pex, qex):
    """RC chain rows lo..hi, cols lo+1..hi+1, matched p.  E_{c,i}^2 = (1/p_i^2) prod_{t=i+1}^{c-1} q_t^2/p_t^2.
    Column c's entries: rows i<c.  R_c = sum_{i<c} E_{ci}^2 ;  F = sum_c R_c."""
    i = zb.ar(lo, hi); p = pex(i); q = qex(i)
    n = len(i)
    rho = q / p                     # same-row ratio q_t^2/p_t^2
    R = np.zeros(n)                 # R for column i+1
    acc = 0.0
    for k in range(n):
        acc = (acc * rho[k] if k > 0 else 0.0) + 1 / p[k]
        R[k] = acc
    return R.sum(), i + 1, R


def best(name, n1, m=None):
    F0c, F1c, Wc = zb.regime_bounds(name, n1, m)
    if name in ('R1', 'R1c', 'R2+', 'R2-'):
        if name == 'R1':
            a = m; m2w = a - K; mu = 0.0 if abs((a + .5) - round(a + .5)) < 1e-9 else .5; lo0 = lo1 = a
            n2L, m2L = n1 + K, mu
        elif name == 'R1c':
            a = H; m2w = 0.0; lo0, lo1 = H, K; n2L, m2L = n1 + K, 0.0
        else:
            a = m; m2w = a + H if name == 'R2+' else a - H; lo0 = a; n2L, m2L = n1 + K, m2w
        F0t, c, R0 = CR_tele(lo0, n1 - 1, lambda c: zb.q0(n1 + K, m2w, c), lambda t: zb.p0(n1, a, t))
        lam = np.maximum(zb.Lam(c, n1, a, n2L, m2L), 0)
        Wt = (lam * R0).sum()
        if name in ('R1', 'R1c'):
            F1t, _, _ = CR_tele(lo1, n1 - 1, lambda c: zb.q1(n1 + K, m2w, c), lambda t: zb.p1(n1, a, t))
        else:
            F1t = F1c
        return min(F0c, F0t), min(F1c, F1t), min(Wc, Wt)
    if name in ('Tp', 'Tm'):
        n2 = n1 + H if name == 'Tp' else n1 - H
        hi0 = n1 - 2 if name == 'Tp' else n1 - 3
        F0t, c, R0 = RC_tele(0.0, hi0, lambda i: zb.p0(n1, 0.0, i), lambda i: zb.q0(n2, H, i))
        Wt = (np.maximum(zb.Lam(c, n1, 0.0, n2, H), 0) * R0).sum()
        return min(F0c, F0t), F1c, min(Wc, Wt)
    return F0c, F1c, Wc


def b8(F0, F1, W):
    if F0 == 0:
        return float('inf')
    return 8 * math.sqrt(zb.sigma2(F0, F1, W))


if __name__ == '__main__':
    NMAX = float(sys.argv[1])
    t0 = time.time()
    for name in ('R1', 'R2+', 'R2-', 'R4', 'R1c', 'R4c', 'Tp', 'Tm'):
        worst = (1e9, None)
        n1d = 4
        while n1d / 2 <= NMAX:
            n1 = n1d / 2
            if name in ('R1', 'R2+', 'R2-'):
                amin = {'R1': 1.5, 'R2+': 0.5, 'R2-': 1.0}[name]
                ms = [x / 2 for x in range(int(2 * amin), n1d) if (n1d - x) % 2 == 0 and n1 - x / 2 >= 2]
            elif name == 'R4':
                ms = [x / 2 for x in range(3, n1d + 2) if x / 2 - 1.5 <= n1 - 1]
            else:
                ms = [None]
                if name == 'R1c' and n1d % 2 == 0: n1d += 1; continue
                if name in ('R4c', 'Tp', 'Tm') and n1d % 2 == 1: n1d += 1; continue
                if name == 'Tm' and n1 < 3: n1d += 1; continue
            for m in ms:
                v = b8(*best(name, n1, m))
                if v < worst[0]:
                    worst = (v, n1, m)
            n1d += 1
        print(f'{name:4s} min 8*sigma bound = {worst[0]:.4f} at n1={worst[1]} m={worst[2]}  ({time.time()-t0:.1f}s)')

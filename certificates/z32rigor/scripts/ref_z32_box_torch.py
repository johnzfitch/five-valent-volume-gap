"""Certified finite box for the j1 = 3/2 regime bounds (float64, vectorised over all parameter sets).

For every parameter set of a regime with n = n1 <= NMAX it computes upper bounds F0, F1, W as the minimum of
the valid forms (count form, and the exact-product 'telescoping' form for CR / RC chains), then the lower bound
  sigma^2 >= min( (1 - tau(4/3 + W))/F0 , tau/((1+tau) F1) )   at the computed optimal tau.
Rigour: all link quantities are products of exactly representable numbers (half-integers < 2^26) and every
accumulation is a sum/product of nonnegative terms, so each F, R, W carries relative rounding error at most
(number of floating operations) * 2^-53 < 1e-11 here; Lambda is a difference of nonnegative terms and gets an
explicit absolute error allowance 2^-40 * (sum of the magnitudes).  We inflate F0, F1, W by (1 + 1e-9), add the
Lambda allowance, and deflate the final sigma^2 by (1 - 1e-9).  IEEE-754 round-to-nearest +,-,*,/ on the device
is assumed (true for CPU numpy/torch and for CUDA float64 without fast-math).

Usage: python z32_box_torch.py REGIME NMAX device [out.json]
  REGIME in R1 R2+ R2- R4 R1c R4c Tp Tm
"""
import sys, json, time, math
import torch

DT = torch.float64
H, K = 0.5, 1.5
INFL = 1 + 1e-9
DEFL = 1 - 1e-9
LAM_ERR = 2.0 ** -40
KAP = 4.0 / 3.0


def P(n, m, L):
    return (n * n - L * L) * (L * L - m * m)


def Lam_terms(c, n1, m1, n2, m2):
    """returns (Lambda, magnitude) with Lambda = A1 - B1 + A2 - B2."""
    A1 = P(n1, m1, c + 1) * c * (c + 2) / ((2 * c + 1) * (2 * c + 3))
    B1 = P(n1, m1, c) * (2 * c + 3) / (4 * (2 * c + 1))
    A2 = P(n2, m2, c + H) * (2 * c + 3) * (2 * c - 1) / (16 * c * (c + 1))
    B2 = P(n2, m2, c + K) * c / (4 * (c + 1))
    lam = (A1 - B1) + (A2 - B2)
    mag = A1.abs() + B1.abs() + A2.abs() + B2.abs()
    return lam, mag


def q0(n2, m2, b): return 3 / 16 * P(n2, m2, b + K) * b / (b + 1)
def p0(n1, m1, b): return 3 / 16 * P(n1, m1, b + 1) * (2 * b + 5) / (2 * b + 3)
def q1(n2, m2, b): return 3 / 16 * P(n2, m2, b - H) * (b + 1) / b
def p1(n1, m1, b): return 3 / 16 * P(n1, m1, b + 1) * (2 * b - 1) / (2 * b + 1)


def pairs(name, NMAX, dev):
    """parameter sets (n, a) of the regime with n <= NMAX (a = m1, or m2 for R4); 1-D families: a = None."""
    ns, as_ = [], []
    n2max = int(round(2 * NMAX))
    if name in ('R1', 'R2+', 'R2-'):
        amin = {'R1': 1.5, 'R2+': 0.5, 'R2-': 1.0}[name]
        for n2 in range(4, n2max + 1):
            n = n2 / 2
            a = n - 2
            while a >= amin - 1e-9:
                ns.append(n); as_.append(a); a -= 1
    elif name == 'R4':
        # n1 >= 2, m2 >= 3/2, m2 <= n1 + 1/2, n1 - m2 = 1/2 mod 1
        for n2 in range(4, n2max + 1):
            n = n2 / 2
            m2 = n + 0.5
            while m2 >= 1.5 - 1e-9:
                ns.append(n); as_.append(m2); m2 -= 1
    else:
        start = {'R1c': 2.5, 'R4c': 2.0, 'Tp': 2.0, 'Tm': 3.0}[name]
        n = start
        while n <= NMAX + 1e-9:
            ns.append(n); as_.append(0.0); n += 1
    return torch.tensor(ns, dtype=DT, device=dev), torch.tensor(as_, dtype=DT, device=dev)


def CR_block(lo, hi, ellf, pf, lamf, dev):
    """CR chain columns c = lo + T - 1, T = 1..N (N = hi - lo + 1, per pair).
    Returns F_count, F_tele, W_count, W_tele (W_* = 0 if lamf is None)."""
    N = (hi - lo + 1).round()
    Tm = int(N.max().item())
    Fc = torch.zeros_like(lo); Ft = torch.zeros_like(lo); Wc = torch.zeros_like(lo); Wt = torch.zeros_like(lo)
    Phi = torch.zeros_like(lo)
    pprev = torch.zeros_like(lo)

    def column(T):
        m = (T <= N)
        c_safe = torch.where(m, lo + (T - 1), lo)
        ell = torch.where(m, ellf(c_safe), torch.ones_like(lo))
        p = torch.where(m, pf(c_safe), torch.zeros_like(lo))
        tau = hi - c_safe + 1
        lamup = None
        if lamf is not None:
            lam, mag = lamf(c_safe)
            lamup = torch.where(m, torch.clamp(lam + LAM_ERR * mag, min=0), torch.zeros_like(lo))
        return m, ell, p, tau, lamup

    for T in range(1, Tm + 1):
        m, ell, p, tau, lamup = column(T)
        Fc = Fc + torch.where(m, tau / ell, torch.zeros_like(ell))
        # forward recursion Phi_T = (p_{T-1}^2 Phi_{T-1} + 1)/ell_T
        Phi = torch.where(m, (pprev * Phi + 1) / ell, Phi)
        Ft = Ft + torch.where(m, Phi, torch.zeros_like(Phi))
        if lamup is not None:
            Wc = Wc + lamup * torch.where(m, tau / ell, torch.zeros_like(ell))
        pprev = p
    if lamf is not None:
        R = torch.zeros_like(lo)
        for T in range(Tm, 0, -1):          # backward recursion R_c = (1 + p_c^2 R_{c+1})/ell_c, recomputed
            m, ell, p, tau, lamup = column(T)
            R = torch.where(m, (1 + p * R) / ell, torch.zeros_like(R))
            Wt = Wt + lamup * R
    return Fc, Ft, Wc, Wt


def RR_block(lo, hi, pf, qf):
    """tall chain rows lo..hi: F <= sum_{i<hi}(hi-i)/p_i^2 + sum_{i>lo}(i-lo)/q_i^2 (0 if hi < lo)."""
    N = (hi - lo + 1).round()
    Tm = int(max(N.max().item(), 0))
    F = torch.zeros_like(lo)
    for T in range(1, Tm + 1):
        m = (T <= N)
        i = lo + (T - 1)
        i_safe = torch.where(m, i, lo + 1)
        a1 = torch.where(m & (i < hi), (hi - i_safe) / pf(i_safe), torch.zeros_like(lo))
        a2 = torch.where(m & (i > lo), (i_safe - lo) / qf(i_safe), torch.zeros_like(lo))
        F = F + a1 + a2
    return F


def RC_block(n, hi, n2, lamf):
    """Tp/Tm zz0: RC chain rows 0..hi, columns 1..hi+1, matched p; exact products.  Returns
    F_count, F_tele, W_count, W_tele."""
    Tm = int((hi + 1).max().item())
    zero = torch.zeros_like(n)
    Fc = zero.clone(); Ft = zero.clone(); Wc = zero.clone(); Wt = zero.clone()
    Rc = zero.clone(); acc = zero.clone()
    for T in range(1, Tm + 1):
        i = torch.full_like(n, float(T - 1))
        m = (i <= hi)
        pi = torch.where(m, p0(n, zero, i), torch.ones_like(n))
        qi = torch.where(m, q0(n2, torch.full_like(n, H), i), zero)
        cnt = hi - i + 1
        Fc = Fc + torch.where(m, cnt / pi, zero)
        Rc = Rc + torch.where(m, 1 / pi, zero)                 # count row norm of column i+1
        acc = torch.where(m, acc * (qi / pi) + 1 / pi, acc)    # exact-product row norm of column i+1
        Ft = Ft + torch.where(m, acc, zero)
        lam, mag = lamf(i + 1)
        lamup = torch.where(m, torch.clamp(lam + LAM_ERR * mag, min=0), zero)
        Wc = Wc + lamup * Rc
        Wt = Wt + lamup * acc
    return Fc, Ft, Wc, Wt


def regime_box(name, NMAX, dev):
    n, a = pairs(name, NMAX, dev)
    one = torch.ones_like(n)
    if name == 'R1':
        m2w = a - K
        mu = torch.where(((a + 0.5) - (a + 0.5).round()).abs() < 1e-9, 0 * one, 0.5 * one)
        F0c, F0t, Wc, Wt = CR_block(a, n - 1, lambda c: q0(n + K, m2w, c), lambda c: p0(n, a, c),
                                    lambda c: Lam_terms(c, n, a, n + K, mu), dev)
        F1c, F1t, _, _ = CR_block(a, n - 1, lambda c: q1(n + K, m2w, c), lambda c: p1(n, a, c), None, dev)
        F0, F1, W = torch.minimum(F0c, F0t), torch.minimum(F1c, F1t), torch.minimum(Wc, Wt)
    elif name in ('R2+', 'R2-'):
        m2 = a + H if name == 'R2+' else a - H
        F0c, F0t, Wc, Wt = CR_block(a, n - 1, lambda c: q0(n + K, m2, c), lambda c: p0(n, a, c),
                                    lambda c: Lam_terms(c, n, a, n + K, m2), dev)
        lo1 = a + 1 if name == 'R2+' else a
        F1 = RR_block(lo1, n - 1, lambda i: p1(n, a, i), lambda i: q1(n + K, m2, i))
        F0, W = torch.minimum(F0c, F0t), torch.minimum(Wc, Wt)
    elif name == 'R4':
        m2 = a; b0 = m2 - K
        F0 = RR_block(b0, n - 1, lambda i: p0(n, b0, i), lambda i: q0(n + K, m2, i))
        F1 = RR_block(m2 + H, n - 1, lambda i: p1(n, b0, i), lambda i: q1(n + K, m2, i))
        W = torch.zeros_like(n)
    elif name == 'R1c':
        z = 0 * one
        F0c, F0t, Wc, Wt = CR_block(H * one, n - 1, lambda c: q0(n + K, z, c), lambda c: p0(n, H * one, c),
                                    lambda c: Lam_terms(c, n, H * one, n + K, z), dev)
        F1c, F1t, _, _ = CR_block(K * one, n - 1, lambda c: q1(n + K, z, c), lambda c: p1(n, H * one, c), None, dev)
        F0, F1, W = torch.minimum(F0c, F0t), torch.minimum(F1c, F1t), torch.minimum(Wc, Wt)
    elif name == 'R4c':
        z = 0 * one
        F0 = RR_block(z, n - 1, lambda i: p0(n, z, i), lambda i: q0(n + K, H * one, i))
        F1 = RR_block(one, n - 1, lambda i: p1(n, z, i), lambda i: q1(n + K, H * one, i))
        W = torch.zeros_like(n)
    elif name in ('Tp', 'Tm'):
        z = 0 * one
        n2 = n + H if name == 'Tp' else n - H
        hi0 = n - 2 if name == 'Tp' else n - 3
        F0c, F0t, Wc, Wt = RC_block(n, hi0, n2, lambda c: Lam_terms(c, n, z, n2, H * one))
        F1 = RR_block(one, n - 1, lambda i: p1(n, z, i), lambda i: q1(n2, H * one, i))
        F0, W = torch.minimum(F0c, F0t), torch.minimum(Wc, Wt)
    else:
        raise ValueError(name)
    F0u, F1u, Wu = F0 * INFL, F1 * INFL, W * INFL
    A = 1 / F0u
    B = torch.where(F1u > 0, 1 / F1u, torch.full_like(F1u, float('inf')))
    k = KAP + Wu
    # optimal tau solves A k t^2 + (B - A + A k) t - A = 0  (where B finite)
    bb = B - A + A * k
    tau = (-bb + torch.sqrt(bb * bb + 4 * A * A * k)) / (2 * A * k)
    tau = torch.where(torch.isfinite(B), tau, torch.zeros_like(tau))
    s2 = torch.minimum(A * (1 - tau * k), torch.where(torch.isfinite(B), tau * B / (1 + tau), A))
    s2 = s2 * DEFL
    return n, a, F0u, F1u, Wu, s2


if __name__ == '__main__':
    name, NMAX, devname = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    dev = torch.device(devname)
    t0 = time.time()
    n, a, F0, F1, W, s2 = regime_box(name, NMAX, dev)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    el = time.time() - t0
    b8 = 8 * torch.sqrt(torch.clamp(s2, min=0))
    i = int(torch.argmin(b8).item())
    n_nan = int(torch.isnan(s2).sum().item()); n_bad = int((s2 <= 0).sum().item())
    res = {'regime': name, 'NMAX': NMAX, 'device': devname, 'count': int(n.numel()), 'seconds': el,
           'nan_count': n_nan, 'nonpositive_count': n_bad,
           'min_8sigma': float(b8[i]), 'argmin_n': float(n[i]), 'argmin_a': float(a[i]),
           'F0': float(F0[i]), 'F1': float(F1[i]), 'W': float(W[i])}
    print(json.dumps(res))
    if len(sys.argv) > 4:
        with open(sys.argv[4], 'w') as f:
            json.dump(res, f, indent=1)

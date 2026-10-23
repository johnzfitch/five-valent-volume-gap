"""Outward-rounded (interval) re-run of the j1 = 3/2 box of dichotomy_z32_20261021 (THEOREM.md sections 5, 7, 8.1).

Same bound as the original box: for each parameter set of each regime,
   sigma^2 >= min( A (1 - tau k), tau B/(1+tau) ),  A = 1/F0, B = 1/F1, k = 4/3 + W,
with F0, F1, W the best of the count and exact-product forms (Lemmas 6-8, Corollary 5) and tau > 0 any number
(here the float optimum; the inequality holds for every tau > 0, so tau needs no rigour).
Rigour (replacing the a-priori allowance of the original): every IEEE-754 binary64 operation result is pushed one ulp
outward with torch.nextafter, in the direction that keeps the quantity a bound of the required side:
  - links used as denominators (q, p of matched pairs, p, q in tall chains): lower bounds;
  - links used in numerators of the exact-product recursions (p_c^2 in Phi and R, q_i/p_i in RC): upper bounds;
  - Lambda: A1_up - B1_dn + A2_up - B2_dn, rounded up, then clamped at 0;
  - F0, F1, W and all partial sums/recursions: upper bounds; A, B: lower bounds; k: upper bound;
  - 1 - tau k: computed as dn(1 - up(tau k)), so the cancellation is covered exactly; its minimum is recorded.
Inputs (n, a, c, b, ...) are half-integers; their squares and the products c(c+2), (2c+1)(2c+3), b(b+1) etc. are exact
(< 2^53).  P(n,m,L) = (n^2-L^2)(L^2-m^2) is the product of two exact factors, rounded outward.
This relies only on correct rounding of +,-,*,/ in binary64 (CUDA and CPU torch without fast-math) and on nextafter.
Usage: python ibox.py REGIME NMAX device [out.json]
"""
import sys, json, time, math
import torch

DT = torch.float64
H, K = 0.5, 1.5
INF = float('inf')

def up(x):
    """one-ulp outward step; an exactly zero result is kept (it arises only from an exactly zero factor or term:
    every nonzero quantity here is a rational function of half-integers <= 2e5 with nonzero values far above the
    binary64 underflow threshold, so no positive true value is flushed to 0)."""
    return torch.where(x == 0, x, torch.nextafter(x, torch.full_like(x, INF)))
def dn(x): return torch.where(x == 0, x, torch.nextafter(x, torch.full_like(x, -INF)))
KAP_UP = math.nextafter(4.0 / 3.0, INF)          # >= 4/3

# --- exact-factor products, outward
def P_up(n, m, L): return up((n * n - L * L) * (L * L - m * m))
def P_dn(n, m, L): return dn((n * n - L * L) * (L * L - m * m))

# --- links: X_dn / X_up are lower / upper bounds of the squared link (valid for any sign of P)
def q0_dn(n2, m2, b): return dn(dn(dn(dn(3 * P_dn(n2, m2, b + K)) / 16) * b) / (b + 1))
def q0_up(n2, m2, b): return up(up(up(up(3 * P_up(n2, m2, b + K)) / 16) * b) / (b + 1))
def p0_dn(n1, m1, b): return dn(dn(dn(dn(3 * P_dn(n1, m1, b + 1)) / 16) * (2 * b + 5)) / (2 * b + 3))
def p0_up(n1, m1, b): return up(up(up(up(3 * P_up(n1, m1, b + 1)) / 16) * (2 * b + 5)) / (2 * b + 3))
def q1_dn(n2, m2, b): return dn(dn(dn(dn(3 * P_dn(n2, m2, b - H)) / 16) * (b + 1)) / b)
def p1_dn(n1, m1, b): return dn(dn(dn(dn(3 * P_dn(n1, m1, b + 1)) / 16) * (2 * b - 1)) / (2 * b + 1))
def p1_up(n1, m1, b): return up(up(up(up(3 * P_up(n1, m1, b + 1)) / 16) * (2 * b - 1)) / (2 * b + 1))

def Lam_up(c, n1, m1, n2, m2):
    """upper bound of Lambda_c = A1 - B1 + A2 - B2 (Proposition 4)."""
    A1 = up(up(P_up(n1, m1, c + 1) * (c * (c + 2))) / ((2 * c + 1) * (2 * c + 3)))
    B1 = dn(dn(P_dn(n1, m1, c) * (2 * c + 3)) / (4 * (2 * c + 1)))
    A2 = up(up(P_up(n2, m2, c + H) * ((2 * c + 3) * (2 * c - 1))) / (16 * c * (c + 1)))
    B2 = dn(dn(P_dn(n2, m2, c + K) * c) / (4 * (c + 1)))
    return up(up(up(A1 - B1) + A2) - B2)

def pairs(name, NMAX, dev):
    ns, as_ = [], []
    n2max = int(round(2 * NMAX))
    if name in ('R1', 'R2+', 'R2-'):
        amin = {'R1': 1.5, 'R2+': 0.5, 'R2-': 1.0}[name]
        for n2 in range(4, n2max + 1):
            n = n2 / 2; a = n - 2
            while a >= amin - 1e-9:
                ns.append(n); as_.append(a); a -= 1
    elif name == 'R4':
        for n2 in range(4, n2max + 1):
            n = n2 / 2; m2 = n + 0.5
            while m2 >= 1.5 - 1e-9:
                ns.append(n); as_.append(m2); m2 -= 1
    else:
        start = {'R1c': 2.5, 'R4c': 2.0, 'Tp': 2.0, 'Tm': 3.0}[name]
        n = start
        while n <= NMAX + 1e-9:
            ns.append(n); as_.append(0.0); n += 1
    return torch.tensor(ns, dtype=DT, device=dev), torch.tensor(as_, dtype=DT, device=dev)

def CR_block(lo, hi, ell_dn, p_up_f, lam_f):
    """CR chain, columns c = lo..hi; ell = matched (lower bound), p = coupling-to-next (upper bound).
    F_count = sum tau_c/ell_c,  Phi_T = (p_{T-1} Phi_{T-1} + 1)/ell_T,  F_tele = sum Phi,
    W_count = sum Lam+ tau/ell,  R_c = (1 + p_c R_{c+1})/ell_c,  W_tele = sum Lam+ R.  All upper bounds."""
    N = torch.round(hi - lo + 1)
    Tm = int(N.max().item()); z = torch.zeros_like(lo); one = torch.ones_like(lo)
    Fc = z.clone(); Ft = z.clone(); Wc = z.clone(); Wt = z.clone(); Phi = z.clone(); pprev = z.clone()
    def column(T):
        m = (T <= N)
        c = torch.where(m, lo + (T - 1), lo)
        ell = torch.where(m, ell_dn(c), one)
        p = torch.where(m, p_up_f(c), z)
        tau = hi - c + 1
        lam = torch.where(m, torch.clamp(lam_f(c), min=0), z) if lam_f is not None else None
        return m, ell, p, tau, lam
    for T in range(1, Tm + 1):
        m, ell, p, tau, lam = column(T)
        tl = up(tau / ell)
        Fc = torch.where(m, up(Fc + tl), Fc)
        Phi = torch.where(m, up(up(up(pprev * Phi) + 1) / ell), Phi)
        Ft = torch.where(m, up(Ft + Phi), Ft)
        if lam is not None:
            Wc = torch.where(m, up(Wc + up(lam * tl)), Wc)
        pprev = p
    if lam_f is not None:
        R = z.clone()
        for T in range(Tm, 0, -1):
            m, ell, p, tau, lam = column(T)
            R = torch.where(m, up(up(1 + up(p * R)) / ell), z)
            Wt = torch.where(m, up(Wt + up(lam * R)), Wt)
    return Fc, Ft, Wc, Wt

def RR_block(lo, hi, p_dn_f, q_dn_f):
    """tall chain rows lo..hi (Lemma 8): F <= sum_{i<hi}(hi-i)/p_i + sum_{i>lo}(i-lo)/q_i, upper bound."""
    N = torch.round(hi - lo + 1)
    Tm = int(max(N.max().item(), 0)); z = torch.zeros_like(lo)
    F = z.clone()
    for T in range(1, Tm + 1):
        m = (T <= N)
        i = lo + (T - 1)
        isafe = torch.where(m, i, lo + 1)
        a1 = torch.where(m & (i < hi), up((hi - isafe) / p_dn_f(isafe)), z)
        a2 = torch.where(m & (i > lo), up((isafe - lo) / q_dn_f(isafe)), z)
        F = up(up(F + a1) + a2)
    return F

def RC_block(n, hi, n2, lam_f):
    """Tp/Tm zigzag 0: RC chain rows 0..hi, columns 1..hi+1, matched by p (Lemma 7), exact products."""
    Tm = int((hi + 1).max().item()); z = torch.zeros_like(n); one = torch.ones_like(n); half = torch.full_like(n, H)
    Fc = z.clone(); Ft = z.clone(); Wc = z.clone(); Wt = z.clone(); Rc = z.clone(); acc = z.clone()
    for T in range(1, Tm + 1):
        i = torch.full_like(n, float(T - 1)); m = (i <= hi)
        pi = torch.where(m, p0_dn(n, z, i), one)
        qi = torch.where(m, q0_up(n2, half, i), z)
        inv = up(1 / pi)
        Fc = torch.where(m, up(Fc + up((hi - i + 1) / pi)), Fc)
        Rc = torch.where(m, up(Rc + inv), Rc)
        acc = torch.where(m, up(up(acc * up(qi / pi)) + inv), acc)
        Ft = torch.where(m, up(Ft + acc), Ft)
        lam = torch.where(m, torch.clamp(lam_f(i + 1), min=0), z)
        Wc = torch.where(m, up(Wc + up(lam * Rc)), Wc)
        Wt = torch.where(m, up(Wt + up(lam * acc)), Wt)
    return Fc, Ft, Wc, Wt

def regime_box(name, NMAX, dev):
    n, a = pairs(name, NMAX, dev)
    one = torch.ones_like(n); z = 0 * one; half = H * one
    if name == 'R1':
        m2w = a - K
        mu = torch.where(((a + 0.5) - torch.round(a + 0.5)).abs() < 1e-9, z, half)
        F0c, F0t, Wc, Wt = CR_block(a, n - 1, lambda c: q0_dn(n + K, m2w, c), lambda c: p0_up(n, a, c),
                                    lambda c: Lam_up(c, n, a, n + K, mu))
        F1c, F1t, _, _ = CR_block(a, n - 1, lambda c: q1_dn(n + K, m2w, c), lambda c: p1_up(n, a, c), None)
        F0, F1, W = torch.minimum(F0c, F0t), torch.minimum(F1c, F1t), torch.minimum(Wc, Wt)
    elif name in ('R2+', 'R2-'):
        m2 = a + H if name == 'R2+' else a - H
        F0c, F0t, Wc, Wt = CR_block(a, n - 1, lambda c: q0_dn(n + K, m2, c), lambda c: p0_up(n, a, c),
                                    lambda c: Lam_up(c, n, a, n + K, m2))
        lo1 = a + 1 if name == 'R2+' else a
        F1 = RR_block(lo1, n - 1, lambda i: p1_dn(n, a, i), lambda i: q1_dn(n + K, m2, i))
        F0, W = torch.minimum(F0c, F0t), torch.minimum(Wc, Wt)
    elif name == 'R4':
        m2 = a; b0 = m2 - K
        F0 = RR_block(b0, n - 1, lambda i: p0_dn(n, b0, i), lambda i: q0_dn(n + K, m2, i))
        F1 = RR_block(m2 + H, n - 1, lambda i: p1_dn(n, b0, i), lambda i: q1_dn(n + K, m2, i))
        W = z.clone()
    elif name == 'R1c':
        F0c, F0t, Wc, Wt = CR_block(half, n - 1, lambda c: q0_dn(n + K, z, c), lambda c: p0_up(n, half, c),
                                    lambda c: Lam_up(c, n, half, n + K, z))
        F1c, F1t, _, _ = CR_block(K * one, n - 1, lambda c: q1_dn(n + K, z, c), lambda c: p1_up(n, half, c), None)
        F0, F1, W = torch.minimum(F0c, F0t), torch.minimum(F1c, F1t), torch.minimum(Wc, Wt)
    elif name == 'R4c':
        F0 = RR_block(z, n - 1, lambda i: p0_dn(n, z, i), lambda i: q0_dn(n + K, half, i))
        F1 = RR_block(one, n - 1, lambda i: p1_dn(n, z, i), lambda i: q1_dn(n + K, half, i))
        W = z.clone()
    elif name in ('Tp', 'Tm'):
        n2 = n + H if name == 'Tp' else n - H
        hi0 = n - 2 if name == 'Tp' else n - 3
        F0c, F0t, Wc, Wt = RC_block(n, hi0, n2, lambda c: Lam_up(c, n, z, n2, half))
        F1 = RR_block(one, n - 1, lambda i: p1_dn(n, z, i), lambda i: q1_dn(n2, half, i))
        F0, W = torch.minimum(F0c, F0t), torch.minimum(Wc, Wt)
    else:
        raise ValueError(name)
    # final combination, outward
    A = dn(1 / F0)
    finite1 = F1 > 0
    B = torch.where(finite1, dn(1 / torch.where(finite1, F1, one)), torch.full_like(F1, INF))
    k = up(KAP_UP + W)
    Af, Bf, kf = 1 / F0, torch.where(finite1, 1 / torch.where(finite1, F1, one), z), 4.0 / 3.0 + W
    bb = Bf - Af + Af * kf
    tau = (-bb + torch.sqrt(bb * bb + 4 * Af * Af * kf)) / (2 * Af * kf)      # float optimum; any tau > 0 is valid
    tau = torch.where(finite1, tau, z)
    gap = dn(1 - up(tau * k))                                             # 1 - tau k, lower bound
    s2a = dn(A * gap)
    s2b = torch.where(finite1, dn(dn(tau * torch.where(finite1, B, one)) / up(1 + tau)), torch.full_like(A, INF))
    s2 = torch.minimum(s2a, s2b)
    return n, a, F0, F1, W, tau, gap, s2

if __name__ == '__main__':
    name, NMAX, devname = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    dev = torch.device(devname); t0 = time.time()
    n, a, F0, F1, W, tau, gap, s2 = regime_box(name, NMAX, dev)
    if dev.type == 'cuda': torch.cuda.synchronize()
    el = time.time() - t0
    i = int(torch.argmin(s2).item())
    bad = int((~(s2 > 0)).sum().item()); nan = int(torch.isnan(s2).sum().item())
    tpos = int((tau[F1 > 0] <= 0).sum().item())
    s2min = float(s2[i])
    res = {'regime': name, 'NMAX': NMAX, 'device': devname, 'count': int(n.numel()), 'seconds': round(el, 1),
           'nan_count': nan, 'nonpositive_count': bad, 'nonpositive_tau': tpos,
           's2_min_lower_bound': repr(s2min), 'min_8sigma_lower_bound': repr(math.nextafter(8 * math.sqrt(s2min) * (1 - 1e-15), 0)),
           'argmin_n': float(n[i]), 'argmin_a': float(a[i]), 'F0': float(F0[i]), 'F1': float(F1[i]), 'W': float(W[i]),
           'min_gap_1_minus_tau_k': repr(float(gap[F1 > 0].min())) if bool((F1 > 0).any()) else None}
    print(json.dumps(res))
    if len(sys.argv) > 4:
        with open(sys.argv[4], 'w') as f: json.dump(res, f, indent=1)

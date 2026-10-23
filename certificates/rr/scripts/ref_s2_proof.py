"""j1 = 2, W2 class: closed-form and finite checks of Theorem 2 (THEOREM.md section 4).

All parameters are integers: pairs (n_i, m_i) with N_i = n_i - m_i >= 3.  lam0 = (q0/8)^2, q0 = 4.5579607.
Sections:
  [K]  rigorous constants S(m), T0
  [NN] two non-wide walls: sigma^2 >= 0.6079  (q >= 6.237) for all sizes
  [C1] chained, n2 >= n1+2: inequality (*) at the worst corner, plus a grid scan of the closed forms
  [T3] chained C3 tail, n1 >= 1000: W <= 72 H + ..., inequality (*) at n1 = 1000 and monotone beyond
  [B3] chained C3, 21 <= n1 <= 999: exact-sum evaluation on the two extremal slices (m2 = m1+2, and (m1,m2)=(0,1))
Usage: python s2_proof.py > ../outputs/s2_proof_out.txt
"""
import math, time, sys
import numpy as np
from fractions import Fraction as Fr

Q0 = 4.5579607
LAM0 = (Q0 / 8) ** 2
SAFE = 1e-9            # relative safety factor applied to every floating comparison


def H(n): return math.fsum(1.0 / i for i in range(1, n + 1)) if n > 0 else 0.0


# ---------------- [K] constants ----------------
def S_of(m):
    """S(m) = sum_{i>=1} 1/(i(i+2m)) : pi^2/6 for m=0, H_{2m}/(2m) for m>=1"""
    return math.pi ** 2 / 6 if m == 0 else H(2 * m) / (2 * m)


def T0_upper(K=20000):
    """T0 = sum_{k>=1} phi(k)/(k+1)^2, phi(k)=(2k+1)(2k+3)/((k+2)(2k-1)) (decreasing for k>=1).
    exact partial sum to K in rationals is slow; use fsum (all terms positive, relative error < 1e-12) + tail bound."""
    phi = lambda k: (2 * k + 1) * (2 * k + 3) / ((k + 2) * (2 * k - 1))
    part = math.fsum(phi(k) / (k + 1) ** 2 for k in range(1, K + 1))
    tail = phi(K + 1) / (K + 1)            # sum_{k>K} 1/(k+1)^2 <= 1/(K+1)
    return (part + tail) * (1 + 1e-12)


T0 = T0_upper()


def T_slice(m1, m2):
    """sup-type check: T(m1,m2) = sum_{k>=m2} phi(k)/((k+1)^2-m1^2) <= T0 for the C1/C3 index sets"""
    phi = lambda k: (2 * k + 1) * (2 * k + 3) / ((k + 2) * (2 * k - 1))
    s = math.fsum(phi(k) / ((k + 1) ** 2 - m1 * m1) for k in range(max(m2, 1), max(m2, 1) + 200000))
    return s


# ---------------- closed-form lower bounds ----------------
def F_wall(n, m, nO, mO, typ):
    """closed-form upper bound for the count-form Frobenius^2 of a wall chain whose matched pair is (n,m)
    (pair 1 of that orientation) and other pair (nO,mO); typ in RC, CR, RR (Lemma 4.3)."""
    if typ == 'RC':
        return 4 * S_of(m) / (n + m + 1)
    if typ == 'CR':
        return 4 * (math.pi ** 2 / 6) / nO
    if typ == 'RR':
        return max(4 * S_of(m) / (n + m + 1), 1.25 * H(nO - 3) / nO)
    raise ValueError


def s_M_closed(n1, m1, n2, m2):
    """sigma_min(M+)^2 lower bound (Lemma 4.4), chained regimes (m1<m2, m2>=1)"""
    fl = (4.0 / 3.0) * T0 / (n1 + m2 + 1)
    fu = (40.0 / 7.0) * H(n2) / (n2 * (n2 + 1)) if n2 >= n1 else 0.0
    return 1.0 / max(fl, fu)


def star_ok(sW, sM, C):
    """inequality (*): sW > lam0 and sM >= lam0 (1 + C sW/(sW - lam0)); conservative safety factors"""
    sW = sW * (1 - SAFE); sM = sM * (1 - SAFE); C = C * (1 + SAFE)
    if sW <= LAM0: return False, 0.0
    need = LAM0 * (1 + C * sW / (sW - LAM0))
    return sM >= need, sM / need


# ---------------- exact sums for C3 ----------------
def W_C3(n1, m1, n2, m2):
    """W = sum_c Lam_c^+ R_c with the certified majorant (Lemma 4.6), exact formula, float64 accumulation."""
    dn = n2 - n1
    k0 = max(m1, m2 - 2, 0)
    c_lo, c_hi = k0 + 1, n2 - 2
    if c_hi < c_lo: return 0.0
    ks = np.arange(k0, c_hi, dtype=np.float64)           # k = k0..c_hi-1
    L = ks + 1
    p2 = (n1 * n1 - L * L) * (L * L - m1 * m1) * (L + 2) / (2 * (2 * L + 1))
    if dn == -1: g = (n2 - ks - 1) * (n2 - ks - 2)
    elif dn == 0: g = (n2 - ks - 2) * (n2 + ks + 2)
    else: g = (n2 + ks + 1) * (n2 + ks + 2)
    term = (ks + 1) / (g * p2)
    A = np.cumsum(term)                                   # A[i] = sum_{k=k0}^{k0+i}
    cs = np.arange(c_lo, c_hi + 1, dtype=np.int64)
    if dn == -1: Fc = (n2 - cs) * (n2 - cs - 1)
    elif dn == 0: Fc = (n2 - cs - 1) * (n2 + cs + 1)
    else: Fc = (n2 + cs) * (n2 + cs + 1)
    Rc = Fc.astype(np.float64) / cs * A[cs - 1 - k0]
    # exact integer numerator of Lambda (int64 safe for n <= 1000: |terms| < 2e15)
    c = cs
    K = c * (2 * c * c + 6 * c + 5)
    N = (n1 * n1 + m1 * m1 - n2 * n2 - m2 * m2) * K + 2 * n1 * n1 * m1 * m1 * (c + 3) \
        - 2 * n2 * n2 * m2 * m2 * (c - 1) - 2 * (n2 * n2 + m2 * m2) + 2 * (2 * c * c + 4 * c + 1) ** 2
    Lam = 3.0 * np.maximum(N, 0).astype(np.float64) / (4.0 * (2 * c + 1) * (2 * c + 3))
    return float(np.sum(Lam * Rc))


def C3_config_ok(n1, m1, n2, m2):
    sW = 1.0 / F_wall(n1, m1, n2, m2, 'RC')
    sM = s_M_closed(n1, m1, n2, m2)
    W = W_C3(n1, m1, n2, m2) * (1 + 1e-9)          # float accumulation of positive terms: rel. error << 1e-9
    return star_ok(sW, sM, 1.5 + W) + (W,)


if __name__ == '__main__':
    t00 = time.time()
    print(f"[K] lam0 = {LAM0:.12f};  T0 <= {T0:.10f};  S(0) = {S_of(0):.10f}")
    phi = lambda k: (2 * k + 1) * (2 * k + 3) / ((k + 2) * (2 * k - 1))
    print(f"[K] phi decreasing on k>=1: {all(phi(k) > phi(k+1) for k in range(1, 100000))};  "
          f"phi(2) * (pi^2/6 - 5/4) = {phi(2)*(math.pi**2/6-1.25):.4f} < T0 (bound for T(m1,m1+2), m1>=0)")
    # sanity: T(m1,m2) <= T0 on the relevant index sets
    worstT = max([T_slice(0, 1)] + [T_slice(a, a + 2) for a in range(0, 60)])
    print(f"[K] max_m T(m1, m2) over (0,1) and m2=m1+2, m1<60: {worstT:.6f} (<= T0: {worstT <= T0})")

    # ---------------- [NN] ----------------
    Fmax = max(4 * (math.pi ** 2 / 6) / 4, 4 * (math.pi ** 2 / 6) / 5, 1.25 * H(2) / 5)
    print(f"[NN] every wall chain: F <= {Fmax:.6f}  => sigma^2 >= {1/Fmax:.6f}, q >= {8*math.sqrt(1/Fmax):.4f}"
          f"  (target {Q0})  {'PASS' if 8*math.sqrt(1/Fmax) > Q0 else 'FAIL'}")

    # ---------------- [C1] ----------------
    n1, m1, n2, m2 = 3, 0, 22, 1
    sW = 1 / F_wall(n1, m1, n2, m2, 'RR'); sM = s_M_closed(n1, m1, n2, m2)
    ok, ratio = star_ok(sW, sM, 1.5)
    print(f"[C1] worst corner (n1,m1,n2,m2)=(3,0,22,1): sW={sW:.5f} sM={sM:.5f} C=3/2  (*) {ok}  margin sM/need = {ratio:.4f}")
    bad = 0; cnt = 0; mn = 9
    for n1 in range(3, 120):
        for m1 in range(0, n1 - 2):
            for n2 in range(max(n1 + 2, 22), 400, 7):
                for m2 in [1, m1 + 2, m1 + 3, m1 + 7, n2 - 3]:
                    if not (m2 >= m1 + 2 or (m1, m2) == (0, 1)) or m2 > n2 - 3: continue
                    sW = 1 / F_wall(n1, m1, n2, m2, 'RR'); sM = s_M_closed(n1, m1, n2, m2)
                    ok, r = star_ok(sW, sM, 1.5); cnt += 1; bad += (not ok); mn = min(mn, r)
    print(f"[C1] grid scan of the closed forms: {cnt} configurations, {bad} failures, min margin {mn:.4f}")

    # ---------------- [T3] C3 tail ----------------
    def Wbar(n):
        Hn = H(n)
        return max(12 * Hn * (6 + (2 * Hn + 5.6) / n),          # dn = -1 (nu=1, a=1)
                   24 * Hn * (1 + (2 * Hn + 5.6) / n),          # dn = 0
                   48 * Hn * (1 + (2 * Hn + 5.6) / n))          # dn = +1
    def Creq(n1, m1=0, m2=1):
        sW = 1 / F_wall(n1, m1, n1, m2, 'RC') * (1 - SAFE)
        sM = 3 * (n1 + m2 + 1) / (4 * T0) * (1 - SAFE)
        return (sM / LAM0 - 1) * (1 - LAM0 / sW)
    for n in (1000, 2000, 10 ** 4, 10 ** 5):
        print(f"[T3] n1={n}: Wbar={Wbar(n):.2f}, Creq-1.5={Creq(n)-1.5:.2f}  {'PASS' if Creq(n)-1.5 >= Wbar(n) else 'FAIL'}")
    # monotonicity beyond 1000: Creq increases by >= 3/(4 T0 lam0)*(1-lam0/sW) - ... per unit n, Wbar by <= 72/n + ...
    inc = min(Creq(n + 1) - Creq(n) for n in range(1000, 1100))
    dW = max(Wbar(n + 1) - Wbar(n) for n in range(1000, 1100))
    print(f"[T3] per-unit increments near 1000: Creq >= {inc:.4f}, Wbar <= {dW:.4f} (Creq slope constant, Wbar slope ~72/n decreasing)")

    # ---------------- [B3] C3 box on extremal slices ----------------
    t0 = time.time()
    worst = (9e9, None); fails = []; cnt = 0
    for n1 in range(21, 1000):
        for dn in (-1, 0, 1):
            n2 = n1 + dn
            slices = [(0, 1)] + [(a, a + 2) for a in range(0, n2 - 4)]
            for (m1, m2) in slices:
                if n1 - m1 < 3 or n2 - m2 < 3: continue
                ok, r, W = C3_config_ok(n1, m1, n2, m2)
                cnt += 1
                if r < worst[0]: worst = (r, (n1, m1, n2, m2), W)
                if not ok: fails.append(((n1, m1, n2, m2), r, max(n1 + m1, n2 + m2) - 1))
        if n1 % 100 == 0:
            print(f"   [B3] up to n1={n1}: {cnt} configurations, {len(fails)} failures, worst margin {worst[0]:.4f} at {worst[1]}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[B3] C3 slices, 21<=n1<=999: {cnt} configurations, {len(fails)} failures; worst margin sM/need = {worst[0]:.4f} at {worst[1]} (W={worst[2]:.3f})")
    if fails: print('   failures (config, margin, 2j_max):', fails[:20])
    print(f"total {time.time()-t00:.0f}s")

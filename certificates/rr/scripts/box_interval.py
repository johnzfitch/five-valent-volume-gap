"""Rigorous re-evaluation of the j1 = 2 box [B3] of dichotomy_fixed_j1_20261021 (Theorem 4.2, case C3, 21 <= n1 <= 999).
Re-implemented from THEOREM.md Lemmas 4.3, 4.6(i),(v), 4.7(b), 4.8 (not imported from s2_proof.py; that script is
copied as ref_s2_proof.py and used only for an agreement check of W).

Inequality checked for every configuration (n1, m1, n2, m2), n2 = n1 + dn, dn in {-1,0,1}, slices (m1,m2) = (0,1) and
(a, a+2), with n1-m1 >= 3, n2-m2 >= 3 (enumeration identical to s2_proof.py [B3]):
   sW > lam0  and  sM >= lam0 * (1 + (3/2 + W) * sW / (sW - lam0)),
   sW = (n1+m1+1)/(4 S(m1)),  S(0) = pi^2/6, S(m) = H_{2m}/(2m);
   sM = 1/max( (4/3) T0/(n1+m2+1), [n2>=n1] (40/7) H_{n2}/(n2(n2+1)) ),  T0 = sum_{k>=1} phi(k)/(k+1)^2;
   W  = sum_{c=k0+1}^{n2-2} Lam_c^+ (F_c/c) sum_{k=k0}^{c-1} (k+1)/(g_k p_k^2),  k0 = max(m1, m2-2, 0),
        p_k^2 = (n1^2-L^2)(L^2-m1^2)(L+2)/(2(2L+1)), L = k+1;  F_c/g_k = nu(k,c) of Lemma 4.7(b);
        Lam_c = 3 N_c/(4(2c+1)(2c+3)), N_c the integer polynomial of Lemma 4.3(d) (exact int64, magnitude asserted).
Rigour: lam0 = (45579607/8e7)^2 exact rational. S(m) for m >= 1 and H_{n2} exact rationals rounded outward; pi^2/6 and
T0 rigorous Arb enclosures (python-flint, 200 bits; T0 = partial sum to K plus the tail bound phi(K+1)/(K+1)).
W: every positive quantity is an upper bound obtained with outward rounding (np.nextafter after each IEEE operation on
exact or upper-bounded operands); prefix sums of nonnegative terms by |fl(sum)-sum| <= (k-1)u/(1-(k-1)u)*sum, applied
outward (factor 1+1.01 k u). The final inequality uses lower bounds on the left (sW, sM) and upper bounds on the right
(lam0, W); relative rounding of the last few operations is absorbed by explicit (1 +- 8u) factors. Any doubt = failure.
Usage: python box_interval.py nproc"""
import sys, time, math
import numpy as np
from fractions import Fraction as Fr
from multiprocessing import Pool
import flint
flint.ctx.prec = 200
u = 2.0 ** -53
LAM0 = Fr(45579607, 80000000) ** 2

def dn_(x): return np.nextafter(x, -np.inf)
def up_(x): return np.nextafter(x, np.inf)
def fdn(q):
    f = float(q)
    return f if Fr(f) <= q else math.nextafter(f, -math.inf)
def fup(q):
    f = float(q)
    return f if Fr(f) >= q else math.nextafter(f, math.inf)

def arb_bounds(a):
    return math.nextafter(float(a.lower()), -math.inf), math.nextafter(float(a.upper()), math.inf)
PI2_6 = arb_bounds(flint.arb.pi() ** 2 / 6)
def T0_bounds(K=20000):
    s = flint.arb(0)
    for k in range(1, K + 1):
        s += flint.arb((2 * k + 1) * (2 * k + 3)) / ((k + 2) * (2 * k - 1) * (k + 1) ** 2)
    phiK1 = Fr((2 * K + 3) * (2 * K + 5), (K + 3) * (2 * K + 1))
    tail = flint.arb(phiK1.numerator) / phiK1.denominator / (K + 1)
    return math.nextafter(float(s.lower()), -math.inf), math.nextafter(float((s + tail).upper()), math.inf)
T0 = T0_bounds()
_H = [Fr(0)]
for i in range(1, 2100): _H.append(_H[-1] + Fr(1, i))
def S_hi(m): return PI2_6[1] if m == 0 else fup(_H[2 * m] / (2 * m))

def W_hi(n1, m1, n2, m2):
    dn = n2 - n1
    k0 = max(m1, m2 - 2, 0); c_lo, c_hi = k0 + 1, n2 - 2
    if c_hi < c_lo: return 0.0
    k = np.arange(k0, c_hi, dtype=np.int64); L = k + 1
    a1 = n1 * n1 - L * L; a2 = L * L - m1 * m1; a3 = L + 2; a4 = 2 * (2 * L + 1)
    if dn == -1: g = (n2 - k - 1) * (n2 - k - 2)
    elif dn == 0: g = (n2 - k - 2) * (n2 + k + 2)
    else: g = (n2 + k + 1) * (n2 + k + 2)
    assert (a1 > 0).all() and (a2 > 0).all() and (g > 0).all()
    num = (L * a4).astype(np.float64)                 # exact integers < 2^53
    d1 = (g * a1); d2 = (a2 * a3)
    assert d1.max() < 2 ** 53 and d2.max() < 2 ** 53
    den_lo = dn_(d1.astype(np.float64) * d2.astype(np.float64))
    term_hi = up_(num / den_lo)
    A_hi = np.cumsum(term_hi)
    kk = np.arange(1, len(A_hi) + 1, dtype=np.float64)
    A_hi = up_(A_hi * (1 + 1.01 * kk * u))
    c = np.arange(c_lo, c_hi + 1, dtype=np.int64)
    if dn == -1: Fc = (n2 - c) * (n2 - c - 1)
    elif dn == 0: Fc = (n2 - c - 1) * (n2 + c + 1)
    else: Fc = (n2 + c) * (n2 + c + 1)
    Kc = c * (2 * c * c + 6 * c + 5)
    assert max(n1, n2) <= 1001
    N = (n1 * n1 + m1 * m1 - n2 * n2 - m2 * m2) * Kc + 2 * n1 * n1 * m1 * m1 * (c + 3) \
        - 2 * n2 * n2 * m2 * m2 * (c - 1) - 2 * (n2 * n2 + m2 * m2) + 2 * (2 * c * c + 4 * c + 1) ** 2
    assert np.abs(N).max() < 2 ** 62
    Np = np.maximum(N, 0)
    Np_hi = up_(Np.astype(np.float64))               # int -> float may round when >= 2^53: outward
    den = (4 * (2 * c + 1) * (2 * c + 3) * c)
    assert den.max() < 2 ** 53
    x = up_(Np_hi * Fc.astype(np.float64))
    x = up_(x * A_hi[c - 1 - k0])
    x = up_(3.0 * x)
    x = up_(x / den.astype(np.float64))
    s = np.sum(x)
    return float(up_(s * (1 + 1.01 * len(x) * u)))

def config_ok(n1, m1, n2, m2):
    sW = fdn(Fr(n1 + m1 + 1, 4)) / S_hi(m1) * (1 - 4 * u)
    fl = (4.0 / 3.0) * T0[1] / (n1 + m2 + 1) * (1 + 8 * u)
    fu = fup(Fr(40, 7) * _H[n2] / (n2 * (n2 + 1))) if n2 >= n1 else 0.0
    sM = (1.0 / max(fl, fu)) * (1 - 4 * u)
    W = W_hi(n1, m1, n2, m2)
    lam_hi = fup(LAM0)
    if not sW > lam_hi * (1 + 4 * u): return False, 0.0, W
    C = (1.5 + W) * (1 + 4 * u)
    f = sW / ((sW - lam_hi) * (1 - 4 * u)) * (1 + 4 * u)
    need = lam_hi * (1 + C * f) * (1 + 8 * u)
    return sM >= need, sM / need, W

def work(n1):
    out = []
    for dn in (-1, 0, 1):
        n2 = n1 + dn
        for (m1, m2) in [(0, 1)] + [(a, a + 2) for a in range(0, n2 - 4)]:
            if n1 - m1 < 3 or n2 - m2 < 3: continue
            ok, r, W = config_ok(n1, m1, n2, m2)
            out.append((n1, m1, n2, m2, ok, r, W))
    return out

if __name__ == '__main__':
    P = int(sys.argv[1]); t0 = time.time()
    with Pool(P) as p: res = [x for blk in p.map(work, range(21, 1000), chunksize=4) for x in blk]
    bad = [x for x in res if not x[4]]; worst = min(res, key=lambda x: x[5])
    import ref_s2_proof as ref
    import random; random.seed(7); smp = random.sample(res, 3000)
    maxrel = max(abs(x[6] - ref.W_C3(x[0], x[1], x[2], x[3])) / max(1e-300, x[6]) for x in smp)
    minrel = min((x[6] - ref.W_C3(x[0], x[1], x[2], x[3])) / max(1e-300, x[6]) for x in smp)
    lines = [f"constants: pi^2/6 in [{PI2_6[0]!r}, {PI2_6[1]!r}];  T0 in [{T0[0]!r}, {T0[1]!r}] (Arb, K=20000 + tail);  lam0 = {LAM0} exact",
             f"configurations: {len(res)} (s2_proof.py [B3] reports 1,489,059)",
             f"failures (rigorous): {len(bad)}",
             f"worst rigorous margin sM/need = {worst[5]:.6f} at (n1,m1,n2,m2) = {worst[:4]}, W_hi = {worst[6]:.6f}",
             f"W_hi - W_C3(reference float) relative, 3000 samples: min {minrel:.2e}, max |.| {maxrel:.2e} (W_hi must not be below the true W)",
             f"wall {time.time()-t0:.0f}s"]
    if bad: lines.append(f"first failures: {bad[:10]}")
    print("\n".join(lines)); open("../outputs/box_interval_out.txt", "w").write("\n".join(lines) + "\n")

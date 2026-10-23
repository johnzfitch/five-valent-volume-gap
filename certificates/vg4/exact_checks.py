"""Exact (rational / symbolic) checks supporting THEOREM.md.  CPU only.

E1  Symbolic one-variable inequalities used in the proof (sympy).
E2  For every sorted admissible tuple with 2*j_max <= NEX (default 40), in exact rational
    arithmetic on a_k^2:
      - discrete log-concavity  (a_k^2)^2 >= a_{k-1}^2 a_{k+1}^2  (Lemma 2),
      - adjacent ratios a_{k+1}^2/a_k^2 <= 6 and a_k^2/a_{k+1}^2 <= 4  (Lemma 3),
      - lower bounds a_k^2 >= s^2 t^2 k^2/16 and a_k^2 >= s t min(s,t) k (j_max+1)^2/96 (Lemma 4).
E3  Exact Sturm certificate: for every such tuple with d >= 2, B^T B - (3/16) I is positive
    definite (all LDL^T pivots > 0), i.e. every nonzero eigenvalue q has q^2 > 3/16, except
    j = (1/2,1/2,1/2,1/2) where q^2 = 3/16 exactly.  (B = odd-even block of the chain.)
E4  sympy characteristic polynomials for all tuples with 2j <= 6: exact smallest positive
    root agrees with the float pipeline.
E5  theta_d = sum_{s=1}^{d-1} 1/(s sqrt((d-s) min(s,d-s))) <= theta_3 for 2 <= d <= 10^5
    (float, margin) plus the analytic tail bound for d >= 20.
E6  d = 2 and d = 3: exact check of lam^2 >= j_max(j_max+1)/4 (Proposition 6).
Run: python exact_checks.py [NEX]   -> exact_checks_out.txt
"""
import sys, os, math, time
from fractions import Fraction
import sympy as sp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vg_core as vg

try:
    from flint import fmpq
except Exception:  # fall back to Fraction
    fmpq = Fraction

NEX = int(sys.argv[1]) if len(sys.argv) > 1 else 40
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

# ------------------------------------------------------------------ E1
x, y, k = sp.symbols("x y k", positive=True)
def pos_on_halfline(P):
    """exact: polynomial P(y) > 0 for all y >= 0 (value at 0 positive, no real root in [0, oo))."""
    P = sp.Poly(P, y)
    return P.eval(0) > 0 and all(r < 0 for r in P.real_roots())
# (a) concavity of log(x^4/(4x^2-1)) for x >= 2:  4(x^2-1/4)^2 - 2x^2(x^2+1/4) > 0
e = sp.expand(4 * (x**2 - sp.Rational(1, 4))**2 - 2 * x**2 * (x**2 + sp.Rational(1, 4)))
ey = sp.Poly(sp.expand(e.subs(x, 2 + y)), y)
log("E1a  4(x^2-1/4)^2-2x^2(x^2+1/4) at x=2+y has coefficients", ey.all_coeffs(), "-> positive for x >= 2:", pos_on_halfline(ey))
h = lambda t: sp.Rational(t**4, 4 * t * t - 1)
log("E1b  discrete check h(2)^2 >= h(1)h(3) for h=k^4/(4k^2-1):", h(2)**2, ">=", h(1) * h(3), h(2)**2 >= h(1) * h(3))
# (c) up-ratio U(k) = 16(k+1)^4(2k-1)/(k^2(2k+1)^2(2k+3)) <= 6 for k >= 1  (sup is about 5.843 at k = 1.259)
num = sp.expand(6 * k**2 * (2 * k + 1)**2 * (2 * k + 3) - 16 * (k + 1)**4 * (2 * k - 1))
py_ = sp.Poly(sp.expand(num.subs(k, 1 + y)), y)
log("E1c  6k^2(2k+1)^2(2k+3) - 16(k+1)^4(2k-1) at k=1+y: coeffs", py_.all_coeffs(), "positive for k >= 1:", pos_on_halfline(py_))
# (d) down-ratio factor: (k+1)^4(4k^2-1) - k^4(4k^2+8k+3) >= 0 for k >= 1
num = sp.expand((k + 1)**4 * (4 * k**2 - 1) - k**4 * (4 * k**2 + 8 * k + 3))
pd = sp.Poly(sp.expand(num.subs(k, 1 + y)), y)
log("E1d  (k+1)^4(4k^2-1)-k^4(4k^2+8k+3) at k=1+y: coeffs", pd.all_coeffs(), "positive for k >= 1:", pos_on_halfline(pd))
# (e) ((x-1)^2-B^2) x^2 <= (x-1)^2 (x^2-B^2)  identity check
B = sp.symbols("B", nonnegative=True)
log("E1e  (x-1)^2(x^2-B^2) - ((x-1)^2-B^2)x^2 =", sp.factor(sp.expand((x - 1)**2 * (x**2 - B**2) - ((x - 1)**2 - B**2) * x**2)))

# ------------------------------------------------------------------ E2, E3
def chain_sq(T):
    """exact c_s^2 (fmpq) for pairing (12)(34) and the doubled kmin."""
    return [fmpq(v.numerator, v.denominator) for v in vg.offdiag2_exact(*T)]

t0 = time.time()
tuples = vg.sorted_tuples(NEX)
nlc = nratio = nlow = ncert = 0
viol = []
eq_cases = []
for T in tuples:
    J1, J2, J3, J4 = T
    c2 = chain_sq(T)
    n = len(c2) + 1
    if n < 2:
        continue
    kmin2, kmax2 = vg.krange2(*T)
    jmax1 = fmpq(J4 + 2, 2)  # j_max + 1
    # E2: log-concavity and ratios
    for i in range(1, len(c2) - 1):
        nlc += 1
        if not (c2[i] * c2[i] >= c2[i - 1] * c2[i + 1]):
            viol.append(("logconc", T, i))
    for i in range(len(c2) - 1):
        nratio += 1
        if not (c2[i + 1] <= 6 * c2[i] and c2[i] <= 4 * c2[i + 1]):
            viol.append(("ratio", T, i))
    for i, v in enumerate(c2):
        s = i + 1; t = n - s; kk = fmpq(kmin2 + 2 * s, 2)
        nlow += 1
        if not (v * 16 >= s * s * t * t * kk * kk and v * 96 >= s * t * min(s, t) * kk * jmax1 * jmax1):
            viol.append(("lower", T, i))
    # E3: Sturm certificate on B^T B (m x m), B rows = odd sites, cols = even sites
    m = n // 2
    diag = []
    off2 = []
    for j in range(1, m + 1):
        dj = c2[2 * j - 2]  # c_{2j-1}^2
        if 2 * j - 1 < len(c2):
            dj = dj + c2[2 * j - 1]  # + c_{2j}^2
        diag.append(dj)
        if j < m:
            off2.append(c2[2 * j - 1] * c2[2 * j])  # (c_{2j} c_{2j+1})^2
    mu = fmpq(3, 16)
    p = diag[0] - mu
    piv = [p]
    for j in range(1, m):
        if p == 0:
            break
        p = diag[j] - mu - off2[j - 1] / p
        piv.append(p)
    ncert += 1
    if any(q < 0 for q in piv) or len(piv) < m:
        viol.append(("sturm", T, [str(q) for q in piv]))
    elif any(q == 0 for q in piv):
        eq_cases.append(T)
# E6: Proposition 6 (d = 2, 3): lam^2 >= j_max(j_max+1)/4, equality only at (1/2,1/2,j,j)
n6 = 0; eq6 = []; v6 = []
for T in tuples:
    c2 = chain_sq(T)
    jm = fmpq(T[3], 2)
    if len(c2) == 1:
        n6 += 1
        if c2[0] < jm * (jm + 1) / 4: v6.append(T)
        if c2[0] == jm * (jm + 1) / 4: eq6.append(T)
    elif len(c2) == 2:
        n6 += 1
        if not (c2[0] > jm * (jm + 1) / 4): v6.append(T)
log(f"E6  d in (2,3): {n6} tuples; violations of lam^2 >= j_max(j_max+1)/4 (first chain element bound): {len(v6)}; equality cases: {len(eq6)}, all of form (1/2,1/2,j,j): {all(t[0]==1 and t[1]==1 and t[2]==t[3] for t in eq6)}")
log(f"E2  tuples with 2j_max <= {NEX}: {len(tuples)}; log-concavity checks {nlc}, ratio checks {nratio}, lower-bound checks {nlow}")
log(f"E3  Sturm certificates {ncert}; equality (q^2 = 3/16 exactly) at: {eq_cases}")
log(f"E2/E3 violations: {len(viol)}  {viol[:10]}")
log(f"     time {time.time() - t0:.1f}s")

# ------------------------------------------------------------------ E4
q = sp.symbols("q")
worst = 0; cnt = 0
for T in vg.sorted_tuples(6):
    c2 = vg.offdiag2_exact(*T)
    n = len(c2) + 1
    if n < 2:
        continue
    # characteristic polynomial via the three-term recurrence (exact): P_k = q P_{k-1} - c_{k-1}^2 P_{k-2}
    P0, P1 = sp.Integer(1), q
    for v in c2:
        P0, P1 = P1, sp.expand(q * P1 - sp.Rational(v.numerator, v.denominator) * P0)
    roots = [r for r in sp.Poly(P1, q).real_roots() if r > 0]
    rmin = min(roots, key=lambda r: float(r))
    lam = vg.lapack_smallest_positive(np.sqrt(np.array([float(v) for v in c2])))
    worst = max(worst, abs(float(rmin) - lam)); cnt += 1
log(f"E4  exact charpoly smallest positive root vs float pipeline, {cnt} tuples with 2j<=6: max |diff| {worst:.2e}")

# ------------------------------------------------------------------ E5
def theta(d):
    s = np.arange(1, d, dtype=np.float64); t = d - s
    return float(np.sum(1.0 / (s * np.sqrt(t * np.minimum(s, t)))))
th3 = 1 / math.sqrt(2) + 0.5
mx = max((theta(d), d) for d in range(2, 100001))
log(f"E5  theta_3 = {th3:.15f}; max_(2<=d<=1e5) theta_d = {mx[0]:.15f} at d = {mx[1]}")
tail = 2.6124 * math.sqrt(2 / 20) + (2 / 20) * (1 + math.log(10))
log(f"E5  analytic tail bound for d >= 20: zeta(3/2) sqrt(2/d) + (2/d)(1+ln(d/2)) <= {tail:.4f} (decreasing in d) < theta_3")

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exact_checks_out.txt"), "w").write("\n".join(out) + "\n")

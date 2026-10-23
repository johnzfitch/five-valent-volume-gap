"""Checks of [LS] Theorems N1 and N1' (test-vector upper bounds, q/j_max -> 0).

N1, family G1 = (1/2; j-1/2, j, j, j), K = 2j:
  chain links a_{2i-1}^2 = 4(2j+1-i)(2j+1+i)i^2,  a_{2i-2}^2 = 4(2j+1-i)(2j+i)(i-1)i   (q-normalisation)
  (i)   the chain spectrum equals q from the independent Racah operator (2j <= 40);
  (ii)  test vector x_i = (-1)^i z_i phi_i, z_i^2 = (2j+2)/(i(2j+1+i)), phi_i = sin(pi ln i/(2 ln K)):
        q_true <= q_test <= 2 pi sqrt(2j(4j+1)(ln K + pi^2/6))/(ln K)^{3/2}, for K = 2^k up to 2^16;
        q_true from the smallest positive eigenvalue of the zero-diagonal chain (bisection, LAPACK stebz).
  (iii) the two elementary inequalities of the proof: sum_{i=2}^K i ln^2(i/(i-1)) <= ln K + pi^2/6 and
        sum_{i=2}^K phi_i^2/i >= ln K / 4, evaluated.
N1', family E1 = (1; j, j, j, j), n = 2j+1:
  pencil A_e = e(n^2-e^2)/2, mu_l = (2l+1)/(l(l+1)), l = 1..n-1, Dirichlet at l = 0 (Prop. R);
  (iv)  pencil q equals the Racah q (2j <= 40);
  (v)   q_true <= q_test <= 8 pi n sqrt((ln n + pi^2/6)/(2 ln^3(n-1))) for n up to 2^16+1.
Usage: python rb_N1.py > ../outputs/rb_N1_out.txt
"""
import sys, os, math, time
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C


def G1_links(j2):          # j2 = 2j (integer), returns a_1..a_{4j-1}
    K = j2
    a = np.zeros(2 * K)     # a[k] for k = 1..2K-1
    for i in range(1, K + 1):
        a[2 * i - 1] = math.sqrt(4 * (K + 1 - i) * (K + 1 + i) * i * i)
        if i >= 2:
            a[2 * i - 2] = math.sqrt(4 * (K + 1 - i) * (K + i) * (i - 1) * i)
    return a[1:]


def chain_qmin(off):
    n = len(off) + 1
    ev = eigvalsh_tridiagonal(np.zeros(n), off, select='i', select_range=(n // 2, n // 2))
    return float(ev[0])


def N1_test(j2):
    K = j2; Lam = math.log(K)
    i = np.arange(1, K + 1, dtype=float)
    a = G1_links(j2)
    d = a[0::2]                     # a_{2i-1}, i=1..K
    sub = a[1::2]                   # a_{2i}, i=1..K-1  -> couples x_i to row i+1 as a_{2(i+1)-2}
    z = np.sqrt((K + 2) / (i * (K + 1 + i)))
    phi = np.sin(math.pi * np.log(i) / (2 * Lam))
    x = (-1) ** i * z * phi
    Bx = d * x
    Bx[1:] += sub * x[:-1]
    qt = math.sqrt(np.sum(Bx ** 2) / np.sum(x ** 2))
    j = K / 2
    bound = 2 * math.pi * math.sqrt(2 * j * (4 * j + 1) * (Lam + math.pi ** 2 / 6)) / Lam ** 1.5
    s1 = float(np.sum(i[1:] * np.log(i[1:] / (i[1:] - 1)) ** 2))
    s2 = float(np.sum(phi[1:] ** 2 / i[1:]))
    return qt, bound, s1, Lam + math.pi ** 2 / 6, s2, Lam / 4


def E1_pencil(n):
    l = np.arange(1, n, dtype=float)
    e = np.arange(1, n + 1, dtype=float)             # edges e = 1..n, edge e between l=e-1 and l=e
    A = e * (n * n - e * e) / 2
    mu = (2 * l + 1) / (l * (l + 1))
    diag = A[:-1] + A[1:]                             # edges e=l and e=l+1
    off = -A[1:-1]                                    # edge e=l+1 couples l, l+1  (l=1..n-2)
    s = 1 / np.sqrt(mu)
    D = diag * s * s; O = off * s[:-1] * s[1:]
    lam = eigvalsh_tridiagonal(D, O, select='i', select_range=(0, 0))[0]
    # test vector
    Lam = math.log(n - 1)
    Phi = np.sin(math.pi * np.log(l) / (2 * Lam))
    Phi0 = np.concatenate([[0.0], Phi])
    En = np.sum(A[:-1] * (Phi0[1:] - Phi0[:-1]) ** 2)  # edges 1..n-1 ; edge n has A_n = 0
    Ms = np.sum(mu * Phi ** 2)
    qt = 8 * math.sqrt(En / Ms)
    bound = 8 * math.pi * n * math.sqrt((math.log(n) + math.pi ** 2 / 6) / (2 * Lam ** 3))
    return 8 * math.sqrt(lam), qt, bound


if __name__ == '__main__':
    t0 = time.time()
    worst = 0.0
    for j2 in range(2, 41):
        q, _ = C.qmin((1, j2 - 1, j2, j2, j2))
        qc = chain_qmin(G1_links(j2))
        worst = max(worst, abs(q - qc) / q)
    print(f"(i) G1 chain vs Racah operator, 2j = 2..40: max rel diff {worst:.2e}")
    print("(ii)/(iii) N1:  K=2j   q_true      q_test      bound      test/true  bound/test  "
          "sum i ln^2  <=  lnK+pi^2/6   sum phi^2/i  >=  lnK/4   q ln(M)/M")
    allok = True
    for k in list(range(1, 17)):
        K = 2 ** k
        if K < 2: continue
        qtrue = chain_qmin(G1_links(K))
        qt, b, s1, r1, s2, r2 = N1_test(K)
        ok = qtrue <= qt * (1 + 1e-12) and qt <= b and s1 <= r1 and s2 >= r2
        allok &= ok
        M = K
        print(f"   {K:6d} {qtrue:11.4f} {qt:11.4f} {b:11.4f}   {qt/qtrue:7.4f}   {b/qt:7.4f}   "
              f"{s1:8.4f} {r1:8.4f}   {s2:8.4f} {r2:8.4f}   {qtrue*math.log(M)/M:.4f}  {'ok' if ok else 'FAIL'}")
    for K in (3, 5, 7, 11, 100, 1001, 30001):
        qtrue = chain_qmin(G1_links(K)); qt, b, s1, r1, s2, r2 = N1_test(K)
        ok = qtrue <= qt * (1 + 1e-12) and qt <= b and s1 <= r1 and s2 >= r2
        allok &= ok
        print(f"   {K:6d} {qtrue:11.4f} {qt:11.4f} {b:11.4f}   {qt/qtrue:7.4f}   {b/qt:7.4f}   {'ok' if ok else 'FAIL'}")
    print(f"   N1 all checks: {'PASS' if allok else 'FAIL'}")
    worst = 0.0
    for j2 in range(2, 41, 2):
        q, _ = C.qmin((2, j2, j2, j2, j2))
        qp, _, _ = E1_pencil(j2 + 1)
        worst = max(worst, abs(q - qp) / q)
    print(f"(iv) E1 pencil (Prop. R) vs Racah operator, integer j = 1..20: max rel diff {worst:.2e}")
    print("(v) N1': n=2j+1   q_true      q_test      bound      test/true  bound/test")
    allok = True
    for k in range(2, 17):
        n = 2 ** k + 1
        qtr, qt, b = E1_pencil(n)
        ok = qtr <= qt * (1 + 1e-12) and qt <= b
        allok &= ok
        print(f"   {n:7d} {qtr:11.4f} {qt:11.4f} {b:11.4f}   {qt/qtr:7.4f}   {b/qt:7.4f}  {'ok' if ok else 'FAIL'}")
    print(f"   N1' all checks: {'PASS' if allok else 'FAIL'}   ({time.time()-t0:.0f}s)")

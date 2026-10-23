"""Exact (sympy) check of the algebra written out in the proofs of Lemmas H.2-H.4 (valence-six ladder).
With alpha_-^2(i) = (n+2+i) i/(n(2i+2)), beta_-^2(i) = (n-i)(i+2)/(n(2i+2)), alpha_+^2(i) = (n+1-i)/(2n+4),
beta_+^2(i) = (n+3+i)/(2n+4), x^2 = n(n+2)/16, y_h^2 = (n+1-h)(n+2+h)h(h+1)/16 and the step magnitudes
  M1(i): a1 = y_{i+1}/(x alpha_+(i+1)), b1 = x alpha_-(i+1) beta_-(i)/(y_{i+1} beta_-(i+1)), c1 = a1/beta_-(i+1)
  M2(i): a2 = y_{i+1}/(x alpha_-(i+1)), b2 = x alpha_+(i+1) beta_+(i)/(y_{i+1} beta_+(i+1)), c2 = a2/beta_+(i+1)
it checks:
  (D)  (a1 b1 a2' b2')^2 = (n+3+i)/(n+5+i) for the two-step factor M2(i+1) M1(i), and the telescoped product;
  (E1) b1/c1 = x^2 alpha_-(i+1) alpha_+(i+1) beta_-(i)/y_{i+1}^2;
  (E2) b2' a1/(c2' c1) = x^2 alpha_+(i+2) beta_+(i+1) alpha_-(i+2) beta_-(i+1)/y_{i+2}^2;
  (R)  a2'/c2' = beta_+(i+2); c1 c2' >= y_{i+1} y_{i+2}/x^2 (factor check);
  (Q)  beta_-(n-1) x^2/y_n = (n+2)/(8 sqrt(n(n+1))) and alpha_-(n) = 1, alpha_+(n)^2 + beta_+(n)^2 < 1, alpha_+(0)^2+beta_+(0)^2 = 1;
  (S)  partial-fraction bound for S used in Lemma H.4, numerically for odd n <= 2001.
Usage: python v6_lemma_algebra.py  (writes v6_lemma_algebra_out.txt beside it)."""
import os, math
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__))
n, i = sp.symbols('n i', positive=True)
am2 = lambda k: (n + 2 + k) * k / (n * (2 * k + 2))
bm2 = lambda k: (n - k) * (k + 2) / (n * (2 * k + 2))
ap2 = lambda k: (n + 1 - k) / (2 * n + 4)
bp2 = lambda k: (n + 3 + k) / (2 * n + 4)
x2 = n * (n + 2) / 16
y2 = lambda h: (n + 1 - h) * (n + 2 + h) * h * (h + 1) / 16
out = []
# squared magnitudes of the step entries
a1s = lambda k: y2(k + 1) / (x2 * ap2(k + 1))
b1s = lambda k: x2 * am2(k + 1) * bm2(k) / (y2(k + 1) * bm2(k + 1))
c1s = lambda k: a1s(k) / bm2(k + 1)
a2s = lambda k: y2(k + 1) / (x2 * am2(k + 1))
b2s = lambda k: x2 * ap2(k + 1) * bp2(k) / (y2(k + 1) * bp2(k + 1))
c2s = lambda k: a2s(k) / bp2(k + 1)
D = sp.simplify(a1s(i) * b1s(i) * a2s(i + 1) * b2s(i + 1) - (n + 3 + i) / (n + 5 + i))
out.append(f"(D) |det M1(i)|^2 |det M2(i+1)|^2 - (n+3+i)/(n+5+i) = {D}")
k = sp.symbols('k', integer=True, positive=True)
prod = sp.simplify(sp.product((n + 3 + 2 * k) / (n + 5 + 2 * k), (k, 1, (n - 3) / 2)))
out.append(f"(D) prod over i = 2,4,...,n-3 of (n+3+i)/(n+5+i) = {sp.simplify(prod)}  (expected (n+5)/(2n+2))")
for nn in (7, 9, 11, 51):
    val = sp.prod([sp.Rational(nn + 3 + ii, nn + 5 + ii) for ii in range(2, nn - 2, 2)])
    out.append(f"    n = {nn}: product = {val}, (n+5)/(2n+2) = {sp.Rational(nn + 5, 2 * nn + 2)}")
E1 = sp.simplify(b1s(i) / c1s(i) - x2 ** 2 * am2(i + 1) * ap2(i + 1) * bm2(i) / y2(i + 1) ** 2)
out.append(f"(E1) (b1/c1)^2 - [x^2 alpha_-(i+1) alpha_+(i+1) beta_-(i)/y_(i+1)^2]^2 = {E1}")
E2 = sp.simplify(b2s(i + 1) * a1s(i) / (c2s(i + 1) * c1s(i)) - x2 ** 2 * ap2(i + 2) * bp2(i + 1) * am2(i + 2) * bm2(i + 1) / y2(i + 2) ** 2)
out.append(f"(E2) (b2 a1/(c2 c1))^2 - [x^2 alpha_+(i+2) beta_+(i+1) alpha_-(i+2) beta_-(i+1)/y_(i+2)^2]^2 = {E2}")
R = sp.simplify(a2s(i + 1) / c2s(i + 1) - bp2(i + 2))
out.append(f"(R) (a2/c2)^2 - beta_+(i+2)^2 = {R}")
C = sp.simplify(c1s(i) * c2s(i + 1) - y2(i + 1) * y2(i + 2) / (x2 ** 2 * ap2(i + 1) * bm2(i + 1) * am2(i + 2) * bp2(i + 2)))
out.append(f"(R) (c1 c2)^2 - y_(i+1)^2 y_(i+2)^2/(x^4 alpha_+^2(i+1) beta_-^2(i+1) alpha_-^2(i+2) beta_+^2(i+2)) = {C}")
Q1 = sp.simplify(bm2(n - 1) * x2 ** 2 / y2(n) - (n + 2) ** 2 / (64 * n * (n + 1)))
out.append(f"(Q) (beta_-(n-1) x^2/y_n)^2 - (n+2)^2/(64 n (n+1)) = {Q1}")
out.append(f"(Q) alpha_-(n)^2 = {sp.simplify(am2(n))}, alpha_+(n)^2+beta_+(n)^2 = {sp.simplify(ap2(n)+bp2(n))}, "
           f"alpha_+(0)^2+beta_+(0)^2 = {sp.simplify(ap2(0)+bp2(0))}, y_n^2 - ((n+1)^2 2n)/16 = {sp.simplify(y2(n) - (n+1)**2*2*n/16)}")
ok = all((nn + 2) / (8 * math.sqrt(nn * (nn + 1))) <= math.sqrt(nn) / 4 for nn in range(2, 100001))
out.append(f"(Q) (n+2)/(8 sqrt(n(n+1))) <= sqrt(n)/4 for 2 <= n <= 1e5: {ok}  (for n >= 2 it follows from (n+2) <= 2n sqrt(n+1))")
worst = 0
for nn in range(5, 2002, 2):
    S = sum(nn * (nn + 2) / (j * (j + 1) * (nn + 1 - j) * (nn + 2 + j)) for j in range(3, nn))
    bnd = 1 / 3 + 2 * math.log(nn) / (nn + 2)
    worst = max(worst, S - bnd)
out.append(f"(S) max over odd 5 <= n <= 2001 of S - (1/3 + 2 ln n/(n+2)) = {worst:.3e} (must be <= 0); "
           f"max of 1/3 + 2 ln n/(n+2) over n >= 5 = {max(1/3 + 2*math.log(m)/(m+2) for m in range(5, 10000)):.4f} <= 4/3")
print("\n".join(out))
open(os.path.join(HERE, 'v6_lemma_algebra_out.txt'), 'w').write("\n".join(out) + "\n")

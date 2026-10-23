"""Exact symbolic checks for THEOREM.md (sympy).

H1. Lemma H, matrix elements: for j (x) j, the operator J1z - J2z in the coupled basis |L M> is tridiagonal with
    zero diagonal and |<L M|J1z-J2z|L-1 M>|^2 = (L^2-M^2)((2j+1)^2-L^2)/(4L^2-1)  (exact CG, j <= 3, all M >= 0).
H2. Lemma H, spectrum for real m: for N = 1..9, det(x - H(m)) = prod_i (x - (N-1-2i)) identically in the symbol m,
    where H(m) is the zero-diagonal Jacobi matrix with h_l^2 = ((m+N)^2-l^2)(l^2-m^2)/(4l^2-1), l = m+1..m+N-1.
F.  Lemma F: (kappa^2-beta^2)(A2^2-kappa^2) - (X^2-1) >= 0, both cases, as polynomials with nonnegative coefficients.
C2. (M+1)(2K+M+1) - (M+2)^2 = 2M^2+4M+1 with K = M+2+x (expansion).
C3. p q - T (p - q) has only nonnegative coefficients after b=y, M=y+z, K=M+3+x, E=Emin+w.
"""
import sympy as sp
from sympy.physics.quantum.cg import CG

out = []

# ---- H1
ok = True
for twoj in range(1, 7):
    j = sp.Rational(twoj, 2)
    for M in [sp.Rational(k, 1) for k in range(0, twoj + 1)]:
        Ls = [L for L in range(int(M), twoj + 1)]
        for L in Ls[1:]:
            # <L M| J1z |L-1 M> = sum_{m1} CG(j m1 j M-m1|L M) CG(j m1 j M-m1|L-1 M) * m1 ; J2z similarly
            s1 = 0; s2 = 0
            m1 = -j
            while m1 <= j:
                m2 = M - m1
                if abs(m2) <= j:
                    a = CG(j, m1, j, m2, L, M).doit(); b = CG(j, m1, j, m2, L - 1, M).doit()
                    s1 += a * b * m1; s2 += a * b * m2
                m1 += 1
            val = sp.nsimplify(sp.simplify((s1 - s2) ** 2))
            pred = sp.Rational((L * L - M * M) * ((twoj + 1) ** 2 - L * L), 4 * L * L - 1)
            if sp.simplify(val - pred) != 0:
                ok = False; out.append(f"H1 mismatch j={j} M={M} L={L}: {val} vs {pred}")
        for L in Ls:   # diagonal
            d = 0; m1 = -j
            while m1 <= j:
                m2 = M - m1
                if abs(m2) <= j:
                    a = CG(j, m1, j, m2, L, M).doit()
                    d += a * a * (m1 - m2)
                m1 += 1
            if sp.simplify(d) != 0:
                ok = False; out.append(f"H1 diagonal nonzero j={j} M={M} L={L}")
out.append(f"H1 (CG matrix elements of J1z-J2z for j<=3): {'PASS' if ok else 'FAIL'}")

# ---- H2
m, x = sp.symbols('m x')
ok = True
for N in range(1, 10):
    H = sp.zeros(N, N)
    for s in range(1, N):
        l = m + s
        h2 = ((m + N) ** 2 - l ** 2) * (l ** 2 - m ** 2) / (4 * l ** 2 - 1)
        H[s - 1, s] = sp.sqrt(h2); H[s, s - 1] = sp.sqrt(h2)
    # characteristic polynomial via the three-term recurrence (only h^2 enters)
    P0, P1 = sp.Integer(1), x
    for s in range(1, N):
        l = m + s
        h2 = ((m + N) ** 2 - l ** 2) * (l ** 2 - m ** 2) / (4 * l ** 2 - 1)
        P0, P1 = P1, sp.expand(x * P1) - h2 * P0
    target = sp.prod([x - (N - 1 - 2 * i) for i in range(N)])
    diff = sp.simplify(sp.together(P1 - target))
    if diff != 0:
        ok = False; out.append(f"H2 FAIL N={N}: {diff}")
out.append(f"H2 (char. polynomial identically prod(x-(N-1-2i)) in m, N=1..9): {'PASS' if ok else 'FAIL'}")

# ---- F
X, B2, b, u, t, y, z = sp.symbols('X B2 b u t y z', nonnegative=True)
# case 1 (M = B2 >= beta = b): kappa = B2+1, A2 = X - B2, with b = B2 - y (0<=y<=B2) and X = 2*B2 + 2 + t (2 j3 = X-2B2-1 >= 1 -> t >= 0)
B2s = sp.symbols('B2s', nonnegative=True)
expr1 = (((B2s + 1) ** 2 - (B2s - y) ** 2) * ((X - B2s) ** 2 - (B2s + 1) ** 2) - (X ** 2 - 1))
expr1 = sp.expand(expr1.subs(X, 2 * B2s + 2 + t))
# y in [0,B2s]: write B2s = y + z
expr1 = sp.Poly(sp.expand(expr1.subs(B2s, y + z)), y, z, t)
neg1 = [c for c in expr1.coeffs() if c < 0]
out.append(f"F case 1: f(kappa)-(X^2-1) in (y,z,t)>=0 has {len(expr1.coeffs())} terms, negative coefficients: {len(neg1)}")
# case 2 (M = B1 > beta = B2): kappa = B1+1 = B2+1+v (v>0), A2 = X - B2, 2j3 = X-B2-1 >= kappa -> X = B2 + 1 + kappa + t
v = sp.symbols('v', positive=True)
kap = B2s + 1 + v
expr2 = ((kap ** 2 - B2s ** 2) * ((X - B2s) ** 2 - kap ** 2) - (X ** 2 - 1))
expr2 = sp.Poly(sp.expand(expr2.subs(X, B2s + 1 + kap + t)), B2s, v, t)
neg2 = [(mm, c) for mm, c in zip(expr2.monoms(), expr2.coeffs()) if c < 0]
out.append(f"F case 2: terms {len(expr2.coeffs())}, negative coefficients: {len(neg2)} {neg2[:6]}")

# ---- C2
K, M, xx = sp.symbols('K M xx', nonnegative=True)
e = sp.expand((M + 1) * (2 * K + M + 1) - (M + 2) ** 2)
out.append(f"C2: (M+1)(2K+M+1)-(M+2)^2 with K = M+2+xx: {sp.expand(e.subs(K, M + 2 + xx))}")

# ---- C3
K, M, bb, E = sp.symbols('K M bb E', real=True)
p = (2 * K - 1) * ((K - 1) ** 2 - (M + 2) ** 2 - E)
q = (K ** 2 - (M + 2) ** 2) * E
T = (K ** 2 - bb ** 2) * (E + (M + 2) ** 2 - bb ** 2)
F = sp.expand(p * q - T * (p - q))
X1, Y1, Z1, W1 = sp.symbols('X1 Y1 Z1 W1', nonnegative=True)
sub = {bb: Y1, M: Y1 + Z1, K: Y1 + Z1 + 3 + X1}
Emin = sp.expand((2 * K * (M + 1) - 2 * M - 3 + bb ** 2).subs(sub))
G = sp.Poly(sp.expand(F.subs(sub).subs(E, Emin + W1)), X1, Y1, Z1, W1)
negC3 = [c for c in G.coeffs() if c < 0]
out.append(f"C3: pq - T(p-q) in (x,y,z,w) has {len(G.coeffs())} terms, negative coefficients: {len(negC3)}")
open("../outputs/vs_symbolic_poly_C3.txt", "w").write(str(G.as_expr()) + "\n")

print("\n".join(out))
open("../outputs/vs_symbolic_output.txt", "w").write("\n".join(out) + "\n")

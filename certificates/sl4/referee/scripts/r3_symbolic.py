"""Referee check 3: independent symbolic certificates (sympy), written without the author's scripts.

Spin encoding (surjective onto the REAL sorted relaxation of the admissible domain):
  j1 = 1/2 + a, j2 = j1 + b, j3 = j2 + c, j4 = j3 + e, a,b,c,e >= 0;  B1 = b, B2 = e, K = j1 + j2 = 1 + 2a + b,
  A2 = j3 + j4 + 1, X = 2 j4 + 1.  Case "B2 >= B1": e = b + g; case "B1 > B2": b = e + g (g >= 0).
  The constraint d >= 2 (resp. d >= 4) is K >= M + 1 (resp. K >= M + 3), encoded by a further substitution.
F : f(kappa) - (X^2 - 1)            (d >= 2)
C0: f(M+2) - f(M+1)                 (d >= 3)
C2: f(K) - f(M+2)                   (d >= 4, i.e. L >= 3)
C3: f(K)^2 - f(K-1) f(M+2)          (d >= 4)
Each is expanded as a polynomial in the free nonnegative variables and its coefficients are inspected.
Also: (i) the author's p, q, T, E identities and E - E_min = A2^2 - (K+M+1)^2;
(ii) Lemma H: det(x - H(m)) == prod (x - (N-1-2i)) identically in symbolic m for N = 1..16;
(iii) Lemma H matrix elements of J1z - J2z on j (x) j in the coupled basis, numerically for 2j <= 16, all m.
"""
import sympy as sp
import numpy as np
import time

out = []
def say(s):
    print(s, flush=True); out.append(s)

a, b, c, e, g, h = sp.symbols('a b c e g h', nonnegative=True)
H = sp.Rational(1, 2)

def setup(case):
    j1 = H + a
    if case == "B2>=B1":
        j2 = j1 + b; j3 = j2 + c; j4 = j3 + b + g
        M, beta = b + g, b
    else:
        j2 = j1 + e + g; j3 = j2 + c; j4 = j3 + e
        M, beta = e + g, e
    K = j1 + j2; A2 = j3 + j4 + 1; X = 2 * j4 + 1
    f = lambda k: (k ** 2 - beta ** 2) * (A2 ** 2 - k ** 2)
    return j1, j2, j3, j4, M, beta, K, A2, X, f

def certify(name, expr, vars_):
    P = sp.Poly(sp.expand(expr), *vars_)
    neg = [(m_, c_) for m_, c_ in zip(P.monoms(), P.coeffs()) if c_ < 0]
    say(f"{name}: {len(P.coeffs())} terms in {vars_}, negative coefficients {len(neg)} {neg[:4]}")
    return len(neg) == 0

t0 = time.time()
for case in ["B2>=B1", "B1>B2"]:
    j1, j2, j3, j4, M, beta, K, A2, X, f = setup(case)
    # d >= 2: K >= M+1.  K - M - 1 = 2a + b - M.  Impose 2a + b - M = h >= 0 by solving for a.
    for dmin, shift in [(2, 1), (3, 2), (4, 3)]:
        sol = sp.solve(sp.Eq(K - M - shift, h), a)[0]
        sub = {a: sol}
        free = [b, c, g, h] if case == "B2>=B1" else [c, e, g, h]
        if dmin == 2:
            certify(f"F  [{case}] f(kappa)-(X^2-1)", (f(M + 1) - (X ** 2 - 1)).subs(sub), free)
        if dmin == 3:
            certify(f"C0 [{case}] f(M+2)-f(M+1)", (f(M + 2) - f(M + 1)).subs(sub), free)
        if dmin == 4:
            certify(f"C2 [{case}] f(K)-f(M+2)", (f(K) - f(M + 2)).subs(sub), free)
            certify(f"C3 [{case}] f(K)^2-f(K-1)f(M+2)", (f(K) ** 2 - f(K - 1) * f(M + 2)).subs(sub), free)
say(f"(spin-encoded certificates: {time.time()-t0:.0f} s)")

# Lemma F equality locus in the spin encoding: f(kappa) = X^2-1 forces b=e=0 (i.e. B1=B2=0)
for case in ["B2>=B1", "B1>B2"]:
    j1, j2, j3, j4, M, beta, K, A2, X, f = setup(case)
    sol = sp.solve(sp.Eq(K - M - 1, h), a)[0]
    P = sp.Poly(sp.expand((f(M + 1) - (X ** 2 - 1)).subs({a: sol})), *sorted((sp.expand(f(M + 1) - (X ** 2 - 1)).subs({a: sol})).free_symbols, key=str))
    say(f"F [{case}] monomials with positive coefficient: {P.monoms()[:40]}{' ...' if len(P.monoms())>40 else ''}")

# (i) author's algebra for C3
K_, M_, be, E = sp.symbols('K M beta E', real=True)
A2sq = E + K_ ** 2 + (M_ + 2) ** 2 - be ** 2
ff = lambda k: (k ** 2 - be ** 2) * (A2sq - k ** 2)
p = (2 * K_ - 1) * ((K_ - 1) ** 2 - (M_ + 2) ** 2 - E)
q = (K_ ** 2 - (M_ + 2) ** 2) * E
T = (K_ ** 2 - be ** 2) * (E + (M_ + 2) ** 2 - be ** 2)
say("author's p == f(K-1)-f(K): " + str(sp.expand(ff(K_ - 1) - ff(K_) - p) == 0))
say("author's q == f(K)-f(M+2): " + str(sp.expand(ff(K_) - ff(M_ + 2) - q) == 0))
say("author's T == f(K): " + str(sp.expand(ff(K_) - T) == 0))
say("f(K)^2 - f(K-1) f(M+2) == pq - T(p-q): " + str(sp.expand(ff(K_) ** 2 - ff(K_ - 1) * ff(M_ + 2) - (p * q - T * (p - q))) == 0))
Emin = 2 * K_ * (M_ + 1) - 2 * M_ - 3 + be ** 2
say("E - E_min == A2^2 - (K+M+1)^2: " + str(sp.expand((E - Emin) - (A2sq - (K_ + M_ + 1) ** 2)) == 0))

# (ii) Lemma H symbolic in m
m, x = sp.symbols('m x')
t0 = time.time(); okH = True
for N in range(1, 17):
    P0, P1 = sp.Integer(1), x
    for s in range(1, N):
        l = m + s
        h2 = sp.factor(((m + N) ** 2 - l ** 2) * (l ** 2 - m ** 2) / (4 * l ** 2 - 1))
        P0, P1 = P1, sp.cancel(x * P1 - h2 * P0)
    target = sp.expand(sp.prod([x - (N - 1 - 2 * i) for i in range(N)]))
    if sp.cancel(P1 - target) != 0:
        okH = False; say(f"Lemma H symbolic FAIL N={N}")
say(f"(ii) Lemma H: characteristic polynomial == prod(x-(N-1-2i)) identically in m, N=1..16: {okH} ({time.time()-t0:.0f} s)")

# (iii) matrix elements of J1z-J2z on j (x) j, numerically
def spin(tj):
    jj = tj / 2; mm = np.arange(jj, -jj - 1, -1)
    jp = np.zeros((tj + 1, tj + 1))
    for i in range(1, tj + 1):
        jp[i - 1, i] = np.sqrt(jj * (jj + 1) - mm[i] * (mm[i] + 1))
    return (jp + jp.T) / 2, (jp - jp.T) / 2j, np.diag(mm)
worst = 0.0
for tj in range(1, 17):
    jx, jy, jz = spin(tj); I = np.eye(tj + 1)
    Jt = [np.kron(o, I) + np.kron(I, o) for o in (jx, jy, jz)]
    D = np.kron(jz, I) - np.kron(I, jz)
    C = sum(o @ o for o in Jt).real
    mz = np.diag(Jt[2]).real
    for M2 in range(0, tj + 1):   # sector J_z = M2 (integer since j(x)j)
        idx = np.where(np.abs(mz - M2) < 1e-9)[0]
        w, V = np.linalg.eigh(C[np.ix_(idx, idx)])
        Lv = (-1 + np.sqrt(1 + 4 * w)) / 2
        o = np.argsort(Lv); Lv = np.round(Lv[o]).astype(int); V = V[:, o]
        Dm = V.T @ D.real[np.ix_(idx, idx)] @ V
        n = tj + 1
        pred = np.zeros_like(Dm)
        for i in range(1, len(Lv)):
            L_ = Lv[i]
            pred[i, i - 1] = pred[i - 1, i] = np.sqrt((L_ ** 2 - M2 ** 2) * (n ** 2 - L_ ** 2) / (4 * L_ ** 2 - 1))
        worst = max(worst, float(np.max(np.abs(np.abs(Dm) - pred))))
        ev = np.sort(np.linalg.eigvalsh(pred))
        N_ = n - M2
        worst = max(worst, float(np.max(np.abs(ev - np.arange(-(N_ - 1), N_, 2)))))
say(f"(iii) |<L m|J1z-J2z|L' m>| vs Stark formula and its spectrum, 2j<=16, all m>=0: max error {worst:.2e}")

# (iv) Lemma H at real (non-integer, incl. half-integer) m in 50-digit arithmetic, N up to 60
import mpmath as mp
mp.mp.dps = 50
worst = mp.mpf(0)
for N in [2, 3, 7, 20, 41, 60]:
    for mv in [mp.mpf('0.5'), mp.mpf('3.5'), mp.mpf('17.5'), mp.mpf('0.3141'), mp.mpf('12.77')]:
        A = mp.zeros(N, N)
        for s in range(1, N):
            l = mv + s; n_ = mv + N
            A[s - 1, s] = A[s, s - 1] = mp.sqrt((n_ ** 2 - l ** 2) * (l ** 2 - mv ** 2) / (4 * l ** 2 - 1))
        ev = sorted(mp.eigsy(A, eigvals_only=True))
        worst = max(worst, max(abs(ev[i] - (-(N - 1) + 2 * i)) for i in range(N)))
say(f"(iv) Lemma H at real m (0.5, 3.5, 17.5, 0.3141, 12.77), N in (2,3,7,20,41,60), 50 digits: max |ev - integer| = {mp.nstr(worst, 5)}")
open("../outputs/r3_symbolic.txt", "w").write("\n".join(out) + "\n")

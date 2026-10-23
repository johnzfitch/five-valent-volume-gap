"""Exact characteristic polynomial of U_z = (J2xJ3 + J2xJ4 + J3xJ4)_z on (1/2 x 1 x 1)_{M=1/2} (sympy),
and the exact sharp slope 8*nu0."""
import sympy as sp


def spin(j):
    j = sp.Rational(j); ms = [j - k for k in range(int(2 * j) + 1)]
    n = len(ms); Jp = sp.zeros(n)
    for i in range(1, n):
        m = ms[i]; Jp[i - 1, i] = sp.sqrt(j * (j + 1) - m * (m + 1))
    return Jp, Jp.T, sp.diag(*ms)


def kron(*Ms):
    out = Ms[0]
    for M in Ms[1:]:
        out = sp.kronecker_product(out, M)
    return out


js = [sp.Rational(1, 2), 1, 1]
ops = [spin(j) for j in js]; I = [sp.eye(int(2 * j) + 1) for j in js]
def emb(k, A):
    m = list(I); m[k] = A; return kron(*m)
Jp = [emb(k, ops[k][0]) for k in range(3)]; Jm = [emb(k, ops[k][1]) for k in range(3)]; Jz = [emb(k, ops[k][2]) for k in range(3)]
U = sp.zeros(Jp[0].shape[0])
for a, b in ((0, 1), (0, 2), (1, 2)):
    U += sp.I / 2 * (Jp[a] * Jm[b] - Jm[a] * Jp[b])
Mz = sum(Jz, sp.zeros(U.shape[0]))
idx = [i for i in range(U.shape[0]) if Mz[i, i] == sp.Rational(1, 2)]
B = sp.simplify(U.extract(idx, idx))
x = sp.symbols('x')
p = sp.factor(sp.expand((B - x * sp.eye(len(idx))).det()))
print("dim", len(idx)); print("char poly:", p)
roots = sp.solve(sp.expand(p / x), x)
r = sorted([sp.nsimplify(rr) for rr in roots], key=lambda z: abs(complex(z)))
print("nonzero roots:", r, [sp.N(rr, 20) for rr in r])
nu0 = min(abs(sp.N(rr, 30)) for rr in r)
print("nu0 =", nu0, " 8 nu0 =", 8 * nu0)

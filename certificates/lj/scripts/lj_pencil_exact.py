"""Small exact (sympy) check of the three-spin pencil identity: det(x - T) == det(x B + A)/det(B)
for a few tuples where every g-sector is live (dim V[M] = dim Inv).  T is built from the exact link
formulas of [DC] Section 1 in the nonnegative gauge."""
import sympy as sp
from math import factorial


def spin(j):
    ms = [j - k for k in range(int(2 * j) + 1)]; n = len(ms); Jp = sp.zeros(n)
    for i in range(1, n):
        m = ms[i]; Jp[i - 1, i] = sp.sqrt(j * (j + 1) - m * (m + 1))
    return Jp, Jp.T, sp.diag(*ms)


def three(js):
    ops = [spin(j) for j in js]; I = [sp.eye(int(2 * j) + 1) for j in js]
    def emb(k, A):
        m = list(I); m[k] = A
        return sp.kronecker_product(sp.kronecker_product(m[0], m[1]), m[2])
    Jp = [emb(k, ops[k][0]) for k in range(3)]; Jm = [emb(k, ops[k][1]) for k in range(3)]; Jz = [emb(k, ops[k][2]) for k in range(3)]
    Jx = [(Jp[k] + Jm[k]) / 2 for k in range(3)]; Jy = [(Jp[k] - Jm[k]) / (2 * sp.I) for k in range(3)]
    pr = ((0, 1), (0, 2), (1, 2))
    Ux = sum((Jy[a] * Jz[b] - Jz[a] * Jy[b] for a, b in pr), sp.zeros(Jp[0].shape[0]))
    Uy = sum((Jz[a] * Jx[b] - Jx[a] * Jz[b] for a, b in pr), sp.zeros(Jp[0].shape[0]))
    Uz = sum((Jx[a] * Jy[b] - Jy[a] * Jx[b] for a, b in pr), sp.zeros(Jp[0].shape[0]))
    return Ux + sp.I * Uy, Ux - sp.I * Uy, Uz, sum(Jp, sp.zeros(Jp[0].shape[0])), sum(Jm, sp.zeros(Jp[0].shape[0])), sum(Jz, sp.zeros(Jp[0].shape[0]))


def pencil(js, s, j5):
    Up, Um, Uz, Gp, Gm, Gz = three(js); n = Gp.shape[0]
    B = sp.zeros(n); Bp = sp.zeros(n); P = sp.eye(n)
    for k in range(int(2 * sum(js)) + 2):
        B += sp.Rational((-1) ** k) / (factorial(k) * sp.rf(2 * s + 2, k)) * P
        Bp += sp.Rational((-1) ** k) / (factorial(k) * sp.rf(2 * s + 3, k)) * P
        P = Gm * P * Gp
    X = s * Uz + (Up * Gm + Um * Gp) / 2
    A = X.H * B - j5 / (2 * s + 2) * Um * Bp * Gp
    M = s - j5
    idx = [i for i in range(n) if Gz[i, i] == M]
    return A.extract(idx, idx), B.extract(idx, idx)


def T_exact(s, j2, j3, j4, j5):
    n1, m1 = j2 + j3 + 1, abs(j2 - j3); n2, m2 = j4 + j5 + 1, abs(j4 - j5)
    S = []
    b = m1
    while b <= n1 - 1:
        c = m2
        while c <= n2 - 1:
            if abs(b - c) <= s <= b + c: S.append((b, c))
            c += 1
        b += 1
    idx = {p: i for i, p in enumerate(S)}; T = sp.zeros(len(S))
    F = lambda t: (s + sp.Rational(1, 2)) ** 2 - t ** 2
    for (b, c), i in idx.items():
        u = b + c; d = b - c; G = (u + s + 1) * (u - s)
        if (b - 1, c) in idx:
            w2 = (n1 ** 2 - b ** 2) * (b ** 2 - m1 ** 2) / (4 * b ** 2 - 1) * F(d - sp.Rational(1, 2)) * G / 16
            T[idx[(b - 1, c)], i] = T[i, idx[(b - 1, c)]] = sp.sqrt(w2)
        if (b, c - 1) in idx:
            w2 = (n2 ** 2 - c ** 2) * (c ** 2 - m2 ** 2) / (4 * c ** 2 - 1) * F(d + sp.Rational(1, 2)) * G / 16
            T[idx[(b, c - 1)], i] = T[i, idx[(b, c - 1)]] = sp.sqrt(w2)
    return T


h = sp.Rational(1, 2); x = sp.symbols('x')
for tup in [(3 * h, h, 1, 1, 1), (2, h, 1, 1, 3 * h), (5 * h, h, 1, 1, 2), (2, 1, 1, 1, 1), (3 * h, 1, 1, 1, 3 * h)]:
    s, j2, j3, j4, j5 = [sp.Rational(v) for v in tup]
    T = T_exact(s, j2, j3, j4, j5)
    A, B = pencil([j2, j3, j4], s, j5)
    if A.shape != T.shape:
        print(tup, "dim V[M] != dim Inv, skipped"); continue
    p1 = sp.expand(sp.simplify((x * sp.eye(T.shape[0]) - T).det()))
    p2 = sp.expand(sp.simplify((x * B + A).det() / B.det()))
    print(tup, "dim", T.shape[0], "charpoly equal:", sp.simplify(p1 - p2) == 0, "  ", sp.factor(p1))

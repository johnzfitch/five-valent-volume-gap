"""Checks of [LJ] Theorem B (sharp slope 4 sqrt(10 - 6 sqrt 2)).

(a) U_z = (J2xJ3 + J2xJ4 + J3xJ4)_z on (1/2 x 1 x 1) restricted to weight M = 1/2, from explicit spin matrices (sympy, exact):
    characteristic polynomial and nu0.
(b) The closed form x_-^2 = 5s^2/2 + 3s/2 - 9/4 - (3s+4) sqrt(8s^2 - 12s + 5)/4 against q_min of the Racah operator rb_core
    at (s; 1/2, 1, 1, s-1/2), 2s = 3..MAX (float), and the M = -1/2 family (s; 1/2,1,1,s+1/2) against its polynomial.
(c) Exact characteristic polynomial of T at (s; 1/2,1,1,s-1/2) from the exact Racah entries (sympy, 5x5), for 2s = 3..41,
    compared with x(16x^4 - (80s^2+48s-72)x^2 + 28s^4+36s^3-29s^2-36s+1)/16 (the [LJ] polynomial written out).
(d) A symbolic-s proof of the same polynomial from the closed-form links (validated exactly by rb_links) on the fixed
    5-site structure, valid for all real s >= 3/2.
Usage: python rb_thmB.py MAX2S > ../outputs/rb_thmB_out.txt
"""
import sys, os, time
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C

R = sp.Rational


def spin_mats(j):
    d = int(2 * j + 1)
    ms = [j - k for k in range(d)]
    Jp = sp.zeros(d, d)
    for k in range(1, d):
        m = ms[k]
        Jp[k - 1, k] = sp.sqrt(j * (j + 1) - m * (m + 1))
    Jm = Jp.T
    return (Jp + Jm) / 2, (Jp - Jm) / (2 * sp.I), sp.diag(*ms)


def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = sp.kronecker_product(out, m)
    return out


def part_a():
    js = [R(1, 2), 1, 1]
    mats = [spin_mats(j) for j in js]
    I = [sp.eye(int(2 * j + 1)) for j in js]

    def leg(k, a):
        f = [I[0], I[1], I[2]]; f[k] = mats[k][a]
        return kron(*f)
    J = [[leg(k, a) for a in range(3)] for k in range(3)]
    # (A x B)_z = A_x B_y - A_y B_x
    Uz = sum((J[a][0] * J[b][1] - J[a][1] * J[b][0] for (a, b) in ((0, 1), (0, 2), (1, 2))), sp.zeros(18, 18))
    Gz = J[0][2] + J[1][2] + J[2][2]
    idx = [i for i in range(18) if Gz[i, i] == R(1, 2)]
    U = Uz.extract(idx, idx)
    x = sp.Symbol('x')
    cp = sp.factor(U.charpoly(x).as_expr())
    nu0 = sp.sqrt((5 - 3 * sp.sqrt(2)) / 2)
    ok = sp.simplify(cp - sp.expand(x * (4 * x ** 4 - 20 * x ** 2 + 7) / 4)) == 0
    print(f"(a) dim (1/2 x 1 x 1)_(M=1/2) = {len(idx)};  charpoly(U_z) = {cp};  equals x(4x^4-20x^2+7)/4: {ok}")
    print(f"    roots x^2 = (5 +- 3 sqrt2)/2;  nu0 = {sp.N(nu0, 15)};  8 nu0 = {sp.N(8*nu0, 15)};  "
          f"4 sqrt(10 - 6 sqrt2) = {sp.N(4*sp.sqrt(10-6*sp.sqrt(2)), 15)};  equal: {sp.simplify(8*nu0 - 4*sp.sqrt(10-6*sp.sqrt(2))) == 0}")
    # the [LJ] text writes -x(4x^4 - 20x^2 + 7)/4 ; sign convention of charpoly only


def part_b(MAX):
    worst = 0.0; worst2 = 0.0
    for s2 in range(3, MAX + 1):
        s = s2 / 2
        q, k = C.qmin((s2, 1, 2, 2, s2 - 1))
        xm2 = 5 * s * s / 2 + 3 * s / 2 - 9 / 4 - (3 * s + 4) * np.sqrt(8 * s * s - 12 * s + 5) / 4
        worst = max(worst, abs(q - 8 * np.sqrt(xm2)) / q)
        # M = -1/2 family: smallest root of 16y^2 - (80s^2+112s-40) y + 28s^4+76s^3+31s^2-18s, y = x^2
        a, b, c = 16, -(80 * s * s + 112 * s - 40), 28 * s ** 4 + 76 * s ** 3 + 31 * s * s - 18 * s
        y = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
        q2, k2 = C.qmin((s2, 1, 2, 2, s2 + 1))
        worst2 = max(worst2, abs(q2 - 8 * np.sqrt(y)) / q2)
    print(f"(b) closed form vs Racah q_min on (s;1/2,1,1,s-1/2), 2s=3..{MAX}: max rel diff {worst:.2e};  "
          f"M=-1/2 family (s;1/2,1,1,s+1/2): max rel diff {worst2:.2e}")
    for s2 in (3, 4, 20, 200, MAX):
        s = s2 / 2
        q, _ = C.qmin((s2, 1, 2, 2, s2 - 1))
        print(f"    2s={s2}: q_min = {q:.9f},  q/s = {q/s:.7f}  (8 nu0 = 4.9229562...)")


def exact_charpoly(tj):
    S, ent = C.R_entries(tj)
    n = len(S)
    sc = C.scale_sq(tj[0])
    Tm = sp.zeros(n, n)
    for (i, j), (sg, sq) in ent.items():
        v = sg * sp.sqrt(sp.Rational(sq.numerator, sq.denominator) * sp.Rational(sc.numerator, sc.denominator))
        # T = (C0/2i) R ;  |entry| = sqrt(sc * sq);  phase: T[i,j] = -i * sign(C0) * R[i,j]*|C0|/2 ; use Hermitian i*antisym
        Tm[i, j] = sp.I * v; Tm[j, i] = -sp.I * v
    x = sp.Symbol('x')
    return sp.expand(Tm.charpoly(x).as_expr()), n


def part_c(MAXc):
    x = sp.Symbol('x')
    bad = 0
    for s2 in range(3, MAXc + 1):
        s = R(s2, 2)
        cp, n = exact_charpoly((s2, 1, 2, 2, s2 - 1))
        ref = sp.expand(x * (16 * x ** 4 - (80 * s ** 2 + 48 * s - 72) * x ** 2 + 28 * s ** 4 + 36 * s ** 3 - 29 * s ** 2 - 36 * s + 1) / 16)
        if n != 5 or sp.simplify(cp - ref) != 0:
            bad += 1; print('   mismatch at 2s =', s2, n, cp)
    print(f"(c) exact charpoly from Racah entries equals the [LJ] polynomial for 2s = 3..{MAXc}: {MAXc-2-bad}/{MAXc-2}")


def part_d():
    t, x = sp.symbols('t x', nonnegative=True)
    s = t + R(3, 2)                      # s >= 3/2
    # pairing (23|45) with (j2,j3)=(1/2,1), (j4,j5)=(1,s-1/2): n1=5/2,m1=1/2, n2=s+3/2, m2=s-3/2
    n1, m1, n2, m2 = R(5, 2), R(1, 2), s + R(3, 2), s - R(3, 2)
    sites = [(R(1, 2), s - R(1, 2)), (R(1, 2), s + R(1, 2)), (R(3, 2), s - R(3, 2)), (R(3, 2), s - R(1, 2)), (R(3, 2), s + R(1, 2))]
    idx = {p: i for i, p in enumerate(sites)}
    w = {}
    for (b, c), i in idx.items():
        if (b - 1, c) in idx:
            w[(idx[(b - 1, c)], i)] = sp.factor((n1 ** 2 - b ** 2) * (b ** 2 - m1 ** 2) * ((s + c + 1) ** 2 - b ** 2) * (b ** 2 - (s - c) ** 2) / (16 * (4 * b ** 2 - 1)))
        if (b, c - 1) in idx:
            w[(idx[(b, c - 1)], i)] = sp.factor((n2 ** 2 - c ** 2) * (c ** 2 - m2 ** 2) * ((s + b + 1) ** 2 - c ** 2) * (c ** 2 - (s - b) ** 2) / (16 * (4 * c ** 2 - 1)))
    # char poly of a bipartite graph: x^n - x^(n-2) sum w^2 + x^(n-4) [sum over disjoint edge pairs w^2 w'^2 - 2 sum over 4-cycles prod w] - ...
    # here n = 5 with classes of sizes 2 and 3, so only matchings of size <= 2 and 4-cycles contribute.
    E = list(w)
    s1 = sum(w.values())
    s2 = 0
    for i in range(len(E)):
        for k in range(i + 1, len(E)):
            if not set(E[i]) & set(E[k]):
                s2 += w[E[i]] * w[E[k]]
    # 4-cycles: pairs of rows sharing two columns
    adj = {}
    for (a, b) in E:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    cyc = 0; ncyc = 0; perfect = True
    verts = range(5)
    seen = set()
    for a in verts:
        for b in verts:
            if a < b and len(adj.get(a, set()) & adj.get(b, set())) >= 2:
                cs = sorted(adj[a] & adj[b])
                for i in range(len(cs)):
                    for k in range(i + 1, len(cs)):
                        key = frozenset((a, b, cs[i], cs[k]))
                        if key in seen: continue
                        seen.add(key)
                        g = lambda u, v: w[(u, v)] if (u, v) in w else w[(v, u)]
                        prod2 = sp.factor(g(a, cs[i]) * g(cs[i], b) * g(b, cs[k]) * g(cs[k], a))
                        r = sp.sqrt(prod2)
                        r = sp.factor(sp.powsimp(r, force=True))
                        ok = sp.simplify(r ** 2 - prod2) == 0 and not r.has(sp.sqrt) and all(not isinstance(f, sp.Pow) or f.exp.is_integer for f in sp.Mul.make_args(r))
                        perfect &= bool(ok)
                        cyc += r; ncyc += 1
    cp = sp.expand(x ** 5 - s1 * x ** 3 + (s2 - 2 * cyc) * x)
    tt = sp.Symbol('t')
    ss = tt + R(3, 2)
    ref = sp.expand((x * (16 * x ** 4 - (80 * ss ** 2 + 48 * ss - 72) * x ** 2 + 28 * ss ** 4 + 36 * ss ** 3 - 29 * ss ** 2 - 36 * ss + 1) / 16).subs(tt, t))
    diff = sp.simplify(sp.expand(cp - ref))
    print(f"(d) symbolic s = 3/2 + t (t >= 0): {len(E)} links, {ncyc} four-cycle(s), cycle products are perfect squares of polynomials: {perfect}")
    print(f"    charpoly = {sp.factor(cp.subs(t, sp.Symbol('s') - R(3, 2)))}")
    print(f"    equals the [LJ] polynomial identically: {diff == 0}")
    S = sp.Symbol('s', positive=True)
    y = sp.Symbol('y')
    roots = sp.solve(16 * y ** 2 - (80 * S ** 2 + 48 * S - 72) * y + 28 * S ** 4 + 36 * S ** 3 - 29 * S ** 2 - 36 * S + 1, y)
    target = R(5, 2) * S ** 2 + R(3, 2) * S - R(9, 4) - (3 * S + 4) * sp.sqrt(8 * S ** 2 - 12 * S + 5) / 4
    print(f"    roots y = x^2: {roots};  [LJ] x_-^2 among them: {any(sp.simplify(r - target) == 0 for r in roots)}")
    lim = sp.limit(sp.sqrt(target) / S, S, sp.oo)
    print(f"    lim x_-/s = {sp.nsimplify(lim)} = {sp.N(lim, 15)};  8*lim = {sp.N(8*lim, 15)}")


if __name__ == '__main__':
    MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 401
    t0 = time.time()
    part_a(); part_b(MAX); part_c(41); part_d()
    print(f"total {time.time()-t0:.0f}s")

"""Exact spectrum of T along (j1; 1/2, 1, 1, j1 -/+ 1/2) for symbolic j1 = s, from the three-spin pencil
(t = (1/2,1,1), M = +-1/2).  Prints the characteristic polynomial and the asymptotics of the small root."""
import sympy as sp, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lj_pencil_exact.py')).read().split('h = sp.Rational')[0]
exec(src)
s, x = sp.symbols('s x', positive=True)
h = sp.Rational(1, 2)
for M in (h, -h):
    j5 = s - M
    A, B = pencil([h, sp.Integer(1), sp.Integer(1)], s, j5)
    A = A.applyfunc(sp.simplify); B = B.applyfunc(sp.simplify)
    p = sp.factor(sp.simplify((x * B + A).det() / B.det()))
    print(f"M = {M}:  det(x - T) = {p}")
    y = sp.symbols('y')
    quart = sp.factor(sp.simplify(p / x)).subs(x ** 2, y)
    sols = sp.solve(sp.numer(sp.together(sp.simplify(p / x))).subs(x, sp.sqrt(y)), y)
    for so in sols:
        so = sp.simplify(so)
        print("   x^2 root:", so, "   series at s=oo:", sp.series(sp.sqrt(so), s, sp.oo, 2))
    for sv in (sp.Rational(3, 2), 2, 10, 100):
        vals = [sp.N(8 * sp.sqrt(so.subs(s, sv))) for so in sols]
        print("   s =", sv, " q values (8|x|):", vals)

"""N = 3, all real m >= 0: B = [[g1, -1], [0, g2]], g1^2 = 4(m+2)/(2m+3), g2^2 = 4(m+1)/(2m+3) (exact, sympy);
sigma_min^2 = (5 - sqrt(9 + 16/q^2))/2 with q = 2m+3, increasing in m, equal to (15 - sqrt 97)/6 at m = 0."""
import sympy as sp
m = sp.symbols('m', nonnegative=True)
n = m + 3
g2 = lambda L: sp.simplify((n**2 - L**2) * (L**2 - m**2) / (4*L**2 - 1))
G1, G2 = sp.factor(g2(m + 1)), sp.factor(g2(m + 2))
x = sp.symbols('x')
B = sp.Matrix([[sp.sqrt(G1), -1], [0, sp.sqrt(G2)]])
cp = sp.expand(sp.simplify((B.T * B - x * sp.eye(2)).det()))
q = 2*m + 3
smin2 = (5 - sp.sqrt(9 + 16 / q**2)) / 2
lines = [f"g1^2 = {G1}, g2^2 = {G2}",
         f"char poly of B^T B: {sp.factor(cp)}",
         f"trace = {sp.simplify(G1 + G2 + 1)}, det = {sp.simplify(G1 * G2)}",
         f"smin^2 formula satisfies char poly: {sp.simplify(cp.subs(x, smin2)) == 0}",
         f"value at m=0: {sp.nsimplify(smin2.subs(m, 0))} = (15-sqrt(97))/6: {sp.simplify(smin2.subs(m, 0) - (15 - sp.sqrt(97))/6) == 0}",
         f"d/dm smin^2 = {sp.simplify(sp.diff(smin2, m))} (> 0 for m >= 0)"]
print("\n".join(lines)); open("../outputs/n3_check_out.txt", "w").write("\n".join(lines) + "\n")

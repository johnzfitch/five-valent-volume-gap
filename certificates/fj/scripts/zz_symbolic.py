"""General zigzag-chain formulas (any j1 = s, any column diagonal d), symbolic, for shift certificates.

Column site on diagonal d: (beta, beta - d).  Owner rows BELOW (diagonal d-1): the row between columns beta and beta+1 is
(beta, beta-d+1) with links
  to column beta   : q^2 = h2(beta-d+1)^2 F(d-1/2) G(2beta-d+1) / 16      (D-link)
  to column beta+1 : p^2 = h1(beta+1)^2   F(d-1/2) G(2beta-d+2) / 16      (X-link)
Rows ABOVE (diagonal d+1): the row between beta and beta+1 is (beta+1, beta-d) with
  to column beta   : q'^2 = h1(beta+1)^2   F(d+1/2) G(2beta-d+1) / 16
  to column beta+1 : p'^2 = h2(beta-d+1)^2 F(d+1/2) G(2beta-d+2) / 16
Lambda (coupled = above, owner = below, kappa = F(d+1/2)/F(d-1/2)) at column beta:
  F(d+1/2)/16 * [ G(u+1)(h1(b+1)^2-h2(c+1)^2) - G(u)(h1(b)^2-h2(c)^2) ],  (b,c) = (beta, beta-d), u = 2beta-d.
h_P(L)^2 = P_P(L)/((2L-1)(2L+1)).  All functions are returned as sympy expressions in beta, n1, m1, n2, m2.
"""
import sympy as sp

beta, n1, m1, n2, m2 = sp.symbols('beta n1 m1 n2 m2')


def P1(L): return (n1**2 - L**2) * (L**2 - m1**2)
def P2(L): return (n2**2 - L**2) * (L**2 - m2**2)
def h1sq(L): return P1(L) / ((2 * L - 1) * (2 * L + 1))
def h2sq(L): return P2(L) / ((2 * L - 1) * (2 * L + 1))


def FG(s):
    F = lambda t: (s + sp.Rational(1, 2))**2 - t**2
    G = lambda u: (u + s + 1) * (u - s)
    return F, G


def links_below(s, d):
    F, G = FG(s)
    q2 = h2sq(beta - d + 1) * F(d - sp.Rational(1, 2)) * G(2 * beta - d + 1) / 16
    p2 = h1sq(beta + 1) * F(d - sp.Rational(1, 2)) * G(2 * beta - d + 2) / 16
    return sp.factor(q2), sp.factor(p2)


def links_above(s, d):
    F, G = FG(s)
    q2 = h1sq(beta + 1) * F(d + sp.Rational(1, 2)) * G(2 * beta - d + 1) / 16
    p2 = h2sq(beta - d + 1) * F(d + sp.Rational(1, 2)) * G(2 * beta - d + 2) / 16
    return sp.factor(q2), sp.factor(p2)


def Lambda(s, d):
    F, G = FG(s)
    b, c, u = beta, beta - d, 2 * beta - d
    return F(d + sp.Rational(1, 2)) / 16 * (G(u + 1) * (h1sq(b + 1) - h2sq(c + 1)) - G(u) * (h1sq(b) - h2sq(c)))


def check_identity(s, d):
    """Lambda = (above diag) - kappa (below diag) at column beta, from the link formulas"""
    F, G = FG(s)
    qb, pb = links_below(s, d); qa, pa = links_above(s, d)
    kap = F(d + sp.Rational(1, 2)) / F(d - sp.Rational(1, 2))
    below = qb + pb.subs(beta, beta - 1)
    above = qa + pa.subs(beta, beta - 1)
    return sp.simplify(above - kap * below - Lambda(s, d)) == 0


if __name__ == '__main__':
    for s in (sp.Rational(3, 2), 2, sp.Rational(5, 2), 3):
        for d in [sp.Rational(k, 2) for k in range(int(-2 * s) + 1, int(2 * s), 2)]:
            print(s, d, 'identity', check_identity(s, d))

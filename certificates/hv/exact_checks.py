"""Exact (rational) checks, small explicitly scoped CPU computations.

X1. For the 5-valent family (1/2, j-1/2, j, j, j), sigma = (4,2,2,0), 2j = 2..14: the exact characteristic
    polynomial of the full recoupling-basis operator (exact_tools.rational_A) equals, coefficient by
    coefficient, the characteristic polynomial of the closed-form Jacobi chain (chain_check), and the
    kernel is trivial.
X2. Same comparison for every tuple (1/2, j2, j3, j4, j5) with 2j <= 5 and sigma in {(4,2,2,0),(2,2,2,0),(2,4,4,0)}.
X3. Non-coplanar 6-valent configuration sigma = (2,0,0,0,0,0,0,0,-2,0) (BR 0706.0382 Table 9 ordering
    123,124,125,134,135,145,234,235,245,345) with spins 2j = (1,1,n,n,n+1,n+1): exact kernel dimension and a
    certified enclosure of the smallest nonzero q^2 (numerical evidence for decay at valence 6; not a proof
    of a limit).
Usage: python exact_checks.py
"""
import itertools
from fractions import Fraction
import flint
from exact_tools import rational_A, certify
from hv_core import sigma_vec_BR
from chain_check import chain_couplings_sq

flint.ctx.prec = 256


def chain_charpoly(c2):
    """det(x - A) for real antisymmetric tridiagonal A with squared couplings c2 (Fractions)."""
    x = flint.fmpq_poly([0, 1])
    p_prev, p = flint.fmpq_poly([1]), x
    for c in c2:
        p_prev, p = p, x * p + flint.fmpq(c.numerator, c.denominator) * p_prev
    return p


def main():
    print("X1. family charpoly identity:")
    for N in range(2, 15):
        Jd = (1, N - 1, N, N, N)
        A, d = rational_A(Jd, sigma_vec_BR((4, 2, 2, 0), 5))
        p_full = A.charpoly()
        _, c2 = chain_couplings_sq(Jd, 4, 2)
        p_ch = chain_charpoly(c2)
        same = (p_full == p_ch)
        k0 = next(i for i, c in enumerate(p_full.coeffs()) if c != 0)
        print(f"   2j={N:2d}: dim {d}, exact charpoly(full) == charpoly(chain): {same}, kernel dim {k0}")
        assert same and k0 == 0
    print("X2. all tuples with 2j<=5, chain-type sigma:")
    n_ok = n_tot = 0
    for J2, J3, J4, J5 in itertools.product(range(1, 6), repeat=4):
        Jd = (1, J2, J3, J4, J5)
        if sum(Jd) % 2:
            continue
        for (al, be) in [(4, 2), (2, 2), (2, 4)]:
            try:
                S, c2 = chain_couplings_sq(Jd, al, be)
            except Exception:
                continue
            if len(S) == 0:
                continue
            A, d = rational_A(Jd, sigma_vec_BR((al, be, be, 0), 5))
            if d == 0:
                continue
            n_tot += 1
            n_ok += (A.charpoly() == chain_charpoly(c2))
    print(f"   {n_ok}/{n_tot} exact charpoly identities")
    assert n_ok == n_tot
    print("X3. 6-valent non-coplanar sigma=(2,0,0,0,0,0,0,0,-2,0), spins 2j=(1,1,n,n,n+1,n+1):")
    sig6 = sigma_vec_BR((2, 0, 0, 0, 0, 0, 0, 0, -2, 0), 6)
    for n in (1, 3, 5, 7, 9):
        r = certify((1, 1, n, n, n + 1, n + 1), sig6)
        ball = r["q2_min_ball"]
        print(f"   n={n}: dim {r['dim']}, exact kernel dim {r['kernel_dim']}, smallest nonzero q^2 in "
              f"{ball.real.str(12, radius=True)}  (q = {ball.real.sqrt().str(8)})")


if __name__ == "__main__":
    main()

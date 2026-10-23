"""The regime plan for j1 = 3/2 (used in the proof).

Sort the spins of legs 2..5: j2<=j3<=j4<=j5, gaps g1=j3-j2, g2=j4-j3, g3=j5-j4 (g1+g3 = 1/2 mod 1).
Each tuple is assigned one regime and one ORDERED pairing (pair 1 | pair 2):

  R1  : g1-g3 >= 3/2                -> (j2 j4 | j3 j5)   zz0 CR, zz1 CR
  R2- : g1-g3 = 1/2, g3 >= 1/2      -> (j2 j4 | j3 j5)   zz0 CR, zz1 RR   (m2 = m1 - 1/2)
  R1c : g1=1/2, g3=0, g2 >= 1/2      -> (j2 j3 | j4 j5)   zz0 CR, zz1 CR   (m1,m2) = (1/2,0)
  Tm  : g1=1/2, g3=0, g2 = 0         -> (j3 j4 | j2 j5)   zz0 RC, zz1 RR   (0,1/2), n2 = n1 - 1/2
  R4  : g3-g1 >= 3/2                -> (j2 j4 | j3 j5)   zz0 RR, zz1 RR
  R2+ : g3-g1 = 1/2, g1 >= 1/2      -> (j2 j4 | j3 j5)   zz0 CR, zz1 RR   (m2 = m1 + 1/2)
  R4c : g1=0, g3=1/2, g2 >= 1/2      -> (j2 j3 | j4 j5)   zz0 RR, zz1 RR   (0,1/2)
  Tp  : g1=0, g3=1/2, g2 = 0         -> (j2 j3 | j4 j5)   zz0 RC, zz1 RR   (0,1/2), n2 = n1 + 1/2

In every case both zigzag imbalances are in {0,+1}, so B is injective and lam(T) = sigma_min(B).
"""
from fractions import Fraction as Fr


def regime(tj):
    """tj = (3, A, B, C, D) doubled, any order.  Returns (name, ordered doubled tuple)."""
    s2 = tj[0]
    J = sorted(tj[1:])
    j2, j3, j4, j5 = J
    g1, g2, g3 = j3 - j2, j4 - j3, j5 - j4          # doubled gaps
    d = g1 - g3                                     # doubled, odd
    if d >= 3:
        return 'R1', (s2, j2, j4, j3, j5)
    if d == 1:
        if g3 >= 1:
            return 'R2-', (s2, j2, j4, j3, j5)
        if g2 >= 1:
            return 'R1c', (s2, j2, j3, j4, j5)
        return 'Tm', (s2, j3, j4, j2, j5)
    if d <= -3:
        return 'R4', (s2, j2, j4, j3, j5)
    # d == -1
    if g1 >= 1:
        return 'R2+', (s2, j2, j4, j3, j5)
    if g2 >= 1:
        return 'R4c', (s2, j2, j3, j4, j5)
    return 'Tp', (s2, j2, j3, j4, j5)


EXPECTED = {'R1': ('CR', 'CR'), 'R2-': ('CR', 'RR'), 'R1c': ('CR', 'CR'), 'Tm': ('RC', 'RR'),
            'R4': ('RR', 'RR'), 'R2+': ('CR', 'RR'), 'R4c': ('RR', 'RR'), 'Tp': ('RC', 'RR')}

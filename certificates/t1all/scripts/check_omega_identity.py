"""Brute-force check (full tensor product, no recoupling theory) of the identities of Lemma O:
 (O1) J1.(J2-J3+J4-J5) = (J3+J5)^2 - (J2+J4)^2 =: Omega            on Inv
 (O2) T = (1/2i)[J2.J3 + J4.J5, Omega]                                on Inv
 (O3) T = (1/2i)([J2.J3, J1.(J2-J3)] + [J4.J5, J1.(J4-J5)])           on Inv
over all tuples with every doubled spin <= MAX.  Usage: python check_omega_identity.py MAX"""
import sys, itertools, numpy as np
from t1core import valid
from explore_omega import ops_for, inv_basis, dot, cross_dot

MX = int(sys.argv[1])
worst = [0, 0, 0]; n = 0
for t in itertools.product(range(1, MX + 1), repeat=5):
    if not valid(*t):
        continue
    ops = ops_for(list(t))
    P = inv_basis(ops)
    if P.shape[1] == 0:
        continue
    J1, J2, J3, J4, J5 = ops
    red = lambda Y: P.conj().T @ Y
    T = red(cross_dot(J1, J2, J3, P) + cross_dot(J1, J4, J5, P))
    Y = red(dot(J1, J2, P) - dot(J1, J3, P)); Z = red(dot(J1, J4, P) - dot(J1, J5, P))
    E = red(dot(J2, J3, P) + dot(J4, J5, P))
    b23 = red(dot(J2, J3, P)); g45 = red(dot(J4, J5, P))
    Pq = [J2[a] + J4[a] for a in range(3)]; Qq = [J3[a] + J5[a] for a in range(3)]
    Om = red(dot(Qq, Qq, P) - dot(Pq, Pq, P))
    sc = max(1.0, np.abs(T).max())
    worst[0] = max(worst[0], np.abs(Y + Z - Om).max() / max(1, np.abs(Om).max()))
    worst[1] = max(worst[1], np.abs(T - (E @ Om - Om @ E) / 2j).max() / sc)
    worst[2] = max(worst[2], np.abs(T - (b23 @ Y - Y @ b23 + g45 @ Z - Z @ g45) / 2j).max() / sc)
    n += 1
print(f'tuples {n}; max rel. residuals: O1 {worst[0]:.2e}  O2 {worst[1]:.2e}  O3 {worst[2]:.2e}')

"""Sanity check (not part of the proof logic, which is analytic): the closed-form tail bound
Wbar(n1) = 12 nu H_n1 [5a + 1 + 3/n1 + (2H_n1 + 2.6)/n1]  (nu = 1,2,4 and a = 1,0,0 for dn = -1,0,+1)
dominates the exact majorant coupling sum W_C3 on C3 configurations (all m1, m2 = m1+2 and (0,1); random larger m2).
Usage: python s2_tailcheck.py > ../outputs/s2_tailcheck_out.txt"""
import sys, os, random
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_proof import W_C3, H
def Wbar(n1, dn):
    nu, a = {-1: (1, 1), 0: (2, 0), 1: (4, 0)}[dn]
    Hn = H(n1)
    return 12 * nu * Hn * (5 * a + 1 + 3 / n1 + (2 * Hn + 2.6) / n1)
worst = 0; cnt = 0; rng = random.Random(3)
for n1 in list(range(21, 200)) + [250, 400, 700, 999]:
    for dn in (-1, 0, 1):
        n2 = n1 + dn
        pairs = [(0, 1)] + [(a, a + 2) for a in range(0, n2 - 4)] + [(a, rng.randint(a + 2, n2 - 3)) for a in range(0, n2 - 5, 3)]
        for m1, m2 in pairs:
            if n1 - m1 < 3 or n2 - m2 < 3: continue
            r = W_C3(n1, m1, n2, m2) / Wbar(n1, dn); cnt += 1
            if r > worst: worst = r; at = (n1, m1, n2, m2)
print(f"{cnt} C3 configurations: max W_C3 / Wbar = {worst:.4f} at {at}  ({'OK' if worst <= 1 else 'VIOLATION'})")

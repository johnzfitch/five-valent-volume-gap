"""Validation of the float certificate against exact rational LDL inertia (certcore.inertia_ldl) for all tuples
with 2j_max <= Mv, at several thresholds eps (including thresholds above the true minimum, where certification must fail
exactly when the exact count of sub-threshold nonzero eigenvalues is positive)."""
import sys, json
from fractions import Fraction as Fr
import numpy as np
from multiprocessing import Pool
from certcore import matrices, rank_exact, Cmat, inertia_ldl
from verify import certify
from verify2 import certify as certify_fast
from run_cert import tuples
TH = [Fr(45579607, 10**7), Fr(5), Fr(9), Fr(14)]       # thresholds for |eig 8T|
def work(tj):
    D = matrices(tj)
    if D is None or D['nO'] == 0: return None
    z = D['nO'] - rank_exact(D); C = Cmat(D); out = []
    for c in TH:
        eps = (c / 8) ** 2
        inn = inertia_ldl(D['nO'], C, D['thO'], eps)
        exact_ok = None if inn is None else (inn[0] == z and inn[1] == 0)
        fok, _ = certify(D['Bf'], z, float(eps) * (1 + 1e-15))
        fok2 = certify_fast(D['Bf'], z, float(eps) * (1 + 1e-15))[0]
        out.append((exact_ok, fok and fok2 if fok == fok2 else 'DISAGREE'))
    return tj, out
if __name__ == '__main__':
    Mv = int(sys.argv[1])
    tl = [t for M in range(1, Mv + 1) for t in tuples(M)]
    with Pool(8) as p: res = [r for r in p.map(work, tl, chunksize=8) if r]
    agree = 0; float_true_exact_false = 0; float_false_exact_true = 0; zero_piv = 0
    for tj, out in res:
        for ex, fl in out:
            if ex is None: zero_piv += 1; continue
            if ex == fl: agree += 1
            elif fl and not ex: float_true_exact_false += 1
            else: float_false_exact_true += 1
    lines = [f"tuples with 2j_max <= {Mv}: {len(res)}; thresholds q = {[float(c) for c in TH]}",
             f"agreement float certificate vs exact LDL: {agree}; float certified but exact fails (UNSOUND): {float_true_exact_false}; "
             f"float inconclusive but exact certifies: {float_false_exact_true}; exact zero pivot (skipped): {zero_piv}"]
    print("\n".join(lines)); open(f"../outputs/validate_M{Mv}.txt", "w").write("\n".join(lines) + "\n")

"""Check of Corollary 10 (integer j1): for every tuple with 2s even,
 (i) the pairing that puts the half-integer spins of legs 2..5 together has a closed colour class;
 (ii) the tuple is covered by Theorem 6/8 iff NOT (all of j2..j5 integer, or all half-integer >= 3/2);
 (iii) in the uncovered integer-j1 tuples, every closed pairing has even N-sum (so Conjecture W2 is what is needed),
      and the minimal lam(T0) over closed pairings is >= 1 numerically (W2 evidence).
Usage: python integer_j1_check.py M
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from survey_cover import tuples, pairings
from closed_class import closed_classes
from core import sites
from explore_T0 import T0, nzmin

M = int(sys.argv[1])
n = 0; fail_i = []; fail_ii = []; w2min = (9e9, None); unc = 0
for tj in tuples(M):
    if tj[0] % 2: continue
    if len(sites(tj)) < 2: continue
    n += 1
    js = tj[1:]
    half = [j for j in js if j % 2 == 1]
    # pairing grouping half-integers together
    if len(half) == 2:
        ints = [j for j in js if j % 2 == 0]
        P = (tj[0], half[0], half[1], ints[0], ints[1])
    else:
        P = tj
    if len(sites(P)) >= 2 and not any(closed_classes(P)):
        fail_i.append(tj)
    covered = False; closed_even = []
    for Q in pairings(tj):
        if len(sites(Q)) < 2: continue
        cl = any(closed_classes(Q))
        N1 = min(Q[1], Q[2]) + 1; N2 = min(Q[3], Q[4]) + 1
        lad = min(Q[1], Q[2]) == 1 or min(Q[3], Q[4]) == 1
        if cl and ((N1 + N2) % 2 == 1 or lad):
            covered = True
        elif cl:
            closed_even.append(Q)
    predicted_unc = (len(half) == 0) or (len(half) == 4 and min(js) >= 3)
    if covered == predicted_unc:
        fail_ii.append(tj)
    if not covered:
        unc += 1
        for Q in closed_even:
            S0, W0 = T0(Q); l0, _ = nzmin(W0, S0, tol=1e-10)
            if l0 is not None and l0 < w2min[0]:
                w2min = (l0, Q)
print(f"integer j1, 2j<= {M}: {n} tuples; uncovered {unc}")
print(f" (i) closed pairing exists (half-integers grouped): failures {len(fail_i)} {fail_i[:5]}")
print(f" (ii) covered <=> not(all integer or all half-integer >=3/2): failures {len(fail_ii)} {fail_ii[:5]}")
print(f" (iii) min lam(T0) over closed even-N-sum pairings of uncovered tuples: {w2min}")

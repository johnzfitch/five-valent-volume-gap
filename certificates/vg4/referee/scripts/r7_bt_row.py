"""Referee check R7: Brunnemann-Thiemann gr-qc/0405060 p.38 table, row k=6, spins (1,1,j-1,j),
printed as lambda = 4 sqrt(j(j+1)) in their q-hat = 4Q normalisation (row k=1 gives 2 sqrt(j(j+1))
for (1/2,1/2,j,j), consistent with q-hat = 4Q).  Brute force (from r1_matrix_validation.py) vs both.
Run: python r7_bt_row.py OUT
"""
import sys, math, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1_matrix_validation.py")).read()
ns = {}
exec(src[:src.index("t0 = time.time()")], ns)   # definitions only (spin matrices, brute force)
out = []
for j2 in (4, 6, 8):
    j = j2/2
    ev, d, _ = ns["brute_spectrum"]((2, 2, j2-2, j2))
    lam = min(abs(e) for e in ev if abs(e) > 1e-9)
    out.append(f"R7 (1,1,{j-1},{j}): dim {d}, 4*lambda_min(Q) brute force = {4*lam:.12f}; "
               f"4sqrt(j(j+1)) = {4*math.sqrt(j*(j+1)):.12f}; 4sqrt(j^2-1) = {4*math.sqrt(j*j-1):.12f}")
print("\n".join(out))
open(sys.argv[1] if len(sys.argv) > 1 else "r7_out.txt", "w").write("\n".join(out)+"\n")

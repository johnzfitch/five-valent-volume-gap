"""Brute-force checks (tensor products, float64) of:
 (I1) grid closed form: |spec T| on Inv equals |spec W| (core.grid), T = J1.(J2xJ3 + J4xJ5);
 (I2) T = J1.(J2xJ3 + J2xJ4 + J3xJ4) on Inv;
 (I3) T = -i[A,B] on Inv, A = J2.J3 - J4.J5, B = J3.J4 - J2.J5;
 (I4) T = -E on Inv where E = sum_{2<=a<b<c<=5} J_a.(J_b x J_c), and E acts on j2..j5 with
      spec(E | spin-s sector of J2+..+J5) = spec(T on Inv(s,j2..j5)) (multiplicities aside);
 (I5) cyclic symmetry: E(J2,J3,J4,J5) = E(J3,J4,J5,J2); reversal gives -E.
All 2j_i <= 3 (and a few larger).
"""
import itertools, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import leg_ops, inv_basis, triple, grid

def dot(A, B):
    return sum(A[a] @ B[a] for a in range(3))

maxerr = {k: 0.0 for k in ['I1', 'I2', 'I3', 'I4', 'I5', 'I5r']}
count = 0
M = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for tj in itertools.product(range(1, M + 1), repeat=5):
    if sum(tj) % 2:
        continue
    if np.prod([t + 1 for t in tj]) > 1100:
        continue
    ops = leg_ops(list(tj))
    V = inv_basis(ops)
    if V.shape[1] == 0:
        continue
    count += 1
    T = triple(ops, 0, 1, 2) + triple(ops, 0, 3, 4)
    TI = V.conj().T @ T @ V
    ev = np.sort(np.abs(np.linalg.eigvalsh((TI + TI.conj().T) / 2)))
    S, W = grid(tj)
    evW = np.sort(np.abs(np.linalg.eigvalsh(W))) if len(S) else np.zeros(0)
    maxerr['I1'] = max(maxerr['I1'], np.max(np.abs(ev - evW)) if ev.size == evW.size else 99)
    T2 = triple(ops, 0, 1, 2) + triple(ops, 0, 1, 3) + triple(ops, 0, 2, 3)
    maxerr['I2'] = max(maxerr['I2'], np.abs(V.conj().T @ (T2 - T) @ V).max())
    A = dot(ops[1], ops[2]) - dot(ops[3], ops[4])
    B = dot(ops[2], ops[3]) - dot(ops[1], ops[4])
    T3 = -1j * (A @ B - B @ A)
    maxerr['I3'] = max(maxerr['I3'], np.abs(V.conj().T @ (T3 - T) @ V).max())
    E = triple(ops, 1, 2, 3) + triple(ops, 1, 2, 4) + triple(ops, 1, 3, 4) + triple(ops, 2, 3, 4)
    maxerr['I4'] = max(maxerr['I4'], np.abs(V.conj().T @ (T + E) @ V).max())
    # cyclic: legs (2,3,4,5)->(3,4,5,2)
    Ec = triple(ops, 2, 3, 4) + triple(ops, 2, 3, 1) + triple(ops, 2, 4, 1) + triple(ops, 3, 4, 1)
    maxerr['I5'] = max(maxerr['I5'], np.abs(Ec - E).max())
    Er = triple(ops, 4, 3, 2) + triple(ops, 4, 3, 1) + triple(ops, 4, 2, 1) + triple(ops, 3, 2, 1)
    maxerr['I5r'] = max(maxerr['I5r'], np.abs(Er + E).max())
print(f"tuples checked: {count} (all 2j_i <= {M}, total dim <= 1100)")
for k, v in maxerr.items():
    print(f"  {k}: max abs error {v:.2e}")

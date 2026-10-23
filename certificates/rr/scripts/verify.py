"""Rigorous floating-point inertia certificate (IEEE-754 binary64, round-to-nearest; error bounds of Higham, Accuracy and
Stability of Numerical Algorithms, 2nd ed., Lemma 3.4/3.5: |fl(x.y) - x.y| <= gamma_k |x|.|y|, gamma_k = k u/(1 - k u)).
Claim certified for a tuple: every nonzero singular value of B (bipartite block of the gauge-fixed T) is >= sqrt(eps),
i.e. every nonzero eigenvalue of 8T has modulus >= 8 sqrt(eps).
Steps:
 1. dim ker B = nO - rank(Bt), Bt rational (exact, flint).                      [z]
 2. Bh = fl(sqrt(fl(r_e))) entrywise; |Bh - B| <= 2.0001u |B| (correct rounding of float(Fraction) and sqrt), so
    |sigma_i(B) - sigma_i(Bh)| <= delta := 2.0001u (1+1e-12) ||Bh||_F (1+2.0001u)    (Weyl for singular values).
 3. A := Bh^T Bh exactly (as a real matrix); target t := (sqrt(eps) + delta)^2 (rounded up).
    Compute X = eigenvectors of fl(A) (any float matrix), Mc = fl(X^T fl(fl(A - tI) X)) and an entrywise bound F >= |Mc - X^T(A - tI)X|
    (error of A, of A - tI, and of the two products, all via gamma bounds).  Also G = fl(X^T X) with bound FG.
 4. If every row of Mc is Gershgorin-separated from 0 (|Mc_ii| > sum_{j!=i}|Mc_ij| + sum_j F_ij) and X^T X is
    Gershgorin-positive, then X is nonsingular and inertia(A - tI) = (#{Mc_ii<0}, 0, #{Mc_ii>0}) (Sylvester).
 5. Certified iff #{Mc_ii < 0} == z: then exactly z singular values of Bh are < sqrt(t), and every nonzero sigma(B) >= sqrt(eps).
All bound arithmetic is done in float64 with explicit upward safety factors."""
import numpy as np, math
u = 2.0 ** -53
def gam(k): return k * u / (1 - k * u) * 1.01 + 1e-300
def certify(Bh, z, eps):
    nE, nO = Bh.shape
    if nO == 0: return True, None
    fro = math.sqrt(float(np.sum(Bh * Bh))) * (1 + 1e-12)
    delta = 2.0001 * u * fro * (1 + 2.0001 * u) * (1 + 1e-12)
    t = (math.sqrt(eps) * (1 + 1e-15) + delta) ** 2 * (1 + 1e-15)
    aB = np.abs(Bh)
    A = Bh.T @ Bh
    FA = gam(nE + 2) * (aB.T @ aB) * (1 + 1e-12)            # |fl(A) - A|
    At = A - t * np.eye(nO)
    FAt = FA + u * np.abs(At) * 1.01 + 1e-300                 # subtraction rounding on the diagonal (bounded generously)
    w, X = np.linalg.eigh(A)
    aX = np.abs(X)
    Y = At @ X
    FY = gam(nO + 2) * (np.abs(At) @ aX) + FAt @ aX           # |fl(At X) - (At_exact) X|
    Mc = X.T @ Y
    F = gam(nO + 2) * (aX.T @ np.abs(Y)) + aX.T @ FY
    F = F * (1 + 1e-10)
    rs = np.sum(np.abs(Mc), axis=1)
    rad = rs - np.abs(np.diag(Mc)) + np.sum(F, axis=1) + 4 * (nO + 2) * u * (rs + np.sum(F, axis=1))   # rounding of these sums
    if not np.all(np.abs(np.diag(Mc)) > rad): return False, 'gershgorin'
    G = X.T @ X
    FG = gam(nO + 2) * (aX.T @ aX) * (1 + 1e-10)
    rsG = np.sum(np.abs(G), axis=1)
    radG = rsG - np.abs(np.diag(G)) + np.sum(FG, axis=1) + 4 * (nO + 2) * u * (rsG + np.sum(FG, axis=1))
    if not np.all(np.diag(G) > radG): return False, 'X singular?'
    neg = int(np.sum(np.diag(Mc) < 0))
    if neg != z: return False, f'count {neg} != ker {z}'
    return True, None

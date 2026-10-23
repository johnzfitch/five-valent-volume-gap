"""Faster rigorous certificate (same theorem as verify.py; used from level 24 on).
Differences: (i) Gershgorin needs only ROW SUMS of the error bound F, computed by matrix-vector products
 (sum_j (|X|^T |Y|)_ij = (|X|^T (|Y| 1))_i etc.), all bounds float with upward safety factors;
(ii) nonsingularity of X follows from Mc being Gershgorin-nonsingular (det Mc = det(X)^2 det(A - tI)), so X^T X is not needed;
(iii) eigenvalues of fl(A) give q (no separate SVD).
Error model as in verify.py: |fl(x.y) - x.y| <= gamma_k |x|.|y|, IEEE binary64 round-to-nearest."""
import numpy as np, math
u = 2.0 ** -53
def gam(k): return k * u / (1 - k * u) * 1.01
def certify(Bh, z, eps):
    nE, nO = Bh.shape
    fro = math.sqrt(float(np.sum(Bh * Bh))) * (1 + 1e-12)
    delta = 2.0001 * u * fro * (1 + 2.0001 * u) * (1 + 1e-12)
    t = (math.sqrt(eps) * (1 + 1e-15) + delta) ** 2 * (1 + 1e-15)
    A = Bh.T @ Bh
    aB = np.abs(Bh)
    w, X = np.linalg.eigh(A)
    aX = np.abs(X); one = np.ones(nO)
    At = A - t * np.eye(nO); aAt = np.abs(At)
    # |fl(A) - A| <= gam(nE+2) |B|^T|B| =: FA ; |fl(At) - At_exact| <= FA + u|At|  =: FAt
    # FY = gam(nO+2)|At||X| + FAt|X| ; F = gam(nO+2)|X|^T|Y| + |X|^T FY ; need row sums F 1.
    Y = At @ X
    Mc = X.T @ Y
    aXone = aX @ one                                  # |X| 1
    FA_v = gam(nE + 2) * (aB.T @ (aB @ aXone))        # FA |X| 1
    FAt_v = FA_v + 1.01 * u * (aAt @ aXone)           # FAt |X| 1
    FY_v = gam(nO + 2) * (aAt @ aXone) + FAt_v        # FY 1
    Frow = gam(nO + 2) * (aX.T @ (np.abs(Y) @ one)) + aX.T @ FY_v
    Frow = Frow * (1 + 1e-9)
    rs = np.abs(Mc) @ one
    rad = rs - np.abs(np.diag(Mc)) + Frow + 4 * (nO + 2) * u * (rs + Frow)
    if not np.all(np.abs(np.diag(Mc)) > rad): return False, 'gershgorin', w
    neg = int(np.sum(np.diag(Mc) < 0))
    if neg != z: return False, f'count {neg} != ker {z}', w
    return True, None, w

"""Validation of the ladder reduction (Proposition L) for the BR valence-6 family.

V1. Embedding: integer tangents, all 20 triple determinants nonzero, sigma = (2,0,0,0,0,0,0,0,-2,0) exactly.
V2. Full recoupling operator (hv_core of ../../volume_gap_higher_valence_20261021, imported read-only) vs the
    closed-form ladder model: all singular values, n = 1..12 (float64).
V3. Link structure and magnitudes read off the full operator in the joint eigenbases of (C13^2, G^2, E^2) and
    (C13^2, H^2, E^2): X links, Y links, Y zero rows, n = 3..8.
V4. Recoupling magnitudes alpha_-, beta_-, alpha_+, beta_+ against exact sympy 6j, n = 2..12, all e.
V5. Ladder q_min at 60 digits vs the exact certified values of the previous folder (n = 1,3,5,7,9).
"""
import sys, itertools
import numpy as np, mpmath as mp
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))
from hv_core import Vertex, sigma_vec_BR
from ladder import model
out = []

# V1
V = [(-1, 2, -3), (-2, 4, 1), (2, -1, -4), (-3, 1, 4), (-4, 3, 3), (-2, -3, 2)]
detx = lambda a, b, c: a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])
eps = {T: (1 if detx(*[V[i] for i in T]) > 0 else -1) for T in itertools.combinations(range(6), 3)}
dets = [detx(*[V[i] for i in T]) for T in itertools.combinations(range(6), 3)]
sig = tuple(eps[(I, J, K)] - eps[(I, J, 5)] + eps[(I, K, 5)] - eps[(J, K, 5)] for (I, J, K) in itertools.combinations(range(5), 3))
out.append(f"V1 tangents {V}: min |det| = {min(abs(d) for d in dets)} (all 20 nonzero: {all(dets)}), sigma = {sig}")

# V2
sig6 = sigma_vec_BR((2, 0, 0, 0, 0, 0, 0, 0, -2, 0), 6)
worst = 0.0
for n in range(1, 13):
    Vx = Vertex((1, 1, n, n, n + 1, n + 1)); A = Vx.Qsigma(sig6)
    sf = np.sort(np.linalg.svd(A, compute_uv=False))
    K, S = model(n, (1, 1), ctx=np)
    sm = np.sort(np.linalg.svd(8 * K, compute_uv=False))
    assert len(sm) == len(sf) == 4 * n + 2
    worst = max(worst, float(np.max(np.abs(sm - sf)) / sf.max()))
out.append(f"V2 full operator vs ladder model, all singular values, n=1..12: dims 4n+2, max rel. deviation {worst:.2e}")

# V3
def casimir(Vx, legs, spins2):
    M = np.eye(Vx.dim) * sum((s / 2) * (s / 2 + 1) for s in spins2)
    for a in range(len(legs)):
        for b in range(a + 1, len(legs)):
            M = M + 2 * Vx.dot(legs[a], legs[b])
    return M
lab = lambda U, M, i: round((-1 + np.sqrt(1 + 4 * (U[:, i] @ M @ U[:, i]))) / 2 * 2) / 2
okX = okY = okZ = True
for n in range(3, 9):
    Vx = Vertex((1, 1, n, n, n + 1, n + 1))
    Xr = Vx.qhat(0, 1, 2) / 4; Yr = Vx.qhat(1, 3, 4) / 4
    C13 = casimir(Vx, [0, 2], [1, n]); G = casimir(Vx, [0, 1, 2], [1, 1, n]); E = casimir(Vx, [3, 4], [n, n + 1])
    H = casimir(Vx, [1, 3, 4], [1, n, n + 1])
    _, U1 = np.linalg.eigh(C13 + np.pi * G + np.e ** 2 * E)
    Xb = U1.T @ Xr @ U1
    x = np.sqrt(n * (n + 2)) / 4
    for a in range(Vx.dim):
        for b in range(a + 1, Vx.dim):
            if abs(Xb[a, b]) > 1e-9:
                ca, ga, ea = lab(U1, C13, a), lab(U1, G, a), lab(U1, E, a)
                cb, gb, eb = lab(U1, C13, b), lab(U1, G, b), lab(U1, E, b)
                if not (ga == gb == n / 2 and ea == eb and ca != cb and abs(abs(Xb[a, b]) - x) < 1e-9):
                    okX = False
    _, U2 = np.linalg.eigh(C13 + np.pi * H + np.e ** 2 * E)
    Yb = U2.T @ Yr @ U2
    zero_rows = []
    for a in range(Vx.dim):
        if np.all(np.abs(Yb[a]) < 1e-9):
            zero_rows.append((lab(U2, C13, a), lab(U2, H, a), lab(U2, E, a)))
        for b in range(a + 1, Vx.dim):
            if abs(Yb[a, b]) > 1e-9:
                ha, hb = lab(U2, H, a), lab(U2, H, b); ea, eb = lab(U2, E, a), lab(U2, E, b)
                k = max(ea, eb)
                yp = np.sqrt(((n + 1.5) ** 2 - k * k) * (k * k - 0.25)) / 4
                if not (ha == hb and abs(ea - eb) == 1 and lab(U2, C13, a) == lab(U2, C13, b) and abs(abs(Yb[a, b]) - yp) < 1e-9):
                    okY = False
    if sorted(zero_rows) != sorted([((n + 1) / 2, 0.0, 0.5), ((n + 1) / 2, n + 1.0, n + 0.5)]):
        okZ = False
out.append(f"V3 (n=3..8) X links only (c-,g=n/2,e)-(c+,g=n/2,e) with |x|=sqrt(n(n+2))/4: {okX}; "
           f"Y links (c,h,e)-(c,h,e+1) with |y_h| closed form: {okY}; Y zero rows exactly (c+,h=0,e=1/2),(c+,h=n+1,e=n+1/2): {okZ}")

# V4
from sympy.physics.wigner import wigner_6j
from sympy import Rational as R, sqrt as ssqrt, simplify
okR = True
for n in range(2, 13):
    j6 = R(n + 1, 2); g = R(n, 2)
    for cs, c in (('-', R(n - 1, 2)), ('+', R(n + 1, 2))):
        for e2 in range(1, 2 * n + 2, 2):
            e = R(e2, 2)
            for h in (e - R(1, 2), e + R(1, 2)):
                if h < 0:
                    continue
                v2 = simplify(((2 * g + 1) * (2 * h + 1)) * wigner_6j(c, R(1, 2), g, e, j6, h) ** 2)
                if cs == '-':
                    pred = (n + e + R(3, 2)) * (e - R(1, 2)) / (n * (2 * e + 1)) if h == e - R(1, 2) else (n + R(1, 2) - e) * (e + R(3, 2)) / (n * (2 * e + 1))
                else:
                    pred = (n + R(3, 2) - e) / (2 * n + 4) if h == e - R(1, 2) else (n + R(5, 2) + e) / (2 * n + 4)
                if h > n + 1 or (cs == '-' and (h < 1 or h > n)):
                    pred = 0
                if simplify(v2 - pred) != 0:
                    okR = False
out.append(f"V4 recoupling magnitudes (exact sympy 6j, n=2..12, all e): {okR}")

# V5
mp.mp.dps = 60
cert = {1: "0.8409", 3: "0.04590", 5: "6.864e-4", 7: "4.655e-6", 9: "1.792e-8"}
for n in (1, 3, 5, 7, 9):
    K, S = model(n, (1, 1), ctx=mp)
    q = 8 * min(mp.svd_r(K, compute_uv=False))
    out.append(f"V5 n={n}: ladder q_min = {mp.nstr(q, 12)}  (previous folder, exact-certified: {cert[n]})")
print("\n".join(out))
open("../outputs/v6_validate_output.txt", "w").write("\n".join(out) + "\n")

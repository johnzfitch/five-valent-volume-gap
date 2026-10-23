"""Remaining checks for THEOREM.md.

C1  Integer tangent vectors realising, in normal-form labelling (distinguished edge = leg 1, leg 5 eliminated),
    sigma = (2,0,0,0), (2,2,2,0), (4,2,2,0); exact integer determinants.
C2  Brute force on the full tensor product (all ten eps-weighted triples, from the tangents of C1) versus the
    normal form -4i[alpha J1.(J2xJ3) + beta J1.(J4xJ5)] built directly on the tensor product, versus grid.py,
    for all labelled spins with 2j_i <= 3 (independent of hv_core's recoupling code).
C3  Analytic constants of Theorem B2: min over n of the even and odd chain bounds.
C4  Proposition T (general ratio t): for the family (1/2; j-1/2, j; j, j) and alpha = 1, beta = t, the explicit
    bound q_min <= U_N(t) := N sqrt(2N+1) t^N (times 1 = alpha... see THEOREM) versus the true q_min.
C5  Orbit of the 5-valent sign pattern used by Sherif (arXiv 2606.18397, footnote, sec 4.5) - curiosity.
"""
import itertools, os, sys
import numpy as np
from fractions import Fraction as Fr
from math import sqrt, pi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orbits import T5, POS, sigma, act, eps_of_vectors
import json
from grid import spectrum, W_matrix
from hv_core import spin_mats

out = []
P = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))

# ---------------- C1
target = {1: (2, 0, 0, 0), 2: (2, 2, 2, 0), 3: (4, 2, 2, 0)}
rng = np.random.default_rng(3)
tang = {}
while len(tang) < 3:
    V = rng.integers(-3, 4, size=(5, 3))
    e = eps_of_vectors(V)
    if 0 in e:
        continue
    s = sigma(e)
    for k, t in target.items():
        if k not in tang and s == t:
            tang[k] = (V.tolist(), e)
P("C1 tangents (rows = legs 1..5) realising the normal forms, with exact integer determinants:")
for k in (1, 2, 3):
    V, e = tang[k]
    dets = [int(round(np.linalg.det(np.array([V[i], V[j], V[l]])))) for (i, j, l) in T5]
    P(f"  orbit {k}: sigma={target[k]} tangents={V} dets(123..345)={dets}")
# closed folder's Theorem-1 tangents
V0 = [(1, 3, 1), (0, -2, 0), (-2, 0, -3), (-3, 1, 0), (-3, 1, 2)]
P("  closed Theorem 1 tangents give sigma =", sigma(eps_of_vectors(V0)))

# ---------------- C2
def legs_ops(Jd):
    dims = [J + 1 for J in Jd]
    ops = []
    for leg, J in enumerate(Jd):
        emb = []
        for Mx in spin_mats(J):
            T = np.array([[1.0 + 0j]])
            for l2, dd in enumerate(dims):
                T = np.kron(T, Mx if l2 == leg else np.eye(dd))
            emb.append(T)
        ops.append(emb)
    tot = [sum(ops[l][a] for l in range(len(Jd))) for a in range(3)]
    C = sum(t @ t for t in tot)
    w, U = np.linalg.eigh(C)
    Pinv = U[:, np.abs(w) < 1e-8]
    return ops, Pinv


def trip(a, b, c):
    return (a[0] @ (b[1] @ c[2] - b[2] @ c[1]) + a[1] @ (b[2] @ c[0] - b[0] @ c[2]) + a[2] @ (b[0] @ c[1] - b[1] @ c[0]))


worst = 0.0; ncase = 0
for Jd in itertools.product(range(1, 4), repeat=5):
    if sum(Jd) % 2:
        continue
    ops, Pinv = legs_ops(Jd)
    if Pinv.shape[1] == 0:
        continue
    for k in (1, 2, 3):
        V, e = tang[k]
        A = sum(4 * e[POS[t]] * trip(ops[t[0]], ops[t[1]], ops[t[2]]) for t in T5)   # i * sum eps qhat
        al, be = target[k][0], target[k][1]
        Bn = 4 * al * trip(ops[0], ops[1], ops[2]) + 4 * be * trip(ops[0], ops[3], ops[4])
        a1 = np.sort(np.abs(np.linalg.eigvalsh(Pinv.conj().T @ A @ Pinv)))
        a2 = np.sort(np.abs(np.linalg.eigvalsh(Pinv.conj().T @ Bn @ Pinv)))
        a3 = spectrum(Jd, al, be)
        R = Pinv.conj().T @ (A - Bn) @ Pinv
        worst = max(worst, np.abs(a1 - a2).max(), np.abs(a1 - a3).max(), np.abs(R).max())
        ncase += 1
P(f"C2 brute force (all ten eps triples) = normal form = grid: {ncase} (spins, orbit) cases with 2j_i<=3; "
  f"max discrepancy (spectra and operator difference on Inv) {worst:.2e}")

# ---------------- C3
NMAX = 200000
H1 = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, NMAX + 1))])
H2 = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, NMAX + 1) ** 2)])
H = lambda K, p=1: (H1 if p == 1 else H2)[K]
ev = min((sqrt(8 * n / (H((n - 1) // 2) ** 2 + H((n - 1) // 2, 2))), n) for n in range(4, NMAX + 1))
P(f"C3 even-chain bound sqrt(8n/(H_K^2+H2_K)), K=floor((n-1)/2), min over 4<=n<=2e5: {ev[0]:.6f} at n={ev[1]}")
P(f"   odd-chain bounds: left >= sqrt(12*6)/pi = {sqrt(72)/pi:.6f}; right >= sqrt(4(n+4)/H_K) >= {min(sqrt(4*(n+4)/H(max((n-2)//2,1))) for n in range(4,NMAX+1)):.6f}")
# monotonicity beyond 2e5: 8n/((1+ln n)^2+2) is increasing for n>=e^... ; check derivative sign numerically on a log grid
import math
g = lambda n: 8 * n / ((1 + math.log(n)) ** 2 + 2)
P("   for n>2e5 use H_K<=1+ln n, H2<=2: 8n/((1+ln n)^2+2) at n=2e5:", f"{g(2e5):.1f}", "(increasing for n>=8)")

# ---------------- C4
import mpmath as mp
mp.mp.dps = 60
def chain_family(N, al, be, swap=False):
    # spins (1/2; j-1/2, j; j, j), N = 2j: D_l = beta*l*sqrt((N+1)^2-l^2) (l=1..N), X_l = alpha*sqrt((N-l)(N+l+1)l(l+1)) (l=1..N-1)
    # (constants: BR weights with alpha=4,beta=2 are 2*D and 4*X/... ; here alpha, beta scale the two pair operators)
    # swap=True: pairs exchanged, i.e. alpha-pair = (j, j), beta-pair = (j-1/2, j): same chain with alpha<->beta roles
    if swap:
        al, be = be, al
    D = [be * l * mp.sqrt((N + 1) ** 2 - l * l) for l in range(1, N + 1)]
    X = [al * mp.sqrt((N - l) * (N + l + 1) * l * (l + 1)) for l in range(1, N)]
    c = []
    for l in range(N):
        c.append(D[l])
        if l < N - 1:
            c.append(X[l])
    T = mp.zeros(2 * N, 2 * N)
    for i, x in enumerate(c):
        T[i, i + 1] = T[i + 1, i] = x
    e = sorted(abs(x) for x in mp.eigsy(T, eigvals_only=True))
    ub = mp.fprod(D) / mp.fprod(X)
    return e[0], ub
P("C4 family (1/2; j-1/2, j; j, j), A ~ J1.(J2xJ3) + t J1.(J4xJ5); q_min <= prod(D)/prod(X) (Lemma 2), 60 digits:")
for t in (0.5, 0.8, 0.9, 0.99):
    rows = []
    for N in (10, 40, 80):
        q, ub = chain_family(N, 1, mp.mpf(t))
        rows.append(f"N={N}: q={mp.nstr(q,4)} <= {mp.nstr(ub,4)}")
    P(f"   t={t}: " + "; ".join(rows))
P("   t>1: by exchanging the two pairs, J1.(T + t U) = t J1.(U + T/t); same bounds with t -> 1/t.")
# ---------------- C5
sher = {(0,1,2):1,(0,1,3):-1,(0,1,4):1,(0,2,3):-1,(0,2,4):-1,(0,3,4):1,(1,2,3):1,(1,2,4):1,(1,3,4):-1,(2,3,4):1}
e = tuple(sher[t] for t in T5)
O = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "orbits.json")))
for o in O["orbits"]:
    if list(sigma(e)) in [list(s) for s in o["sigma"]]:
        P(f"C5 Sherif's 5-valent pattern: sigma = {sigma(e)} -> orbit {o['orbit']}")
with open(sys.argv[1] if len(sys.argv) > 1 else "theorem_checks_out.txt", "w") as f:
    f.write("\n".join(out) + "\n")

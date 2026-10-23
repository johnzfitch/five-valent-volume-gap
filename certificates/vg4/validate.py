"""Validation of the closed-form 4-valent triple-product matrix.

1. Brute force: full tensor-product construction of J1.(J2xJ3) restricted to the
   SU(2)-invariant subspace, compared with the closed-form tridiagonal matrix, for
   every sorted spin tuple with 2j <= 4 and for all 24 leg orderings of a sample.
2. Published values: Bianchi-Haggard PRL 107 (2011) 011301 (arXiv:1102.5439) table
   (V = (sqrt2/3) sqrt|q|); Brunnemann-Thiemann gr-qc/0405060 table p.38
   (eigenvalue series of q_hat = 4Q); equal-spin characteristic polynomials.
3. Batched Sturm bisection (numpy and torch) against LAPACK on a sample.
Run:  python validate.py   (writes validate_out.txt)
"""
import itertools, math, sys, os
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vg_core as vg

out = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.append(s)

# ---------------- 1. brute force
worst = 0.0; n = 0
for T in vg.sorted_tuples(4):
    bf, dinv = vg.bruteforce_Q_spectrum(T)
    cf = np.sort(np.linalg.eigvalsh(vg.q_matrix(*T)))
    assert dinv == len(cf), (T, dinv, len(cf))
    worst = max(worst, np.max(np.abs(bf - cf))); n += 1
log(f"[1] brute force vs closed form: {n} sorted tuples with 2j<=4, max |diff| = {worst:.2e}")
worst = 0.0; n = 0
for T in [(1, 2, 3, 4), (2, 2, 3, 3), (1, 3, 3, 3), (2, 3, 4, 3)]:
    ref = np.sort(np.linalg.eigvalsh(vg.q_matrix(*sorted(T))))
    for P in set(itertools.permutations(T)):
        if not vg.admissible(*P):
            continue
        cf = np.sort(np.linalg.eigvalsh(vg.q_matrix(*P)))
        bf, _ = vg.bruteforce_Q_spectrum(P)
        worst = max(worst, np.max(np.abs(cf - ref)), np.max(np.abs(bf - ref))); n += 1
log(f"[1b] leg-permutation invariance of the spectrum: {n} orderings, max |diff| = {worst:.2e}")

# ---------------- 2a. Bianchi-Haggard table
BH = {(1, 1, 1, 1): [0.310], (1, 1, 2, 2): [0.396], (1, 1, 3, 3): [0.464], (1, 2, 2, 3): [0.498],
      (2, 2, 2, 2): [0.0, 0.620], (1, 1, 4, 4): [0.522], (1, 2, 3, 4): [0.577], (2, 2, 2, 4): [0.620],
      (12, 12, 12, 14): [1.828, 3.204, 4.225, 5.133, 5.989, 6.817]}
worst = 0.0
for T, vals in BH.items():
    q = np.sort(np.linalg.eigvalsh(vg.q_matrix(*T)))
    v = np.unique(np.round(math.sqrt(2) / 3 * np.sqrt(np.abs(q)), 3))
    v = sorted(set(np.round(math.sqrt(2) / 3 * np.sqrt(np.abs(q)), 3).tolist()))
    ok = all(min(abs(x - y) for y in v) <= 0.0006 for x in vals)
    worst = max(worst, max(min(abs(x - y) for y in v) for x in vals))
    log(f"[2a] BH {tuple(t/2 for t in T)}: computed V = {v}  published {vals}  match={ok}")
log(f"[2a] worst deviation from BH table (3-digit rounding): {worst:.4f}")

# ---------------- 2b. Brunnemann-Thiemann series (q_hat = 4Q, smallest positive eigenvalue)
s3, s2, s5, s6 = math.sqrt(3), math.sqrt(2), math.sqrt(5), math.sqrt(6)
BT = [  # (label, tuple(j) as function of j, formula, min j)
    ("1/2,1/2,j,j", lambda j: (0.5, 0.5, j, j), lambda j: 2 * math.sqrt(j * (j + 1)), 0.5),
    ("1/2,1,j-1/2,j", lambda j: (0.5, 1, j - 0.5, j), lambda j: 2 * math.sqrt((j + 1) * (2 * j - 1)), 1),
    ("3/2,3/2,j,j", lambda j: (1.5, 1.5, j, j), lambda j: 2 * math.sqrt(17 * j * j + 17 * j - 21 - math.sqrt(208 * j**4 + 416 * j**3 - 344 * j * j - 552 * j + 441)), 3),
    ("1/2,3/2,j-1,j", lambda j: (0.5, 1.5, j - 1, j), lambda j: 2 * s3 * math.sqrt((j + 1) * (j - 1)), 1.5),
    ("1/2,2,j-3/2,j", lambda j: (0.5, 2, j - 1.5, j), lambda j: 2 * s2 * math.sqrt((j + 1) * (2 * j - 3)), 2),
    ("1,1,j-1,j", lambda j: (1, 1, j - 1, j), lambda j: 4 * math.sqrt(j * (j + 1)), 1.5),
    ("1,1,j-1,j [printed 4sqrt(j(j+1)) above; this row tests 4sqrt(j^2-1), see THEOREM.md]", lambda j: (1, 1, j - 1, j), lambda j: 4 * math.sqrt(j * j - 1), 1.5),
    ("1/2,3/2,j,j", lambda j: (0.5, 1.5, j, j), lambda j: 2 * math.sqrt((2 * j + 3) * (2 * j - 1)), 1.5),
    ("3/2,2,j-1/2,j", lambda j: (1.5, 2, j - 0.5, j), lambda j: math.sqrt(108 * j * j + 54 * j - 216 - 6 * math.sqrt(228 * j**4 + 228 * j**3 - 903 * j * j - 480 * j + 1152)), 2.5),
    ("1/2,5/2,j-2,j", lambda j: (0.5, 2.5, j - 2, j), lambda j: 2 * s5 * math.sqrt((j + 1) * (j - 2)), 2.5),
    ("1/2,3,j-5/2,j", lambda j: (0.5, 3, j - 2.5, j), lambda j: 2 * s3 * math.sqrt((j + 1) * (2 * j - 5)), 3),
    ("1,3/2,j-3/2,j", lambda j: (1, 1.5, j - 1.5, j), lambda j: 2 * s3 * math.sqrt((j + 1) * (2 * j - 3)), 2),
    ("1,3/2,j-1/2,j (odd)", lambda j: (1, 1.5, j - 0.5, j), lambda j: 2 * math.sqrt(14 * j * j + 7 * j - 16), 1.5),
    ("2,2,j,j (odd)", lambda j: (2, 2, j, j), lambda j: 2 * math.sqrt(52 * j * j + 52 * j - 114 - 18 * math.sqrt(4 * j**4 + 8 * j**3 - 16 * j * j - 20 * j + 33)), 2),
]
for label, tup, f, jmin in BT:
    worst = 0.0; cnt = 0
    for J in range(int(2 * jmin), 81):
        j = J / 2
        T = tuple(int(round(2 * x)) for x in tup(j))
        if not vg.admissible(*T) or min(T) < 1:
            continue
        c = vg.offdiag(*T)
        lam = 4 * vg.lapack_smallest_positive(c)
        worst = max(worst, abs(lam - f(j)) / f(j)); cnt += 1
    log(f"[2b] BT series ({label}): {cnt} values of j up to 40, max rel. deviation {worst:.2e}")

# ---------------- 2c. equal-spin characteristic polynomials (exact)
q = sp.symbols("q")
known = {1: 16 * q**2 - 3, 2: q * (q**2 - 3), 3: (16 * q**2 - 27) * (16 * q**2 - 315), 4: q * (q**4 - 99 * q**2 + 1296)}
for J, poly in known.items():
    c2 = vg.offdiag2_exact(J, J, J, J)
    d = len(c2) + 1
    M = sp.zeros(d, d)
    for s, v in enumerate(c2):
        r = sp.sqrt(sp.Rational(v.numerator, v.denominator))
        M[s, s + 1] = sp.I * r; M[s + 1, s] = -sp.I * r
    cp = sp.expand(M.charpoly(q).as_expr())
    ok = sp.simplify(cp - sp.expand(poly) / sp.Poly(poly, q).LC()) == 0
    log(f"[2c] equal spin j={J}/2: charpoly {sp.factor(cp)}  matches skill table: {ok}")

# ---------------- 3. Sturm bisection vs LAPACK
rng = np.random.default_rng(1)
tup = vg.sorted_tuples_np(40)
sample = tup[rng.choice(len(tup), 3000, replace=False)]
dims = np.array([vg.dim(*t) for t in sample])
worst_np = worst_t = 0.0
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
for dd in np.unique(dims):
    if dd < 2:
        continue
    S = sample[dims == dd]
    b2 = vg.offdiag2_batch(S)
    ref = np.array([vg.lapack_smallest_positive(np.sqrt(r)) for r in b2])
    a = vg.smallest_positive_eig(b2)
    b = vg.smallest_positive_eig(b2, xp=torch, device=dev).cpu().numpy()
    worst_np = max(worst_np, np.max(np.abs(a - ref) / ref)); worst_t = max(worst_t, np.max(np.abs(b - ref) / ref))
log(f"[3] Sturm bisection vs LAPACK on 3000 random tuples (2j<=40): max rel diff numpy {worst_np:.2e}, torch({dev}) {worst_t:.2e}")

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_out.txt"), "w").write("\n".join(out) + "\n")

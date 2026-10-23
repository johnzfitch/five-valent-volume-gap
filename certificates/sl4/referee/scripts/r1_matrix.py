"""Referee check 1: build the 4-valent volume matrix independently and test the factorisation.

(a) Brute force: for every ordered spin tuple with 2j_i <= 4 (and every sorted tuple with 2j_max = 5),
    build J_i on (C^{2j+1})^{(x)4}, find Inv as the kernel of the total J^2, restrict Q = J1.(J2xJ3), and
    compare its spectrum with the spectrum of the zero-diagonal Jacobi matrix whose squared links are
    h_k^2 f(k)/16 (hydrogen chain times multiplier, built from the SORTED spins as in Lemma P).
(b) Exact 6j: Q = (i/4)[K12^2, K23^2], so |Q_{k,k-1}|^2 = (k^2/4) |<k|K23^2|k-1>|^2.  With
    <(j1j2)k,j3;j4 | j1,(j2j3)l;j4> = +-sqrt((2k+1)(2l+1)) {j1 j2 k; j3 j4 l} (Racah formula, exact integers),
    the square of the matrix element is an exact rational (the l-dependent radicals pair up).
    Check a_k^2 == h_k^2 * f(k) / 16 exactly for all sorted tuples with 2j_max <= N6J and a list of large ones.
Nothing is imported from the author's scripts.
"""
import sys, itertools, time
from fractions import Fraction as Fr
from math import factorial
import numpy as np

out = []
def say(s):
    print(s, flush=True); out.append(s)

# ---------------- closed form via Lemma P (sorted spins), doubled-integer input
def chain_hf(J):
    Js = sorted(J)
    j1, j2, j3, j4 = [Fr(x, 2) for x in Js]
    A1, B1, A2, B2 = j1 + j2 + 1, abs(j1 - j2), j3 + j4 + 1, abs(j3 - j4)
    M, beta = max(B1, B2), min(B1, B2)
    K = j1 + j2
    if (Js[0] + Js[1] + Js[2] + Js[3]) % 2 or M > K:
        return None
    ks = []
    k = M + 1
    while k <= K:
        ks.append(k); k += 1
    h2 = [(A1 * A1 - k * k) * (k * k - M * M) / (4 * k * k - 1) for k in ks]
    f = [(k * k - beta * beta) * (A2 * A2 - k * k) for k in ks]
    return ks, [h2[i] * f[i] / 16 for i in range(len(ks))], int(K - M) + 1

def spin_ops(tj):
    j = tj / 2; m = np.arange(j, -j - 1, -1)
    jz = np.diag(m)
    jp = np.zeros((tj + 1, tj + 1))
    for i in range(1, tj + 1):
        jp[i - 1, i] = np.sqrt(j * (j + 1) - m[i] * (m[i] + 1))
    jx = (jp + jp.T) / 2; jy = (jp - jp.T) / (2j)
    return [jx, jy, jz]

def brute_spectrum(J):
    ops = [spin_ops(t) for t in J]
    dims = [t + 1 for t in J]
    def emb(i, A):
        mats = [np.eye(dd) for dd in dims]; mats[i] = A
        r = mats[0]
        for mm in mats[1:]:
            r = np.kron(r, mm)
        return r
    E = [[emb(i, ops[i][a]) for a in range(3)] for i in range(4)]
    T = [sum(E[i][a] for i in range(4)) for a in range(3)]
    C = sum(t @ t for t in T)
    w, V = np.linalg.eigh(C)
    P = V[:, np.abs(w) < 1e-8]
    if P.shape[1] == 0:
        return None
    J1, J2, J3 = E[0], E[1], E[2]
    cross = [J2[1] @ J3[2] - J2[2] @ J3[1], J2[2] @ J3[0] - J2[0] @ J3[2], J2[0] @ J3[1] - J2[1] @ J3[0]]
    Q = sum(J1[a] @ cross[a] for a in range(3))
    Qr = P.conj().T @ Q @ P
    assert np.allclose(Qr, Qr.conj().T, atol=1e-9)
    return np.sort(np.linalg.eigvalsh(Qr))

def chain_spectrum(c2, d):
    T = np.zeros((d, d))
    for i, x in enumerate(c2):
        T[i, i + 1] = T[i + 1, i] = np.sqrt(float(x))
    return np.sort(np.linalg.eigvalsh(T))

t0 = time.time()
nb = 0; worst = 0.0
tuples = [J for J in itertools.product(range(1, 5), repeat=4)]
tuples += [J for J in itertools.product(range(1, 6), repeat=4) if max(J) == 5 and list(J) == sorted(J)]
for J in tuples:
    if sum(J) % 2:
        continue
    cf = chain_hf(J)
    bs = brute_spectrum(J)
    if cf is None:
        assert bs is None, J
        continue
    ks, c2, d = cf
    assert bs is not None and len(bs) == d, (J, d, None if bs is None else len(bs))
    cs = chain_spectrum(c2, d)
    worst = max(worst, float(np.max(np.abs(cs - bs)))); nb += 1
say(f"(a) brute force vs Lemma-P chain: {nb} tuples (all ordered with 2j<=4, sorted with 2j_max=5); "
    f"max |spectrum difference| = {worst:.2e}; dims match in all; {time.time()-t0:.0f} s")

# ---------------- exact 6j
FC = [1]
def fact(n):
    while len(FC) <= n:
        FC.append(FC[-1] * len(FC))
    return FC[n]

def tri2(a, b, c):  # doubled; Delta^2 as Fraction, or None if not a triad
    if (a + b + c) % 2 or a > b + c or b > a + c or c > a + b:
        return None
    return Fr(fact((a + b - c) // 2) * fact((a - b + c) // 2) * fact((-a + b + c) // 2), fact((a + b + c) // 2 + 1))

def racah_sum(a, b, c, d, e, f):  # doubled args; the rational sum part of {a b c; d e f}
    t1 = [(a + b + c) // 2, (a + e + f) // 2, (d + b + f) // 2, (d + e + c) // 2]
    t2 = [(a + b + d + e) // 2, (a + c + d + f) // 2, (b + c + e + f) // 2]
    s = 0
    for t in range(max(t1), min(t2) + 1):
        den = 1
        for x in t1: den *= fact(t - x)
        for x in t2: den *= fact(x - t)
        s += Fr((-1) ** t * fact(t + 1), den)
    return s

def a2_6j(J):
    """exact |Q_{k,k-1}|^2 for k = kmin+1..kmax, basis (j1j2)k, using 6j with l = (j2 j3) coupling."""
    j1, j2, j3, j4 = J
    ks = [k for k in range(abs(j1 - j2), j1 + j2 + 1, 2) if tri2(k, j3, j4) is not None]
    ls = [l for l in range(abs(j2 - j3), j2 + j3 + 1, 2) if tri2(j1, j4, l) is not None]
    res = []
    for k in ks[1:]:
        km = k - 2
        Ck = tri2(j1, j2, k) * tri2(j3, j4, k) * tri2(j1, j2, km) * tri2(j3, j4, km)
        S = Fr(0)
        for l in ls:
            w = tri2(j1, j4, l) * tri2(j3, j2, l)
            S += Fr(l * (l + 2), 4) * (l + 1) * racah_sum(j1, j2, k, j3, j4, l) * racah_sum(j1, j2, km, j3, j4, l) * w
        mel2 = Ck * (k + 1) * (km + 1) * S * S          # <k|K23^2|k-1>^2
        kk = Fr(k, 2)
        res.append(kk * kk / 4 * mel2)
    return res

# sanity: orthogonality of the recoupling matrix for a few tuples (exact squares summing to 1)
def orth_check(J):
    j1, j2, j3, j4 = J
    ks = [k for k in range(abs(j1 - j2), j1 + j2 + 1, 2) if tri2(k, j3, j4) is not None]
    ls = [l for l in range(abs(j2 - j3), j2 + j3 + 1, 2) if tri2(j1, j4, l) is not None]
    for k in ks:
        s = sum((k + 1) * (l + 1) * tri2(j1, j2, k) * tri2(j3, j4, k) * tri2(j1, j4, l) * tri2(j3, j2, l)
                * racah_sum(j1, j2, k, j3, j4, l) ** 2 for l in ls)
        if s != 1:
            return False
    return True

say("(b) recoupling-matrix row norms == 1 exactly for (1,1,1,1),(2,3,4,5),(7,8,9,10),(6,10,11,13): " +
    str(all(orth_check(J) for J in [(1, 1, 1, 1), (2, 3, 4, 5), (7, 8, 9, 10), (6, 10, 11, 13)])))

N6J = int(sys.argv[1]) if len(sys.argv) > 1 else 20
t0 = time.time(); n6 = 0; bad = []
for J4 in range(1, N6J + 1):
    for J3 in range(1, J4 + 1):
        for J2 in range(1, J3 + 1):
            for J1 in range(1, J2 + 1):
                J = (J1, J2, J3, J4)
                cf = chain_hf(J)
                if cf is None or cf[2] < 2:
                    continue
                if a2_6j(J) != cf[1]:
                    bad.append(J)
                n6 += 1
say(f"(b) exact 6j |Q_(k,k-1)|^2 == h_k^2 f_k/16 for all {n6} sorted tuples with d>=2, 2j_max<={N6J}: "
    f"mismatches {len(bad)} {bad[:5]}; {time.time()-t0:.0f} s")
big = [(50, 50, 50, 50), (1, 1, 60, 60), (3, 17, 40, 56), (21, 33, 44, 70), (13, 29, 30, 46), (60, 61, 62, 63),
       (2, 41, 41, 80)]
t0 = time.time(); okb = []
for J in big:
    cf = chain_hf(J)
    okb.append((J, cf[2], a2_6j(J) == cf[1]))
say(f"(b) large tuples (doubled spins, d, exact match): {okb}; {time.time()-t0:.0f} s")
open("../outputs/r1_matrix.txt", "w").write("\n".join(out) + "\n")

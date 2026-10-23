"""Referee check R1: independent construction of Q = J1.(J2 x J3) on Inv(j1 j2 j3 j4).

Nothing is imported from the author's scripts.

(a) Brute force: spin matrices, 4-fold tensor product, invariant subspace = kernel of the
    total Casimir, Q = eps_abc J1a J2b J3c restricted there.  Done for EVERY ordered tuple
    (not only sorted ones) with 2j_i <= 4, plus a few with 2j_i up to 6.
(b) Recoupling: Q = (i/4)[J12^2, J23^2] (derived: [J12^2,J23^2] = 4[J1.J2, J2.J3] = -4i Q).
    J12^2 = diag k(k+1) in the basis ((j1 j2)k j3) j4; J23^2 = W diag l(l+1) W^T with
    W_kl = (-1)^(j1+j2+j3+j4) sqrt((2k+1)(2l+1)) {j1 j2 k; j3 j4 l}  (Edmonds 6.1.5),
    the 6j symbol from the Racah formula evaluated EXACTLY (integers/rationals), then
    converted to 60-digit mpmath.  Used for large spins.
(c) Closed form a_k^2 (Levy-Leblond & Levy-Nahas / Brunnemann-Thiemann (67)) implemented
    here from the formula, compared with (a) and (b).
"""
import itertools, math, sys, time
from fractions import Fraction as F
import numpy as np
import mpmath as mp

mp.mp.dps = 60
def mf(x):
    x = F(x); return mp.mpf(x.numerator)/x.denominator
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

# ---------------- closed form
def a2(j1, j2, j3, j4, k):
    return (((j1+j2+1)**2-k*k)*(k*k-(j1-j2)**2)*((j3+j4+1)**2-k*k)*(k*k-(j3-j4)**2)) / (16*(4*k*k-1))

def krange(j1, j2, j3, j4):
    return max(abs(j1-j2), abs(j3-j4)), min(j1+j2, j3+j4)

def chain2(js):
    j1, j2, j3, j4 = [F(x) for x in js]
    kmin, kmax = krange(j1, j2, j3, j4)
    d = int(kmax-kmin)+1
    return [a2(j1, j2, j3, j4, kmin+s) for s in range(1, d)], d

def closed_spectrum(js):
    c2, d = chain2(js)
    M = np.zeros((d, d))
    for i, v in enumerate(c2):
        M[i, i+1] = M[i+1, i] = math.sqrt(float(v))
    return np.sort(np.linalg.eigvalsh(M))

# ---------------- brute force
def spin(j2):
    j = j2/2; ms = [j-i for i in range(j2+1)]
    Jz = np.diag(ms).astype(complex)
    Jp = np.zeros((j2+1, j2+1), complex)
    for i in range(1, j2+1):
        m = ms[i]; Jp[i-1, i] = math.sqrt(j*(j+1)-m*(m+1))
    Jm = Jp.conj().T
    return [(Jp+Jm)/2, (Jp-Jm)/(2j), Jz]

def brute_spectrum(J2s):
    dims = [x+1 for x in J2s]
    emb = []
    for leg, x in enumerate(J2s):
        mats = spin(x); L = []
        for M in mats:
            T = np.ones((1, 1), complex)
            for l2, dd in enumerate(dims):
                T = np.kron(T, M if l2 == leg else np.eye(dd))
            L.append(T)
        emb.append(L)
    tot = [sum(emb[l][a] for l in range(4)) for a in range(3)]
    C = sum(t@t for t in tot)
    w, V = np.linalg.eigh(C)
    P = V[:, np.abs(w) < 1e-8]
    Q = (emb[0][0]@(emb[1][1]@emb[2][2]-emb[1][2]@emb[2][1])
         + emb[0][1]@(emb[1][2]@emb[2][0]-emb[1][0]@emb[2][2])
         + emb[0][2]@(emb[1][0]@emb[2][1]-emb[1][1]@emb[2][0]))
    Qr = P.conj().T@Q@P
    herm = np.abs(Qr-Qr.conj().T).max()
    return np.sort(np.linalg.eigvalsh((Qr+Qr.conj().T)/2)), P.shape[1], herm

t0 = time.time()
worst = 0.0; n = 0; dimmis = 0
for J2s in itertools.product(range(1, 5), repeat=4):
    if sum(J2s) % 2: continue
    js = [F(x, 2) for x in J2s]
    kmin, kmax = krange(*js)
    if kmax < kmin: continue
    ev, dd, herm = brute_spectrum(J2s)
    ref = closed_spectrum(js)
    if dd != len(ref): dimmis += 1; continue
    worst = max(worst, np.abs(ev-ref).max(), herm); n += 1
log(f"R1a brute force vs closed form: {n} ORDERED admissible tuples with 1<=2j_i<=4; dim mismatches {dimmis}; max |diff| {worst:.2e}; {time.time()-t0:.1f}s")
extra = [(5, 5, 5, 5), (1, 5, 5, 5), (2, 4, 5, 5), (6, 6, 5, 5), (1, 1, 6, 6), (3, 5, 6, 6), (6, 5, 4, 3), (6, 6, 6, 6)]
for J2s in extra:
    ev, dd, herm = brute_spectrum(J2s)
    ref = closed_spectrum([F(x, 2) for x in J2s])
    log(f"R1a extra {J2s}: dim {dd} vs {len(ref)}, max |diff| {np.abs(ev-ref).max():.2e}, smallest |q|>0 = {min(abs(e) for e in ev if abs(e)>1e-9) if dd>1 else None}")

# ---------------- exact 6j via Racah formula
from math import factorial as fa
def tri2(a, b, c):  # Delta^2 for doubled-free half-integers given as Fractions
    return F(fa(int(a+b-c))*fa(int(a-b+c))*fa(int(-a+b+c)), fa(int(a+b+c+1)))

def sixj(j1, j2, j3, j4, j5, j6):
    """{j1 j2 j3; j4 j5 j6} returned as mpf, computed from exact rationals."""
    tris = [(j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)]
    for a, b, c in tris:
        if a+b < c or a+c < b or b+c < a or (a+b+c) != int(a+b+c): return mp.mpf(0)
    D2 = F(1)
    for t in tris: D2 *= tri2(*t)
    s1, s2, s3, s4 = [int(sum(t)) for t in tris]
    p1, p2, p3 = int(j1+j2+j4+j5), int(j2+j3+j5+j6), int(j3+j1+j6+j4)
    S = 0
    for z in range(max(s1, s2, s3, s4), min(p1, p2, p3)+1):
        S += (-1)**z * F(fa(z+1), fa(z-s1)*fa(z-s2)*fa(z-s3)*fa(z-s4)*fa(p1-z)*fa(p2-z)*fa(p3-z))
    if S == 0: return mp.mpf(0)
    val = D2*S*S
    return mp.sign(S)*mp.sqrt(mp.mpf(val.numerator)/val.denominator)

def recoupling_Q(js):
    j1, j2, j3, j4 = [F(x) for x in js]
    kmin, kmax = max(abs(j1-j2), abs(j3-j4)), min(j1+j2, j3+j4)
    lmin, lmax = max(abs(j2-j3), abs(j1-j4)), min(j2+j3, j1+j4)
    ks = [kmin+i for i in range(int(kmax-kmin)+1)]
    ls = [lmin+i for i in range(int(lmax-lmin)+1)]
    assert len(ks) == len(ls)
    ph = (-1)**int(j1+j2+j3+j4)
    W = mp.matrix(len(ks), len(ls))
    for a, k in enumerate(ks):
        for b, l in enumerate(ls):
            W[a, b] = ph*mp.sqrt(mf((2*k+1)*(2*l+1)))*sixj(j1, j2, k, j3, j4, l)
    d = len(ks)
    orth = mp.norm(W*W.T - mp.eye(d), 1)
    Dl = mp.diag([mf(l*(l+1)) for l in ls])
    M = W*Dl*W.T
    # Q = (i/4)[Dk, M]:  Q_ab = (i/4)(k_a(k_a+1) - k_b(k_b+1)) M_ab  -> store real part R with Q = iR
    R = mp.matrix(d, d)
    for a in range(d):
        for b in range(d):
            R[a, b] = (mf(ks[a]*(ks[a]+1)) - mf(ks[b]*(ks[b]+1)))*M[a, b]/4
    return R, ks, orth

def check_recoupling(js, eig=True):
    t = time.time()
    R, ks, orth = recoupling_Q(js)
    d = len(ks)
    offtri = max([abs(R[a, b]) for a in range(d) for b in range(d) if abs(a-b) > 1] + [mp.mpf(0)])
    diag = max(abs(R[a, a]) for a in range(d))
    rel = mp.mpf(0)
    for s in range(1, d):
        ex = a2(*[F(x) for x in js], ks[s])
        got = R[s-1, s]**2
        rel = max(rel, abs(got/(mp.mpf(ex.numerator)/ex.denominator) - 1))
    msg = (f"R1b recoupling {tuple(str(x) for x in js)}: d={d}, |W W^T - I|={mp.nstr(orth,3)}, max|off-tridiag|={mp.nstr(offtri,3)}, "
           f"max|diag|={mp.nstr(diag,3)}, max rel err of Q_(k-1,k)^2 vs closed form a_k^2 = {mp.nstr(rel,3)}")
    if eig:
        # spectrum of Hermitian Q = iR (R real antisymmetric): eigenvalues of iR via float64 from the 60-digit matrix
        A = np.array([[float(R[a, b]) for b in range(d)] for a in range(d)])
        ev = np.sort(np.linalg.eigvalsh(1j*A))
        ref = closed_spectrum(js)
        msg += f", spectrum max|diff| vs closed form {np.abs(ev-ref).max():.2e}"
    log(msg + f" ({time.time()-t:.1f}s)")

for js in [(F(1,2),)*4, (1, 1, 1, 1), (F(3,2), 2, F(5,2), 3), (F(1,2), F(1,2), 10, 10), (5, 7, F(17,2), F(19,2)),
           (10, 10, 10, 10), (F(1,2), 15, 15, F(29,2)), (3, 20, 21, 30), (25, 25, 25, 25), (F(7,2), 40, F(81,2), 44)]:
    check_recoupling(js)
open(sys.argv[1] if len(sys.argv) > 1 else "r1_out.txt", "w").write("\n".join(out)+"\n")

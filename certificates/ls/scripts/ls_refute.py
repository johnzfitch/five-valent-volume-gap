"""Rigorous upper bounds refuting a linear-in-j_max lower bound (Theorems N1, N1', N2 of THEOREM.md).

N1  : j1 = 1/2 family G1 = (1/2; j-1/2, j, j, j).  [GD] Prop. 1 / Lemma 5 chain; closed forms of the zero mode;
      test vector x_i = (-1)^i z_i phi_i, phi_i = sin(pi ln i / (2 ln K)), K = 2j.
N1' : j1 = 1 family E1 = (1; j, j, j, j).  One-dimensional pencil (Proposition R of THEOREM.md);
      test function Phi_l = sin(pi ln l / (2 ln(n-1))).
N2  : half-integer j1: lambda(T) <= sigma_min(wall chain) when the zigzag block is injective; evaluated on
      A_s = (s; j, j, j-1/2, j) with an explicit test vector.
Checks: closed forms in exact rationals; chain spectrum = q from ls_core; test-vector bound >= true q;
explicit analytic bounds >= test-vector values.
Usage: python ls_refute.py > ../outputs/refute_out.txt
"""
import sys, os, math, warnings
from fractions import Fraction as Fr
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ls_core as L


# ---------------- N1 -----------------
def g1_links(j2):
    """doubled j2 = 2j. Returns a[1..N-1] (a[k] joins chain sites k, k+1), N = 4j sites, via [GD] Lemma 5."""
    j = Fr(j2, 2)
    nX, mX = 4 * j + 1, 1          # doubled-unit (l = 2k) X pair (j-1/2, j)
    nD, mD = 4 * j + 2, 0          # D pair (j, j)
    m, n = 1, 4 * j + 1
    a = {}
    for l in range(m + 1, int(n)):
        if l % 2 == 0:
            c2 = (nD ** 2 - l * l) * (l * l - mD ** 2) / 4
        else:
            c2 = (nX ** 2 - l * l) * (l * l - mX ** 2) / 4
        a[l - m] = Fr(c2)
    return a, int(4 * j)


def g1_check_closed_forms(j2):
    a, N = g1_links(j2)
    j = Fr(j2, 2)
    ok = True
    for i in range(1, int(2 * j) + 1):
        A_odd = a.get(2 * i - 1)            # a_{2i-1} = c_{2i}
        if A_odd != 4 * (2 * j + 1 - i) * (2 * j + 1 + i) * i * i:
            ok = False
        if i >= 2 and a[2 * i - 2] != 4 * (2 * j + 1 - i) * (2 * j + i) * (i - 1) * i:
            ok = False
    # zero-mode ratio
    z2 = Fr(1)
    for i in range(2, int(2 * j) + 1):
        z2 = z2 * a[2 * i - 2] / a[2 * i - 1]
        if z2 != Fr(2 * j + 2) / (i * (2 * j + 1 + i)):
            ok = False
    return ok


def g1_chain_qmin(j2):
    a, N = g1_links(j2)
    K = N // 2
    B = np.zeros((K, K))
    for i in range(1, K + 1):
        B[i - 1, i - 1] = math.sqrt(a[2 * i - 1])
        if i >= 2:
            B[i - 1, i - 2] = math.sqrt(a[2 * i - 2])
    return float(np.linalg.svd(B, compute_uv=False).min())


def g1_testbound(j2):
    j = j2 / 2; K = j2
    Lam = math.log(K)
    i = np.arange(1, K + 1, dtype=float)
    phi = np.sin(math.pi * np.log(i) / (2 * Lam))
    num = 4 * (2 * j) * (2 * j + 2) * phi[0] ** 2
    num += np.sum(4 * (2 * j + 2) * (2 * j + 1 - i[1:]) * i[1:] * (phi[1:] - phi[:-1]) ** 2)
    den = np.sum((2 * j + 2) * phi ** 2 / (i * (2 * j + 1 + i)))
    return math.sqrt(num / den)


def g1_analytic(j2):
    j = j2 / 2; Lam = math.log(j2)
    return 2 * math.pi * math.sqrt(2 * j * (4 * j + 1) * (Lam + math.pi ** 2 / 6) / Lam ** 3)


# ---------------- N1' -----------------
def e1_test(j2):
    n = j2 + 1                       # n = 2j+1, identical pairs (j, j), m = 0
    ells = np.arange(1, n, dtype=float)          # sites l = 1..n-1 (corner removes l = 0)
    Lam = math.log(n - 1)
    Phi = np.sin(math.pi * np.log(ells) / (2 * Lam))
    e = ells                                     # edge e joins l = e-1 and e; e=1 joins ghost (Phi_0 = 0)
    A = e * (n * n - e * e) / 2.0
    Pm = np.concatenate([[0.0], Phi])
    E = np.sum(A * (Pm[1:] - Pm[:-1]) ** 2)       # edge e = n has A = 0
    mu = (2 * ells + 1) / (ells * (ells + 1))
    return 8 * math.sqrt(E / np.sum(mu * Phi ** 2))


def e1_analytic(j2):
    n = j2 + 1; Lam = math.log(n - 1)
    return 8 * math.pi * n * math.sqrt((math.log(n) + math.pi ** 2 / 6) / (2 * Lam ** 3))


# ---------------- N2 (wall chain) -----------------
def wall_chain(tj):
    """rows d = s-1, columns d = s of the zigzag block; returns the dense block (rows x cols) and the full
    zigzag block B (rows d+s even x columns d+s odd)."""
    s2 = tj[0]
    Bd, Cd, lo, up, w = L.links(tj)
    dd = Bd - Cd
    isrow = ((dd + s2) // 2) % 2 == 0
    r = np.where(isrow)[0]; c = np.where(~isrow)[0]
    pr = {x: i for i, x in enumerate(r)}; pc = {x: i for i, x in enumerate(c)}
    M = np.zeros((len(r), len(c)))
    for a, b, x in zip(lo, up, w):
        if isrow[a]:
            M[pr[a], pc[b]] = x
        else:
            M[pr[b], pc[a]] = x
    W = M[np.ix_(dd[r] == s2 - 2, dd[c] == s2)]
    return M, W


def wall_testbound(W):
    """explicit test vector on the wall chain: alternating zero mode times sin(pi ln k/(2 ln K)) envelope,
    cut where the zero mode stops decaying; returns ||W x|| / ||x||."""
    nr, nc = W.shape
    # W is bidiagonal: row t joins columns t-1 and t (tall: one extra row at the bottom)
    # find for each row the two column entries
    z = np.zeros(nc); z[0] = 1.0
    for k in range(1, nc):
        # row k joins col k-1 (entry W[k,k-1]) and col k (entry W[k,k]); zero it
        z[k] = -W[k, k - 1] * z[k - 1] / W[k, k]
    K = max(2, nc // 2)
    kk = np.arange(1, nc + 1, dtype=float)
    env = np.where(kk <= K, np.sin(math.pi * np.log(kk) / (2 * math.log(K)) + 0.0), 1.0)
    env = np.where(kk <= K, env, np.cos(math.pi * (kk - K) / (2 * (nc - K + 1))))
    x = z * env
    return float(np.linalg.norm(W @ x) / np.linalg.norm(x))


if __name__ == '__main__':
    print("N1: G1 = (1/2; j-1/2, j, j, j).  closed forms exact; q_chain = q(T); test bound; analytic bound")
    for j2 in (2, 3, 4, 8, 16):
        print(f"  closed forms exact for 2j={j2}: {g1_check_closed_forms(j2)}")
    for j2 in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        tj = (1, j2 - 1, j2, j2, j2)
        q = float(L.qmin(tj)[0][0])
        qc = g1_chain_qmin(j2) if j2 <= 2048 else float('nan')
        tb = g1_testbound(j2); an = g1_analytic(j2)
        M = j2
        print(f"  2j={j2:6d}: q={q:12.4f} chain={qc:12.4f} test={tb:12.4f} analytic={an:12.4f}"
              f"  test/q={tb / q:.4f}  q lnM/M={q * math.log(M) / M:.4f}  test lnM/M={tb * math.log(M) / M:.4f}"
              f"  ok={q <= tb * (1 + 1e-12) and tb <= an}")
    print("N1': E1 = (1; j, j, j, j).  test bound (8 sqrt(Rayleigh)) and analytic bound")
    for j2 in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        tj = (2, j2, j2, j2, j2)
        q = float(L.qmin(tj)[0][0])
        tb = e1_test(j2); an = e1_analytic(j2)
        M = j2
        print(f"  2j={j2:6d}: q={q:12.4f} test={tb:12.4f} analytic={an:12.4f}  test/q={tb / q:.4f}"
              f"  q lnM/M={q * math.log(M) / M:.4f}  ok={q <= tb * (1 + 1e-12) and tb <= an}")
    print("N2: wall chain, A_s = (s; j, j, j-1/2, j) in pairing (j,j | j-1/2, j)")
    for s2 in (5, 7, 9, 11):
        for j2 in (16, 32, 64, 128, 256, 512, 1024, 2048):
            tj = (s2, j2, j2, j2 - 1, j2)
            if L.dim_inv(tj) > 60000:
                continue
            q = float(L.qmin(tj)[0][0])
            M_, W = wall_chain(tj)
            # injectivity of the full zigzag block on the column side
            if M_.shape[1] <= 3000:
                sv = np.linalg.svd(M_, compute_uv=False)
                inj = bool((sv > 1e-9 * sv.max()).sum() == M_.shape[1])
            else:
                inj = 'n/a'
            sw = float(np.linalg.svd(W, compute_uv=False).min())
            tb = wall_testbound(W)
            print(f"  2s={s2} 2j={j2:5d}: q={q:10.3f}  8 smin(wall)={8 * sw:10.3f}  8 test={8 * tb:10.3f}"
                  f"  col-injective={inj}  wall shape={W.shape}  8test lnM/M={8 * tb * math.log(j2) / j2:.3f}"
                  f"  q<=8smin: {q <= 8 * sw * (1 + 1e-9)}")

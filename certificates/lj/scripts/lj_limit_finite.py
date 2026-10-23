"""Finite part of the three-bounded-legs theorem: for each certified case (t, M) of limit_cert_K{K}.json,
every admissible s < s0 is either inside the finite certificate range (2j_max <= 38, research/dichotomy_finite_cert_20261021)
or is certified here: structure stable (s >= s_min_struct) and interval Cholesky shows
Y(s)^T Y(s) - tau I > 0 with tau = (target/8)^2, Y(s) the exact bipartite block from the smaller colour class
(entries sqrt of exact rationals, mpmath.iv).  target = max(4.5579607, 8*sigma_K*s - C_K) is checked for both
the dichotomy constant and the linear law.  Usage: python lj_limit_finite.py K"""
import sys, os, json, numpy as np
from fractions import Fraction as Fr
from mpmath import iv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lj_limit_core as LC
import lj_core as LCORE
import warnings; warnings.filterwarnings('ignore')


def structure_matches(t2, M2, s):
    """compare the limit-formula site set and link weights with the direct builder at this s"""
    tj = (int(2 * s), t2[0], t2[1], t2[2], int(2 * s) - M2)
    o = LCORE.build(tj)
    S, Ls = LC.links(t2, M2)
    mp = {(Fr(int(b), 2), Fr(int(c), 2) - s): i for i, (b, c) in enumerate(zip(o['Bd'], o['Cd']))}
    if set(mp) != set(S):
        return False
    W = {frozenset((int(a), int(b))): w for a, b, w in zip(o['lo'], o['up'], o['w']) if w > 0}
    Wf = {frozenset((mp[S[l[0]]], mp[S[l[1]]])): LC.w_exact(l, float(s), t2, M2) for l in Ls}
    return set(W) == set(Wf) and all(abs(W[k] - Wf[k]) <= 1e-12 * W[k] for k in W)
from lj_limit_cert import chol_pos, s_min_struct, analyse
iv.dps = 40
QSTAR = Fr(45579607, 10 ** 7)


def Y_iv(t2, M2, s):
    S, Ls = LC.links(t2, M2)
    col = [int((b + g) % 2) for (b, g) in S]
    E_ = [i for i in range(len(S)) if col[i] == 0]; O_ = [i for i in range(len(S)) if col[i] == 1]
    small, large = (O_, E_) if len(O_) <= len(E_) else (E_, O_)
    ps = {v: k for k, v in enumerate(small)}; pl = {v: k for k, v in enumerate(large)}
    Y = [[iv.mpf(0) for _ in small] for _ in large]
    sv = iv.mpf(s.numerator) / s.denominator
    for lk in Ls:
        a, b = lk[0], lk[1]
        if a in ps: a, b = b, a
        d = lk[3]; c2 = iv.mpf(d['c2'].numerator) / d['c2'].denominator
        if lk[2] == 'X':
            val = iv.sqrt(c2) / 4 * iv.sqrt((2 * sv + 1 + iv.mpf(float(d['g']))) ** 2 - iv.mpf(float(d['b'])) ** 2)
        else:
            N = iv.mpf(1)
            for a_ in d['al']:
                N = N * (2 * sv + iv.mpf(a_.numerator) / a_.denominator)
            g = iv.mpf(d['g'].numerator) / d['g'].denominator
            val = iv.sqrt(c2) / 4 * iv.sqrt(N / (4 * (sv + g) ** 2 - 1))
        Y[pl[a]][ps[b]] = val
    return Y, len(small)


def certify(Y, n, target):
    tau = (iv.mpf(target.numerator) / target.denominator / 8) ** 2
    G = [[sum((Y[r][i] * Y[r][j] for r in range(len(Y))), iv.mpf(0)) - (tau if i == j else 0) for j in range(n)] for i in range(n)]
    return chol_pos(G)


if __name__ == '__main__':
    K = int(sys.argv[1])
    here = os.path.dirname(os.path.abspath(__file__))
    cases = json.load(open(os.path.join(here, '..', 'outputs', f'limit_cert_K{K}.json')))
    sigK = min(Fr(c['sigma_inf']) for c in cases)
    CK = max(max(8 * Fr(c['E_s0']) for c in cases), max(8 * sigK * Fr(c['s0']) - QSTAR for c in cases))
    CK = Fr(int(CK) + 1)
    print(f"K={K}: sigma_K = {float(sigK):.8f}, C_K = {CK}")
    n_fc = n_dir = n_fail = 0
    for c in cases:
        t2 = tuple(c['assign']); M2 = c['M2']; s0 = Fr(c['s0'])
        s = Fr(1, 2)
        while s < s0:
            j5 = s - Fr(M2, 2)
            if j5 >= Fr(1, 2):
                if 2 * max(s, j5, Fr(max(t2), 2)) <= 38:
                    n_fc += 1
                elif structure_matches(t2, M2, s):
                    Y, n = Y_iv(t2, M2, s)
                    tgt = max(QSTAR, 8 * sigK * s - CK)
                    if n and not certify(Y, n, tgt):
                        n_fail += 1; print("FAIL", t2, M2, s)
                    else:
                        n_dir += 1
                else:
                    n_fail += 1; print("UNSTABLE STRUCTURE OUTSIDE CERT RANGE", t2, M2, s)
            s += Fr(1, 2)
    print(f"finite part: {n_fc} tuples inside the 2j_max<=38 certificate; {n_dir} certified directly here; failures {n_fail}")
    print(f"Linear law on the class: q >= {8*float(sigK):.6f} j1 - {CK} for all tuples with three of legs 2-5 at 2j <= {K}")

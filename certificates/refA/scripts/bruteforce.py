"""Brute-force checks on the tensor product (own code):
 (B1) |spec T| on Inv(j1..j5) equals |spec W| of the closed-form nonnegative grid ([CL] sec. 1 + Lemma 2) for every
      valid doubled tuple with entries <= MB;
 (B2) Prop. 1(b),(c): on Inv, T = -E and T = -i[A,B] (A = J2.J3 - J4.J5, B = J3.J4 - J2.J5), on the same tuples;
 (B3) Prop. 1(d): |spec W| (closed form) is invariant under all 24 permutations of legs 2-5, on random tuples <= 16;
 (B4) Lemma 2: sign of the 6j symbol {s c b; 1 b-1 c} equals (-1)^(s+b+c) (sympy, exact) for 2s,2b,2c <= MS.
Inv is computed as ker J+ on the M = 0 subspace (sparse)."""
import sys, itertools, random, time
import numpy as np
import scipy.sparse as sp
from fractions import Fraction as Fr
from refcore import links_closed_form, region, valid


def spin_ops(tj):
    d = tj + 1; j = tj / 2; m = j - np.arange(d)
    jp = np.zeros((d, d))
    for k in range(1, d):
        jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    return sp.csr_matrix(jp), sp.csr_matrix(np.diag(m))


def leg_vec_ops(tjs):
    dims = [t + 1 for t in tjs]; out = []
    for i, t in enumerate(tjs):
        jp, jz = spin_ops(t)
        def emb(M):
            R = sp.identity(1, format='csr')
            for k, dk in enumerate(dims):
                R = sp.kron(R, M if k == i else sp.identity(dk, format='csr'), format='csr')
            return R
        Jp = emb(jp); Jz = emb(jz); Jm = Jp.T.tocsr()
        Jx = (Jp + Jm) * 0.5; Jy = (Jp - Jm) * (-0.5j)
        out.append((Jx.astype(complex), Jy.astype(complex), Jz.astype(complex), Jp))
    return out, dims


def mvals(tjs):
    ms = [np.arange(t, -t - 1, -2) for t in tjs]   # doubled m
    tot = np.zeros(1, dtype=int)
    for m in ms:
        tot = (tot[:, None] + m[None, :]).ravel()
    return tot


def inv_basis(ops, tjs):
    tot = mvals(tjs)
    i0 = np.where(tot == 0)[0]; i1 = np.where(tot == 2)[0]
    Jp = sum(o[3] for o in ops)
    Jp01 = Jp[i1][:, i0].toarray()
    if Jp01.shape[0] == 0:
        V = np.eye(len(i0))
    else:
        u_, s_, vt = np.linalg.svd(Jp01)
        rank = int(np.sum(s_ > 1e-9))
        V = vt[rank:].T
    full = np.zeros((len(tot), V.shape[1])); full[i0] = V
    return full


def triple(a, b, c):
    eps = [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]
    return sum(s * (a[i] @ b[j] @ c[k]) for i, j, k, s in eps)


def dot(a, b):
    return sum(a[i] @ b[i] for i in range(3))


def closed_abs_spec(tj):
    s, _, _, rect, R = region(tj)
    idx = {p: i for i, p in enumerate(R)}
    W = np.zeros((len(R), len(R)))
    for (p, q), w2 in links_closed_form(tj).items():
        W[idx[p], idx[q]] = W[idx[q], idx[p]] = np.sqrt(float(w2))
    return np.sort(np.abs(np.linalg.eigvalsh(W))) if len(R) else np.zeros(0)


if __name__ == '__main__':
    MB = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    MS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    t0 = time.time(); n = 0; worst = 0.0; worst2 = 0.0
    for tj in itertools.product(range(1, MB + 1), repeat=5):
        if not valid(tj): continue
        ops, dims = leg_vec_ops(tj)
        V = inv_basis(ops, tj)
        if V.shape[1] == 0: continue
        J1, J2, J3, J4, J5 = [o[:3] for o in ops]
        X23 = triple(J1, J2, J3); X45 = triple(J1, J4, J5)
        Tm = V.conj().T @ (X23 + X45) @ V
        ev = np.sort(np.abs(np.linalg.eigvalsh((Tm + Tm.conj().T) / 2)))
        cs = closed_abs_spec(tj)
        assert len(cs) == len(ev), (tj, len(cs), len(ev))
        worst = max(worst, float(np.max(np.abs(cs - ev))) if len(ev) else 0.0)
        # Prop 1(b),(c)
        E = triple(J2, J3, J4) + triple(J2, J3, J5) + triple(J2, J4, J5) + triple(J3, J4, J5)
        Aop = dot(J2, J3) - dot(J4, J5); Bop = dot(J3, J4) - dot(J2, J5)
        Em = V.conj().T @ E @ V
        Cm = V.conj().T @ (-1j * (Aop @ Bop - Bop @ Aop)) @ V
        worst2 = max(worst2, float(np.max(np.abs(Tm + Em))), float(np.max(np.abs(Tm - Cm))))
        n += 1
    print(f"(B1) brute force vs closed form, all valid doubled tuples <= {MB}: {n} tuples with Inv != 0; "
          f"max | |spec T| - |spec W| | = {worst:.2e}")
    print(f"(B2) on Inv: max |T + E|, |T + i[A,B]| entries = {worst2:.2e}  ({time.time()-t0:.0f}s)")
    random.seed(5); m = 0; w3 = 0.0
    while m < 300:
        tj = tuple(random.randint(1, 16) for _ in range(5))
        if not valid(tj): continue
        base = closed_abs_spec(tj)
        for p in itertools.permutations(tj[1:]):
            other = closed_abs_spec((tj[0],) + p)
            assert len(other) == len(base)
            if len(base): w3 = max(w3, float(np.max(np.abs(other - base))))
        m += 1
    print(f"(B3) 300 random tuples <= 16, all 24 permutations of legs 2-5: max | |spec| difference | = {w3:.2e}")
    from sympy.physics.wigner import wigner_6j
    from sympy import Rational as Q
    cnt = 0; bad = 0
    for S2 in range(0, MS + 1):
        for B2 in range(2, MS + 1):
            for C2 in range(0, MS + 1):
                s, b, c = Q(S2, 2), Q(B2, 2), Q(C2, 2)
                if (S2 + B2 + C2) % 2: continue
                if not (abs(b - c) <= s <= b + c): continue
                v = wigner_6j(s, c, b, 1, b - 1, c)
                if v == 0: continue
                cnt += 1
                sign = 1 if v > 0 else -1
                if sign != (-1) ** int(s + b + c): bad += 1
    print(f"(B4) 6j sign {{s c b; 1 b-1 c}} = (-1)^(s+b+c): {cnt} nonzero cases with 2s,2b,2c <= {MS}, {bad} exceptions")

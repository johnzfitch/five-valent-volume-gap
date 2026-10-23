"""Validate ref_core against a brute-force tensor-product construction.

For every ordered 5-tuple with 2j_i <= JMAX (and the dims below a cap), build
J_I on the full tensor product, take the invariant subspace as the kernel of
J_tot^2, form q_T = [(J_I+J_J)^2,(J_J+J_K)^2] there, and compare basis-free
invariants with ref_core: spectra of each i q_T, of Q_RS = sum |q_T|, and of a
fixed random real combination sum c_T q_T (sensitive to relative phases).
"""
import sys, itertools
import numpy as np
from ref_core import q_triples, absq, dimInv


def spin_mats(tj):
    j = tj / 2
    m = np.arange(j, -j - 1, -1)
    d = len(m)
    Jz = np.diag(m)
    Jp = np.zeros((d, d))
    for k in range(1, d):
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    Jx = (Jp + Jp.T) / 2
    Jy = (Jp - Jp.T) / (2j)
    return [Jx, Jy, Jz]


def brute(js5):
    dims = [tj + 1 for tj in js5]
    N = int(np.prod(dims))
    Js = []
    for I in range(5):
        S = spin_mats(js5[I])
        comp = []
        for c in range(3):
            mats = [np.eye(d) for d in dims]
            mats[I] = S[c]
            M = mats[0]
            for X in mats[1:]:
                M = np.kron(M, X)
            comp.append(M)
        Js.append(comp)
    Jt = [sum(Js[I][c] for I in range(5)) for c in range(3)]
    J2 = sum(Jt[c] @ Jt[c] for c in range(3))
    w, U = np.linalg.eigh(J2)
    W = U[:, np.abs(w) < 1e-8]
    def P(I, J):
        return sum((Js[I][c] + Js[J][c]) @ (Js[I][c] + Js[J][c]) for c in range(3))
    qs = {}
    for (I, J, K) in itertools.combinations(range(5), 3):
        A = W.conj().T @ P(I, J) @ W; B = W.conj().T @ P(J, K) @ W
        qs[(I, J, K)] = A @ B - B @ A
    return qs, W.shape[1]


def main(JMAX=3, cap=700):
    rng = np.random.default_rng(1)
    coef = {T: rng.normal() for T in itertools.combinations(range(5), 3)}
    worst = 0.0; n = 0; nd = 0
    for js5 in itertools.product(range(1, JMAX + 1), repeat=5):
        if sum(js5) % 2:
            continue
        if np.prod([t + 1 for t in js5]) > cap ** 1 * 4:
            continue
        d = dimInv(js5)
        if d == 0:
            continue
        qb, db = brute(js5)
        qr, _ = q_triples(js5)
        assert db == d, (js5, db, d)
        def specs(qs):
            out = []
            for T in sorted(qs):
                out.append(np.sort(np.linalg.eigvalsh(1j * qs[T])))
            Q = sum(np.abs(np.linalg.eigvalsh(1j * qs[T])).sum() * 0 + 0 for T in qs)
            QRS = sum(absq(qs[T])[0] for T in qs)
            out.append(np.sort(np.linalg.eigvalsh(QRS)))
            C = sum(coef[T] * qs[T] for T in qs)
            out.append(np.sort(np.linalg.eigvalsh(1j * C)))
            return out
        sb = specs(qb); sr = specs(qr)
        e = max(np.max(np.abs(x - y)) for x, y in zip(sb, sr))
        worst = max(worst, e); n += 1
        if d >= 2:
            nd += 1
    print(f"ordered tuples checked: {n} (dim>=2: {nd}); 2j_i <= {JMAX}; max spectral discrepancy {worst:.3e}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)

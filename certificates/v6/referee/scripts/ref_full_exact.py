"""Referee check, valence 6: exact full-operator computation, written independently.

Nothing is imported from the author's scripts or from hv_core.

Method. Work in the rational frame J+|m> = |m+1>, J-|m> = (j+m)(j-m+1)|m-1>,
Jz|m> = m|m>, which is a diagonal similarity of the standard frame, so spectra are
unchanged and every matrix entry is an integer.  D_IJ := 4 J_I.J_J is an integer
matrix.  q_IJK := [(J_I+J_J)^2, (J_J+J_K)^2] = (1/4)[D_IJ, D_JK].

(a) sigma-reduced operator: A = sum_{I<J<K<=5} sigma(IJK) q_IJK acting on the
    highest-weight space H of total spin j6 inside V1 x ... x V5 (M = j6, ker J+).
    H is isomorphic to Inv(j1..j6) and the operator is the restriction of the full one.
(b) full operator (small n only): A = sum_{I<J<K<=6} eps(IJK) q_IJK on
    Inv(j1..j6) = ker J+ in the M = 0 sector of V1 x ... x V6.

The characteristic polynomial is computed exactly over Q (python-flint).
Eigenvalues are +-i q; kernel dim = multiplicity of lambda = 0; q_min from rigorous
real-root isolation of R(t) = prod (t - q^2).

usage: python -I ref_full_exact.py NMAX_FULL NMAX_SIGMA out.txt
"""
import sys, itertools, time
from fractions import Fraction
import flint

TANG = [(-1, 2, -3), (-2, 4, 1), (2, -1, -4), (-3, 1, 4), (-4, 3, 3), (-2, -3, 2)]


def det3(a, b, c):
    return (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0])
            + a[2]*(b[0]*c[1]-b[1]*c[0]))


def eps_table():
    eps = {}
    for t in itertools.combinations(range(6), 3):
        d = det3(*(TANG[i] for i in t))
        eps[t] = (d > 0) - (d < 0)
    return eps


def sector(tj, M2):
    """tuples of twice-m with given twice-sum"""
    out = []
    def rec(k, acc, cur):
        if k == len(tj):
            if acc == M2:
                out.append(tuple(cur))
            return
        rest = sum(tj[k+1:])
        for m in range(-tj[k], tj[k]+1, 2):
            if abs(M2 - acc - m) <= rest:
                cur.append(m); rec(k+1, acc+m, cur); cur.pop()
    rec(0, 0, [])
    return out


def Dop(tj, I, J, basis, index):
    """sparse integer matrix of 4 J_I.J_J on a fixed-M sector: dict col -> list of (row, val)"""
    cols = {}
    for c, st in enumerate(basis):
        mi, mj = st[I], st[J]
        ent = {c: mi*mj}  # 4 m_I m_J with twice-m: (2mI)(2mJ)
        # 2(J+_I J-_J + J-_I J+_J)
        if mi + 2 <= tj[I] and mj - 2 >= -tj[J]:
            # J-_J |mj> = (j+m)(j-m+1) |mj-1>, with twice values: (tj+mj)/2 * (tj-mj+2)/2
            f = ((tj[J]+mj)//2) * ((tj[J]-mj+2)//2)
            ns = list(st); ns[I] += 2; ns[J] -= 2
            r = index[tuple(ns)]
            ent[r] = ent.get(r, 0) + 2*f
        if mj + 2 <= tj[J] and mi - 2 >= -tj[I]:
            f = ((tj[I]+mi)//2) * ((tj[I]-mi+2)//2)
            ns = list(st); ns[I] -= 2; ns[J] += 2
            r = index[tuple(ns)]
            ent[r] = ent.get(r, 0) + 2*f
        cols[c] = [(r, v) for r, v in ent.items() if v]
    return cols


def apply(op, Mcols):
    """op (sparse cols) times dense matrix given as list of columns (lists of ints)"""
    n = len(Mcols[0])
    out = []
    for col in Mcols:
        res = [0]*n
        for c, v in enumerate(col):
            if v:
                for r, a in op[c]:
                    res[r] += a*v
        out.append(res)
    return out


def sub(a, b):
    return [[x-y for x, y in zip(ca, cb)] for ca, cb in zip(a, b)]


def add(a, b, s=1):
    return [[x+s*y for x, y in zip(ca, cb)] for ca, cb in zip(a, b)]


def hw_space(tj, M2):
    S = sector(tj, M2)
    S1 = sector(tj, M2+2)
    idx = {s: i for i, s in enumerate(S)}
    idx1 = {s: i for i, s in enumerate(S1)}
    Jp = flint.fmpz_mat(len(S1), len(S))
    for c, st in enumerate(S):
        for k in range(len(tj)):
            if st[k]+2 <= tj[k]:
                ns = list(st); ns[k] += 2
                Jp[idx1[tuple(ns)], c] += 1
    X, nul = Jp.nullspace()
    cols = [[int(X[r, j]) for r in range(len(S))] for j in range(nul)]
    return S, idx, cols


def restrict(EN, N):
    """matrix C with E N = N C, exact"""
    d = len(N)
    Nt = flint.fmpz_mat([list(c) for c in N])  # d x |S|
    R, _den, rk = Nt.rref()
    assert rk == d
    piv = []
    for i in range(d):
        for j in range(R.ncols()):
            if R[i, j] != 0:
                piv.append(j); break
    NR = flint.fmpq_mat(d, d, [N[j][r] for r in piv for j in range(d)])
    ER = flint.fmpq_mat(d, d, [EN[j][r] for r in piv for j in range(d)])
    return NR.solve(ER)


def commut_apply(DI, DJ, N):
    return sub(apply(DI, apply(DJ, N)), apply(DJ, apply(DI, N)))


def spectrum_data(C):
    """C has eigenvalues +- i*2q (E = 2A). Return kernel dim, list of exact poly for t=(2q)^2."""
    P = C.charpoly()  # fmpq_poly in lambda
    coeffs = [P[k] for k in range(P.degree()+1)]
    odd = all(coeffs[k] == 0 for k in range(1, len(coeffs), 2))
    kdim = 0
    while coeffs[kdim] == 0:
        kdim += 1
    # R(t): lambda^2 = -t
    ev = coeffs[0::2]
    Rt = [c*(-1)**m for m, c in enumerate(ev)]
    while Rt[0] == 0:
        Rt = Rt[1:]
    Rz = flint.fmpq_poly(Rt).numer()  # clear denominators, exact
    return odd, kdim, Rz


def qmin_from(Rz, prec_bits):
    flint.ctx.prec = prec_bits
    roots = Rz.complex_roots()
    re = []
    for r, mult in roots:
        assert r.imag.contains(0) and mult == 1, (r, mult)
        assert r.real > 0
        re.append(r.real)
    # all roots t = (2q)^2 > 0; pick the smallest (arb comparisons are rigorous)
    tmin = re[0]
    for a in re[1:]:
        if a < tmin:
            tmin = a
    q = tmin.sqrt() / 2  # (2q)^2 = t  -> q = sqrt(t)/2   (eigenvalue of A is i q, of E is 2 i q)
    return q, len(re)


def run_sigma(n, out, prec_bits):
    tj = [1, 1, n, n, n+1]
    M2 = n+1  # twice j6
    t0 = time.time()
    S, idx, N = hw_space(tj, M2)
    d = len(N)
    D12 = Dop(tj, 0, 1, S, idx); D23 = Dop(tj, 1, 2, S, idx)
    D24 = Dop(tj, 1, 3, S, idx); D45 = Dop(tj, 3, 4, S, idx)
    EN = sub(commut_apply(D12, D23, N), commut_apply(D24, D45, N))  # E = [D12,D23]-[D24,D45], A = E/2
    C = restrict(EN, N)
    odd, kdim, Rz = spectrum_data(C)
    line = f"sigma-reduced n={n}: |S|={len(S)} dim H={d} (4n+2={4*n+2}) char poly even: {odd} kernel dim={kdim}"
    if kdim < d:
        q, nr = qmin_from(Rz, prec_bits)
        line += f" distinct roots={nr} q_min = {q.str(15)}"
    line += f"  [{time.time()-t0:.1f}s]"
    print(line); out.write(line+"\n"); out.flush()
    return C, Rz, kdim


def run_full(n, out, sigma_Rz, sigma_k):
    eps = eps_table()
    tj = [1, 1, n, n, n+1, n+1]
    S, idx, N = hw_space(tj, 0)
    D = {}
    for I in range(6):
        for J in range(I+1, 6):
            D[(I, J)] = Dop(tj, I, J, S, idx)
    EN = [[0]*len(S) for _ in N]
    for (I, J, K), e in eps.items():
        if e == 0:
            continue
        # q_IJK = 1/4 [D_IJ, D_JK]; sum eps q = (1/4) sum eps [ , ]; E := 2*sum = (1/2) sum eps [D_IJ,D_JK]
        term = commut_apply(D[(I, J)], D[(J, K)], N)
        EN = add(EN, term, e)
    # E_full := sum eps [D,D] = 4 A_full ; sigma version E = 2 A. Halve to compare: use 2A = E_full/2
    C = restrict(EN, N) * flint.fmpq(1, 2)
    odd, kdim, Rz = spectrum_data(C)
    same = (kdim == sigma_k) and (Rz == sigma_Rz or Rz == -sigma_Rz)
    # compare monic forms
    def monic(p):
        lc = p[p.degree()]
        return flint.fmpq_poly([flint.fmpq(p[k], lc) for k in range(p.degree()+1)])
    same = (kdim == sigma_k) and monic(Rz) == monic(sigma_Rz)
    line = f"full 20-triple eps operator n={n}: dim Inv={len(N)} kernel dim={kdim}; char poly identical to sigma-reduced: {same}"
    print(line); out.write(line+"\n"); out.flush()


def main():
    nfull = int(sys.argv[1]); nsig = int(sys.argv[2]); path = sys.argv[3]
    with open(path, "w", encoding="utf-8") as out:
        eps = eps_table()
        dets = {t: det3(*(TANG[i] for i in t)) for t in itertools.combinations(range(6), 3)}
        out.write(f"tangents {TANG}\n")
        out.write(f"min |det| over 20 triples = {min(abs(v) for v in dets.values())}; all nonzero: {all(dets.values())}\n")
        order = [(0,1,2),(0,1,3),(0,1,4),(0,2,3),(0,2,4),(0,3,4),(1,2,3),(1,2,4),(1,3,4),(2,3,4)]
        sig = []
        for (I, J, K) in order:
            sig.append(eps[(I,J,K)] - eps[(I,J,5)] + eps[(I,K,5)] - eps[(J,K,5)])
        out.write(f"sigma (BR order 123..345) = {tuple(sig)}\n")
        out.write("eps (lex order) = " + str([eps[t] for t in sorted(eps)]) + "\n")
        assert tuple(sig) == (2,0,0,0,0,0,0,0,-2,0)
        for n in range(1, nsig+1):
            prec = 64 + int(12*n*max(1, __import__('math').log2(n+1)))
            C, Rz, k = run_sigma(n, out, prec)
            if n <= nfull:
                run_full(n, out, Rz, k)


if __name__ == "__main__":
    main()

"""Referee check of Proposition L directly on the full (sigma-reduced, = full on Inv) operator.

Independent of the author's code. Uses only ref_full_exact.py (written by the referee).

In the rational frame every operator restricted to H (= Inv) is an exact rational matrix,
similar to the Hermitian one by a diagonal-times-basis change.  Joint eigenvectors of the
Casimirs c^2=(J1+J3)^2, e^2=(J4+J5)^2, g^2=(J1+J2+J3)^2 (basis B1) and c^2, e^2,
h^2=(J2+J4+J5)^2 (basis B2) are found exactly.  For any operator O and basis W of
(rescaled) orthonormal eigenvectors, M = W^-1 O W satisfies M_ab M_ba = |O_ab|^2 (O
Hermitian), and for the change of basis R = W1^-1 W2, R_ab (R^-1)_ba = |<a|b>|^2.  So all
squared magnitudes are obtained exactly, independent of phase conventions.

Checks, for each n:
 X = J1.(J2xJ3) in B1: only links (c-,g=n/2,e)-(c+,g=n/2,e), |X|^2 = n(n+2)/16.
 Y = J2.(J4xJ5) in B2: only links (c,h,e=h-1/2)-(c,h,e=h+1/2), |Y|^2 = y_h^2; zero rows.
 overlaps |<c,e,g=n/2|c,e,h>|^2 = alpha^2, beta^2 closed forms.
usage: python ref_propL.py NMAX out.txt
"""
import sys
from fractions import Fraction as F
import flint
from ref_full_exact import sector, Dop, hw_space, apply, sub, add, restrict, commut_apply


def restrict_op(opN, N):
    return restrict(opN, N)


def joint_eigvecs(mats_vals_list, d):
    """mats: list of fmpq_mat; vals: list of eigenvalue tuples (one per label). Returns columns."""
    pass


def nullvec(M):
    # M fmpq_mat; return a basis of its right nullspace (list of fmpq lists)
    num, den = M.numer_denom() if hasattr(M, 'numer_denom') else (None, None)
    Z = num  # fmpz_mat scaled
    X, nul = Z.nullspace()
    return [[X[r, j] for r in range(Z.ncols())] for j in range(nul)]


def main():
    nmax = int(sys.argv[1]); path = sys.argv[2]
    neg = len(sys.argv) > 3 and sys.argv[3] == 'neg'  # negative control: wrong constants must fail
    out = open(path, "w", encoding="utf-8")
    allok = True
    for n in range(2, nmax+1):
        tj = [1, 1, n, n, n+1]
        S, idx, N = hw_space(tj, n+1)
        d = len(N)
        D = {}
        for (I, J) in [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (3, 4)]:
            D[(I, J)] = Dop(tj, I, J, S, idx)
        EX = restrict(commut_apply(D[(0, 1)], D[(1, 2)], N), N)   # X = i EX/16
        EY = restrict(commut_apply(D[(1, 3)], D[(3, 4)], N), N)   # Y = i EY/16
        Cc = restrict(apply(D[(0, 2)], N), N)
        Ce = restrict(apply(D[(3, 4)], N), N)
        Cg = restrict(add(add(apply(D[(0, 1)], N), apply(D[(0, 2)], N)), apply(D[(1, 2)], N)), N)
        Ch = restrict(add(add(apply(D[(1, 3)], N), apply(D[(1, 4)], N)), apply(D[(3, 4)], N)), N)
        j = [F(1, 2), F(1, 2), F(n, 2), F(n, 2), F(n+1, 2)]
        cas = lambda s: s*(s+1)
        I_ = flint.fmpq_mat(d, d, [1 if a == b else 0 for a in range(d) for b in range(d)])

        def ev_c(c): return 2*(cas(c)-cas(j[0])-cas(j[2]))
        def ev_e(e): return 2*(cas(e)-cas(j[3])-cas(j[4]))
        def ev_g(g): return 2*(cas(g)-cas(j[0])-cas(j[1])-cas(j[2]))
        def ev_h(h): return 2*(cas(h)-cas(j[1])-cas(j[3])-cas(j[4]))
        j6 = F(n+1, 2)
        cs = [F(n-1, 2), F(n+1, 2)]
        es = [F(2*i+1, 2) for i in range(n+1)]

        def tri(a, b, c):
            return abs(a-b) <= c <= a+b and (a+b+c).denominator == 1

        def build(labels, mats):
            cols = []
            for lab in labels:
                rows = []
                for Mt, v in zip(mats, lab[1]):
                    Ms = Mt - I_*flint.fmpq(v.numerator, v.denominator)
                    rows.append(Ms)
                # stack
                big = flint.fmpq_mat(d*len(rows), d, [rows[k][r, c] for k in range(len(rows)) for r in range(d) for c in range(d)])
                vs = nullvec(big)
                assert len(vs) == 1, (lab[0], len(vs))
                cols.append(vs[0])
            W = flint.fmpq_mat(d, d, [cols[c][r] for r in range(d) for c in range(d)])
            return W

        lab1 = []
        for c in cs:
            for e in es:
                for g in [c-F(1, 2), c+F(1, 2)]:
                    if g >= 0 and tri(g, e, j6):
                        lab1.append(((c, e, g), (ev_c(c), ev_e(e), ev_g(g))))
        lab2 = []
        for c in cs:
            for e in es:
                for h in [e-F(1, 2), e+F(1, 2)]:
                    if h >= 0 and tri(c, h, j6):
                        lab2.append(((c, e, h), (ev_c(c), ev_e(e), ev_h(h))))
        ok_dim = (len(lab1) == d and len(lab2) == d)
        W1 = build(lab1, [Cc, Ce, Cg]); W2 = build(lab2, [Cc, Ce, Ch])
        W1i = W1.inv(); W2i = W2.inv()
        MX = W1i*EX*W1
        MY = W2i*EY*W2
        x2 = F(n*(n+2), 15 if neg else 16)
        okX = True
        for a in range(d):
            for b in range(d):
                v = -MX[a, b]*MX[b, a]/256  # |X_ab|^2
                la, lb = lab1[a][0], lab1[b][0]
                link = (la[1] == lb[1] and la[2] == lb[2] == F(n, 2) and la[0] != lb[0])
                want = flint.fmpq(x2.numerator, x2.denominator) if link else 0
                if v != want:
                    okX = False
        okY = True
        zero_rows = []
        for a in range(d):
            if all(MY[a, b] == 0 for b in range(d)):
                zero_rows.append(lab2[a][0])
            for b in range(d):
                v = -MY[a, b]*MY[b, a]/256
                la, lb = lab2[a][0], lab2[b][0]
                link = (la[0] == lb[0] and la[2] == lb[2] and abs(la[1]-lb[1]) == 1)
                if link:
                    h = la[2]
                    y2 = F(1, 16)*(n+1-h)*(n+2+h)*h*(h+1)*(F(17,16) if neg else 1)
                    want = flint.fmpq(y2.numerator, y2.denominator)
                else:
                    want = 0
                if v != want:
                    okY = False
        want_zero = {(cs[1], F(1, 2), F(0)), (cs[1], F(2*n+1, 2), F(n+1))}
        okZ = set(zero_rows) == want_zero
        R = W1i*W2; Ri = W2i*W1
        okR = True
        for a in range(d):
            ca, ea, ga = lab1[a][0]
            for b in range(d):
                cb, eb, hb = lab2[b][0]
                v = R[a, b]*Ri[b, a]
                if (ca, ea) != (cb, eb):
                    if v != 0:
                        okR = False
                    continue
                if ga != F(n, 2):
                    continue
                i = ea - F(1, 2)
                L = (hb == ea - F(1, 2))
                if ca == cs[0]:
                    w = (n+2+i)*i/(n*(2*i+2)) if L else (n-i)*(i+2)/(n*(2*i+2))
                else:
                    w = F(n+1-i, 2*n+4) if L else F((n+1+i) if neg else (n+3+i), 2*n+4)
                if v != flint.fmpq(w.numerator, w.denominator):
                    okR = False
        line = (f"n={n}: dim={d} labels complete {ok_dim}; X links/magnitudes exact {okX}; "
                f"Y links/magnitudes exact {okY}; Y zero rows exactly L+(0),R+(n): {okZ}; "
                f"alpha/beta overlaps exact (and block-diagonal in (c,e)) {okR}")
        allok &= ok_dim and okX and okY and okZ and okR
        print(line); out.write(line+"\n"); out.flush()
    out.write(f"ALL OK: {allok}\n")
    print("ALL OK", allok)


if __name__ == "__main__":
    main()

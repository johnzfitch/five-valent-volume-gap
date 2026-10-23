"""Referee's independent exact check (no import of author code).

Representation: for spin j, basis |m>, m=j..-j, with the non-unitary but
rational frame  J+|m> = |m+1>,  J-|m> = (j+m)(j-m+1)|m-1>,  Jz|m> = m|m>.
This is diagonally similar to the standard frame, so every polynomial in the
J's (and its restriction to the invariant subspace) has the same characteristic
polynomial as in the unitary frame.  Everything is exact over Q (python-flint).
"""
import itertools, sys
from fractions import Fraction as Fr
import flint

def det3(a, b, c):
    return (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]))

def sgn(x): return (x > 0) - (x < 0)

def eps_from_tangents(T):
    n = len(T)
    return {(I, J, K): sgn(det3(T[I], T[J], T[K])) for I, J, K in itertools.combinations(range(n), 3)}

def sigma_BR(eps, n):
    """BR II eq (4.12): sigma(IJK) = e(IJK) - e(IJN) + e(IKN) - e(JKN), N = last leg."""
    N = n - 1
    return {(I, J, K): eps[(I, J, K)] - eps[(I, J, N)] + eps[(I, K, N)] - eps[(J, K, N)]
            for I, J, K in itertools.combinations(range(n - 1), 3)}

class Space:
    """M_total = Mtarget sector of j1 x ... x jn, states stored as tuples of 2m."""
    def __init__(self, twoj, twoM=0):
        self.twoj = twoj
        ranges = [range(-t, t + 1, 2) for t in twoj]
        self.states = [s for s in itertools.product(*ranges) if sum(s) == twoM]
        self.index = {s: i for i, s in enumerate(self.states)}

def apply_pm(twoj, s, leg, up):
    """J+ or J- on leg; returns (coeff, newstate) or None. m = s/2."""
    t = twoj[leg]; m2 = s[leg]
    if up:
        if m2 == t: return None
        ns = list(s); ns[leg] += 2; return Fr(1), tuple(ns)
    else:
        if m2 == -t: return None
        j = Fr(t, 2); m = Fr(m2, 2)
        ns = list(s); ns[leg] -= 2; return (j + m) * (j - m + 1), tuple(ns)

def dot_matrix(sp, I, J):
    """J_I . J_J on the sector (I may equal J -> Casimir)."""
    d = len(sp.states); M = [[Fr(0)] * d for _ in range(d)]
    tj = sp.twoj
    for c, s in enumerate(sp.states):
        # z part
        M[c][c] += Fr(s[I], 2) * Fr(s[J], 2)
        # 1/2 (I+ J- + I- J+)
        for upI, upJ in ((True, False), (False, True)):
            r1 = apply_pm(tj, s, J, upJ)
            if r1 is None: continue
            c1, s1 = r1
            r2 = apply_pm(tj, s1, I, upI)
            if r2 is None: continue
            c2, s2 = r2
            if s2 in sp.index:
                M[sp.index[s2]][c] += Fr(1, 2) * c1 * c2
    return M

def to_fmpq(M):
    d = len(M)
    return flint.fmpq_mat(d, d, [flint.fmpq(x.numerator, x.denominator) for row in M for x in row])

def build_ops(twoj):
    sp = Space(twoj)
    n = len(twoj)
    dots = {}
    for I in range(n):
        for J in range(I, n):
            dots[(I, J)] = to_fmpq(dot_matrix(sp, I, J))
    def dot(I, J): return dots[(min(I, J), max(I, J))]
    def cas(I, J):
        return dot(I, I) + dot(J, J) + 2 * dot(I, J)
    cache = {}
    def C(I, J):
        if (I, J) not in cache: cache[(I, J)] = cas(I, J)
        return cache[(I, J)]
    def qhat(I, J, K):
        a = C(I, J); b = C(J, K); return a * b - b * a
    return sp, qhat

def invariant_basis(twoj):
    sp0 = Space(twoj, 0); sp1 = Space(twoj, 2)
    d0, d1 = len(sp0.states), len(sp1.states)
    Jp = [[0] * d0 for _ in range(d1)]
    for c, s in enumerate(sp0.states):
        for leg in range(len(twoj)):
            r = apply_pm(twoj, s, leg, True)
            if r: Jp[sp1.index[r[1]]][c] += int(r[0])
    if d1 == 0:
        return flint.fmpz_mat(d0, d0, [int(i == k) for i in range(d0) for k in range(d0)]), d0
    Z = flint.fmpz_mat(d1, d0, [x for row in Jp for x in row])
    X, nul = Z.nullspace()
    cols = [[X[i, k] for i in range(d0)] for k in range(nul)]
    B = flint.fmpz_mat(d0, nul, [cols[k][i] for i in range(d0) for k in range(nul)])
    return B, nul

def restrict(A, B):
    """Return R with A B = B R (A preserves span B)."""
    d0, k = B.nrows(), B.ncols()
    Bq = flint.fmpq_mat(B)
    AB = A * Bq
    # pick k independent rows of B via rref of B^T
    BT = Bq.transpose()
    R_, rank = BT.rref()
    piv = []
    r = 0
    for c in range(d0):
        if r < rank and R_[r, c] != 0:
            piv.append(c); r += 1
    assert len(piv) == k
    Bs = flint.fmpq_mat(k, k, [Bq[i, j] for i in piv for j in range(k)])
    ABs = flint.fmpq_mat(k, k, [AB[i, j] for i in piv for j in range(k)])
    R = Bs.solve(ABs)
    # verify exactly
    assert Bq * R == AB, "span not invariant"
    return R

def full_operator(twoj, weights):
    """weights: dict (I,J,K)->int over sorted triples; returns restriction to Inv."""
    sp, qhat = build_ops(twoj)
    d = len(sp.states)
    A = flint.fmpq_mat(d, d)
    for (I, J, K), w in weights.items():
        if w: A += w * qhat(I, J, K)
    B, k = invariant_basis(twoj)
    if k == 0: return None
    return restrict(A, B)

# ---------- chain from Proposition 1 (referee's implementation of the stated formulas)
def F(p, q, k):
    return ((p + q + 1) ** 2 - k ** 2) * (k ** 2 - (p - q) ** 2) / Fr(16)

def chain_sq_couplings(j2, j3, j4, j5, alpha, beta):
    sites = []
    b = abs(j2 - j3)
    while b <= j2 + j3:
        for c in (b - Fr(1, 2), b + Fr(1, 2)):
            if c >= 0 and abs(j4 - j5) <= c <= j4 + j5 and (c + j4 + j5).denominator == 1:
                sites.append((b, c))
        b += 1
    sites.sort(key=lambda t: t[0] + t[1])
    sq = []
    for (b1, c1), (b2, c2) in zip(sites, sites[1:]):
        assert b1 + c1 + 1 == b2 + c2, ("gap in path", sites)
        if b1 == b2:      # D-link (b,b-1/2)-(b,b+1/2)
            assert c2 == c1 + 1 and c1 == b1 - Fr(1, 2)
            sq.append(16 * beta ** 2 * F(j4, j5, b1 + Fr(1, 2)))
        else:             # X-link (c-1/2,c)-(c+1/2,c)
            assert c1 == c2 and b2 == b1 + 1
            sq.append(16 * alpha ** 2 * F(j2, j3, c1 + Fr(1, 2)))
    return sites, sq

def antisym_chain_charpoly(sq, n):
    x = flint.fmpq_poly([0, 1])
    p0, p1 = flint.fmpq_poly([1]), x
    if n == 0: return p0
    for k in range(1, n):
        c2 = sq[k - 1]
        p0, p1 = p1, x * p1 + flint.fmpq(c2.numerator, c2.denominator) * p0
    return p1

TANG = [(1, 3, 1), (0, -2, 0), (-2, 0, -3), (-3, 1, 0), (-3, 1, 2)]

if __name__ == "__main__":
    import time
    out = []
    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
    eps = eps_from_tangents(TANG)
    sig = sigma_BR(eps, 5)
    log("determinants:", {k: det3(*(TANG[i] for i in k)) for k in eps})
    log("sigma(123,124,134,234) =", [sig[k] for k in sorted(sig)])
    # 1. Theorem-1 family, all ten eps triples on the full tensor product vs chain
    for N in range(2, int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 6):
        t = time.time()
        twoj = (1, N - 1, N, N, N)
        R = full_operator(twoj, eps)
        R2 = full_operator(twoj, {k: v for k, v in sig.items()})
        pA = R.charpoly(); pS = R2.charpoly()
        j = Fr(N, 2)
        sites, sq = chain_sq_couplings(j - Fr(1, 2), j, j, j, 4, 2)
        pc = antisym_chain_charpoly(sq, len(sites))
        log(f"2j={N}: dim Inv={R.nrows()} sites={len(sites)}  charpoly(eps,10 triples)==charpoly(chain): {pA == pc}"
            f"  charpoly(sigma-reduced)==chain: {pS == pc}  p(0)!=0: {pc(0) != 0}  [{time.time()-t:.1f}s]")
    open(sys.argv[2] if len(sys.argv) > 2 else "ref_exact_out.txt", "w").write("\n".join(out) + "\n")

"""Referee: Theorem 3 (valence 6 with e6 tangent to e5), exact over Q.
Builds the 6-valent operator from all 20 eps triples (eps from tangents, T6 = T5),
checks that the Theorem-1 chain charpoly divides the full charpoly exactly, that
A commutes with K^2 = (J5+J6)^2 on Inv, and reports kernel dimension."""
import sys
from fractions import Fraction as Fr
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from ref_exact import *

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
H = Fr(1, 2)
T6 = TANG + [TANG[4]]
eps = eps_from_tangents(T6)
log("zero eps triples:", [k for k, v in eps.items() if v == 0])
for N, tj5, tj6 in [(2, 1, 1), (2, 2, 2), (3, 1, 2), (3, 2, 1), (4, 2, 2), (4, 3, 1)]:
    twoj = (1, N - 1, N, N, tj5, tj6)
    sp, qhat = build_ops(twoj)
    d = len(sp.states)
    A = flint.fmpq_mat(d, d)
    for k, w in eps.items():
        if w: A += w * qhat(*k)
    B, kdim = invariant_basis(twoj)
    R = restrict(A, B)
    # K^2 = (J5+J6)^2 on the M=0 sector
    K2 = to_fmpq(dot_matrix(sp, 4, 4)) + to_fmpq(dot_matrix(sp, 5, 5)) + 2 * to_fmpq(dot_matrix(sp, 4, 5))
    RK = restrict(K2, B)
    comm = (R * RK - RK * R) == flint.fmpq_mat(kdim, kdim)
    jv = Fr(N, 2)
    sites, sq = chain_sq_couplings(jv - H, jv, jv, jv, 4, 2)
    pc = antisym_chain_charpoly(sq, len(sites))
    pA = R.charpoly()
    q, r = divmod(flint.fmpq_poly(pA), pc)
    # kernel dim of A on Inv
    rank = R.rank()
    log(f"2j=({','.join(map(str,twoj))}): dimInv={kdim}  [A,K^2]=0: {comm}  chain charpoly divides full charpoly: {r == 0}  "
        f"dim ker A|Inv = {kdim - rank}")
open(sys.argv[1] if len(sys.argv) > 1 else "ref_thm3_out.txt", "w").write("\n".join(out) + "\n")

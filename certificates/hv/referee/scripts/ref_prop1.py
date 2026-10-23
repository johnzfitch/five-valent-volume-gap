"""Referee: exact tests of Proposition 1 (general spins), Theorem 2, a second embedding,
and negative controls.  Uses only ref_exact.py (referee's own)."""
import itertools, random, sys, time
from fractions import Fraction as Fr
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from ref_exact import *

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

H = Fr(1, 2)
# ---- negative controls: the comparison must be able to fail
twoj = (1, 3, 4, 4, 4); j = Fr(2)
R = full_operator(twoj, {(0,1,2): 4, (0,1,3): 2, (0,2,3): 2, (1,2,3): 0})
sites, sq = chain_sq_couplings(j - H, j, j, j, 4, 2)
log("control: correct chain matches:", R.charpoly() == antisym_chain_charpoly(sq, len(sites)))
sites, sq3 = chain_sq_couplings(j - H, j, j, j, 3, 2)
log("control: alpha=3 chain matches (should be False):", R.charpoly() == antisym_chain_charpoly(sq3, len(sites)))
R1 = full_operator(twoj, {(0,1,2): 4, (0,1,3): 2, (0,2,3): 2, (1,2,3): 1})
log("control: sigma234=1 operator matches chain (should be False):", R1.charpoly() == antisym_chain_charpoly(sq, len(sites)))

# ---- Proposition 1, general spins j2..j5 (2j <= MAX), several (alpha, beta)
MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ab = [(4, 2), (2, 2), (2, 4), (1, 3), (-4, 2), (3, -5), (0, 2), (2, 0)]
ncase = nok = 0; bad = []
t = time.time()
for tj in itertools.product(range(0, MAX + 1), repeat=4):
    twoj = (1,) + tj
    if sum(twoj) % 2: continue
    jj = [Fr(x, 2) for x in tj]
    B, k = invariant_basis(twoj)
    if k == 0: continue
    sp, qhat = build_ops(twoj)
    Bq = flint.fmpq_mat(B)
    Rs = {key: restrict(qhat(*key), B) for key in [(0,1,2), (0,1,3), (0,2,3)]}
    for a_, b_ in ab:
        Rm = a_ * Rs[(0,1,2)] + b_ * (Rs[(0,1,3)] + Rs[(0,2,3)])
        sites, sq = chain_sq_couplings(*jj, a_, b_)
        ok = (len(sites) == k) and Rm.charpoly() == antisym_chain_charpoly(sq, len(sites))
        ncase += 1; nok += ok
        if not ok: bad.append((twoj, a_, b_, k, len(sites)))
log(f"Prop 1 general: {nok}/{ncase} (tuple,alpha,beta) cases exact charpoly match, 2j_i<=" f"{MAX}, j1=1/2  [{time.time()-t:.0f}s]")
if bad: log("  failures (first 10):", bad[:10])
# zero couplings? (does every link weight > 0 hold?)
zero = 0
for tj in itertools.product(range(0, 9), repeat=4):
    jj = [Fr(x, 2) for x in tj]
    if (1 + sum(tj)) % 2: continue
    sites, sq = chain_sq_couplings(*jj, 1, 1)
    zero += sum(1 for s in sq if s == 0)
log("zero link weights among all tuples 2j<=8 with alpha=beta=1:", zero)

# ---- Theorem 2 family exact (small M, 2Delta)
for M in range(1, 5):
    for tD in range(1, 4):
        D = Fr(tD, 2)
        twoj = (1, M, M + tD, M + 1, M + tD)
        eps = eps_from_tangents(TANG)
        R = full_operator(twoj, eps)
        jj = [Fr(M, 2), Fr(M, 2) + D, Fr(M + 1, 2), Fr(M, 2) + D]
        sites, sq = chain_sq_couplings(*jj, 4, 2)
        pc = antisym_chain_charpoly(sq, len(sites))
        # P^2 = 4^{1-M}(M+1)^2 4 D (M+D+1); E_{N1}^{-2} = prod odd^2 / prod even^2
        odd = sq[0::2]; even = sq[1::2]
        num = Fr(1); den = Fr(1)
        for s in odd: num *= s
        for s in even: den *= s
        P2 = Fr(4) ** (1 - M) * (M + 1) ** 2 * 4 * D * (M + D + 1)
        log(f"Thm2 M={M} 2D={tD}: dimInv={R.nrows()} sites={len(sites)} charpoly match={R.charpoly()==pc} "
            f"p(0)!=0={pc(0)!=0} prodID={num/den==P2} startsD/endsD={len(sites)%2==0}")

# ---- second embedding: random integer tangents with a DIFFERENT eps vector but sigma=(4,2,2,0)
random.seed(12345)
base = eps_from_tangents(TANG)
classes = {}
for trial in range(400000):
    T = [tuple(random.randint(-6, 6) for _ in range(3)) for _ in range(5)]
    e = eps_from_tangents(T)
    if 0 in e.values(): continue
    s = sigma_BR(e, 5)
    if [s[k] for k in sorted(s)] == [4, 2, 2, 0]:
        key = tuple(e[k] for k in sorted(e))
        if key not in classes: classes[key] = T
log(f"distinct non-coplanar eps realising sigma=(4,2,2,0) among 400000 random integer tangent sets: {len(classes)}")
for key, T in classes.items():
    e = dict(zip(sorted(base), key))
    log("embedding tangents:", T, "eps:", list(key), "(author's eps)" if e == base else "(different eps)")
    if T == TANG: continue
    for N in (2, 3, 4, 5):
        twoj = (1, N - 1, N, N, N); jv = Fr(N, 2)
        R = full_operator(twoj, e)
        sites, sq = chain_sq_couplings(jv - H, jv, jv, jv, 4, 2)
        log(f"   2j={N}: all-ten-eps charpoly == chain: {R.charpoly()==antisym_chain_charpoly(sq,len(sites))}")
log("eps of author's embedding:", [base[k] for k in sorted(base)])
open(sys.argv[2] if len(sys.argv) > 2 else "ref_prop1_out.txt", "w").write("\n".join(out) + "\n")

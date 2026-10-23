"""Referee numerics (independent code; reads the author's sorted-survey JSONL only as data
to pick witness spin tuples, recomputing every value with my own operator).
Usage: python ref_numerics.py LAB_M SORT_M AUTHOR_SORTED_JSONL OUTFILE
"""
import sys, os, itertools, math, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
import mpmath as mp
from ref_ops import bc_operator, qvals

LAB_M, SORT_M = int(sys.argv[1]), int(sys.argv[2]); JS = sys.argv[3]; outf = sys.argv[4]
out = []
def P(*s):
    out.append(' '.join(str(x) for x in s)); print(*s, flush=True)


def chain_c2(J, alpha, beta):
    """j1 = 1/2 chain (doubled spins J), exact c^2 list in site order."""
    _, a2, a3, a4, a5 = J
    mX, nX, mD, nD = abs(a2 - a3), a2 + a3 + 2, abs(a4 - a5), a4 + a5 + 2
    S = sorted([(b, c) for b in range(mX, nX - 1, 2) for c in range(mD, nD - 1, 2) if abs(b - c) == 1],
               key=lambda x: x[0] + x[1])
    cs = []
    for (b, c), (b2, c2) in zip(S, S[1:]):
        if c2 == c:
            l = c + 1; cs.append(Fr(alpha * alpha, 16) * (nX * nX - l * l) * (l * l - mX * mX))
        else:
            l = b + 1; cs.append(Fr(beta * beta, 16) * (nD * nD - l * l) * (l * l - mD * mD))
    return cs


# (a) closed form of the conjectured constant
cs = chain_c2((1, 1, 2, 2, 2), 2, 2)
n = len(cs) + 1
x = sp.symbols('x')
Mx = sp.zeros(n, n)
for i, c in enumerate(cs):
    Mx[i, i + 1] = Mx[i + 1, i] = sp.sqrt(sp.Rational(c.numerator, c.denominator))
cp = sp.factor((x * sp.eye(n) - Mx).det())
roots = [r for r in sp.solve(sp.expand(cp), x)]
pos = sorted([sp.nsimplify(r) for r in roots if r.is_positive], key=lambda r: float(r))
target = 2 * sp.sqrt(18 - 2 * sp.sqrt(41))
P(f'(a) chain (1/2;1/2,1;1,1): c^2 = {cs}; charpoly = {cp}; positive roots = {pos};'
  f' min root - 2sqrt(18-2sqrt41) simplifies to {sp.simplify(pos[0] - target)}; value {float(target):.10f}')

# (b) Theorem A family and (c) Proposition T
mp.mp.dps = 60


def qmin_mp(cs):
    """smallest positive eigenvalue of even zero-diagonal Jacobi chain with link squares cs (exact)."""
    K = (len(cs) + 1) // 2
    c = [mp.sqrt(mp.mpf(v.numerator) / v.denominator) for v in cs]
    B = mp.zeros(K, K)
    for i in range(K):
        B[i, i] = c[2 * i]
        if i + 1 < K:
            B[i + 1, i] = c[2 * i + 1]
    s = mp.svd_r(B, compute_uv=False)
    return min(s[i] for i in range(K))


mx = 0
for N in range(2, 9):
    J = (1, N - 1, N, N, N)
    S, A = bc_operator(J, 4.0, 2.0)
    q_op, k_op = qvals(A)
    q_ch = float(qmin_mp(chain_c2(J, 4, 2)))
    mx = max(mx, abs(q_op[0] - q_ch) / q_ch)
    assert k_op == 0
P(f'(b) Theorem A family: chain q_min vs signed operator N=2..8, max rel err {mx:.1e}, kernel trivial in operator: True')
rat = []
for N in list(range(2, 41)) + [60, 80, 100]:
    q = qmin_mp(chain_c2((1, N - 1, N, N, N), 4, 2))
    PN = 4 * N * mp.sqrt(2 * N + 1) * mp.mpf(2) ** (-N)
    rat.append((N, float(q / PN)))
P('    q_min/P_N for N=2..40,60,80,100: min %.5f max %.5f; all in [3/8,1]: %s; N=100 ratio %.5f (kappa=0.80732)'
  % (min(r for _, r in rat), max(r for _, r in rat), all(0.375 <= r <= 1 for _, r in rat), rat[-1][1]))
for t in (Fr(1, 2), Fr(4, 5), Fr(9, 10), Fr(99, 100)):
    row = []
    for N in (10, 40, 80):
        alpha = 4
        q = qmin_mp(chain_c2((1, N - 1, N, N, N), alpha, alpha * t))
        bnd = alpha * N * mp.sqrt(2 * N + 1) * mp.mpf(t.numerator) ** N / mp.mpf(t.denominator) ** N
        row.append(f'N={N}: q={mp.nstr(q, 5)} <= {mp.nstr(bnd, 5)} {bool(q <= bnd)}')
    P(f'(c) Prop T, alpha=4, t={t}: ' + '; '.join(row))
# exchange relation A_t(j1;p;r) = t * A_{1/t}(j1;r;p)
mx = 0
for J in [(2, 3, 3, 2, 4), (3, 2, 3, 4, 4), (2, 4, 4, 3, 3), (4, 3, 4, 4, 3)]:
    for t in (0.5, 0.8, 3.0):
        _, A1 = bc_operator(J, 1.0, t)
        _, A2 = bc_operator((J[0], J[3], J[4], J[1], J[2]), 1.0, 1 / t)
        assert A1.shape[0] >= 2, J
        e1 = np.sort(np.abs(np.linalg.eigvals(A1))); e2 = np.sort(np.abs(np.linalg.eigvals(t * A2)))
        mx = max(mx, np.abs(e1 - e2).max())
P(f'    pair-exchange identity A_t = t*A_(1/t) (relabelled), incl. j1>=1 grids: max abs err {mx:.1e}')

# (d) labelled surveys with my signed operator; also test the moduli-only matrix W
def labelled(orbit, M):
    ab = {1: (2.0, 0.0), 2: (2.0, 2.0), 3: (4.0, 2.0)}[orbit]
    res = {}
    wbad = 0; wn = 0
    for J in itertools.product(range(1, M + 1), repeat=5):
        if sum(J) % 2 or J[1] > J[2] or J[3] > J[4]:
            continue
        if orbit == 2 and (J[1], J[2]) > (J[3], J[4]):
            continue
        S, A = bc_operator(J, *ab)
        if len(S) < 2:
            continue
        q, k = qvals(A)
        W = np.abs(A); W = (W + W.T) / 2
        ew = np.sort(np.abs(np.linalg.eigvalsh(W)))
        ea = np.sort(np.abs(np.linalg.eigvals(A)))
        wn += 1
        if np.abs(ew - ea).max() > 1e-9 * max(1, ea.max()):
            wbad += 1
        if q.size == 0:
            continue
        m = max(J)
        if m not in res or q[0] < res[m][0]:
            res[m] = (q[0], J)
    return res, wbad, wn


t0 = time.time()
for orbit in (2, 3):
    res, wbad, wn = labelled(orbit, LAB_M)
    P(f'(d) labelled orbit {orbit}, 2j_max<= {LAB_M}: |spec(W)| != |spec(A)| in {wbad}/{wn} tuples;'
      f' min per 2j_max: ' + '; '.join(f'{m}:{res[m][0]:.9e}@{res[m][1]}' for m in sorted(res)))
P(f'    ({time.time()-t0:.0f}s)')

# (e) sorted convention, orbit 3
recs = [json.loads(l) for l in open(JS, encoding='utf-8')]
def nf_q(sj, pat):
    k, (p, q) = pat
    rest = [i for i in range(1, 6) if i not in (k, p, q)]
    J = (sj[k - 1], sj[p - 1], sj[q - 1], sj[rest[0] - 1], sj[rest[1] - 1])
    S, A = bc_operator(J, 4.0, 2.0)
    qq, kk = qvals(A, rel=1e-13)
    return qq[0] if qq.size else None

const_pats = [(4, (3, 5)), (2, (4, 5)), (3, (4, 5)), (5, (3, 4)), (5, (2, 3)), (3, (1, 5)), (3, (2, 5))]
# exhaustive sorted minima for constant patterns up to SORT_M, compared with author's records
t0 = time.time()
mine = {}
for J in itertools.product(range(1, SORT_M + 1), repeat=5):
    if sum(J) % 2 or list(J) != sorted(J):
        continue
    for pat in const_pats:
        v = nf_q(J, pat)
        if v is None:
            continue
        key = (pat, max(J))
        if key not in mine or v < mine[key][0]:
            mine[key] = (v, J)
auth = {((r['pattern'][0], tuple(r['pattern'][1])), r['m']): r for r in recs}
mxd = 0; cnt = 0
for key, (v, J) in mine.items():
    a = auth[key]['q_min']; mxd = max(mxd, abs(a - v) / v); cnt += 1
P(f'(e) sorted exhaustive, 7 "constant" patterns, each 2j_max<= {SORT_M}: {cnt} (pattern,2j_max) minima;'
  f' max rel diff to author = {mxd:.1e} ({time.time()-t0:.0f}s)')
# witnesses: every pattern at 2j_max = 22 and 34
mxd = 0
lines = []
for r in recs:
    if r['m'] in (22, 34):
        pat = (r['pattern'][0], tuple(r['pattern'][1]))
        v = nf_q(r['argmin_sorted'], pat)
        mxd = max(mxd, abs(v - r['q_min']) / r['q_min'])
        if pat in const_pats:
            lines.append(f'{pat} m={r["m"]}: author {r["q_min"]:.4e} mine {v:.4e} at {r["argmin_sorted"]}')
P(f'    witnesses at 2j_max=22 and 34, all 30 patterns: max rel diff {mxd:.1e}')
for l in lines:
    P('    ' + l)
open(outf, 'w', encoding='utf-8').write('\n'.join(out) + '\n')

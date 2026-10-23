"""Referee: Theorem B2 (orbit 2, alpha=beta=2, j1=1/2), every proof step.
(1) closed-form chain vs my signed (b,c) operator (float) for 2j<=SMALL;
(2) exact-rational checks of Lemma 5, Lemma 6, the even-N ratio and link
    bounds, the odd-N monotonicity and link bounds, the Frobenius sums,
    and q_min >= proved bound, for all chains with 2j_2..2j_5 <= M and for
    RANDOM random chains with 2j up to BIG;
(3) the constants: min over n of the even-N and odd-N final bounds.
Usage: python ref_b2.py M SMALL RANDOM BIG OUTFILE
"""
import sys, os, itertools, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from ref_ops import bc_operator, qvals

M, SMALL, RANDOM, BIG = map(int, sys.argv[1:5]); outf = sys.argv[5]
out = []
def P(*s):
    out.append(' '.join(str(x) for x in s)); print(*s, flush=True)


def chain(a2, a3, a4, a5):
    """Doubled spins. Returns list of links (type, l, c2) in site order, plus m/n data."""
    mX, nX, mD, nD = abs(a2 - a3), a2 + a3 + 2, abs(a4 - a5), a4 + a5 + 2
    bs = range(mX, nX - 1, 2); cs = range(mD, nD - 1, 2)
    S = sorted([(b, c) for b in bs for c in cs if abs(b - c) == 1], key=lambda x: x[0] + x[1])
    links = []
    for (b, c), (b2, c2) in zip(S, S[1:]):
        assert b2 + c2 == b + c + 2, 'not consecutive in s'
        if c2 == c:          # X link, b: c-1 -> c+1 (doubled), label l = c+1
            assert b == c - 1 and b2 == c + 1
            l = c + 1; cc = Fr((nX * nX - l * l) * (l * l - mX * mX), 4); links.append(('X', l, cc))
        else:
            assert b2 == b and c == b - 1 and c2 == b + 1
            l = b + 1; cc = Fr((nD * nD - l * l) * (l * l - mD * mD), 4); links.append(('D', l, cc))
    return S, links, dict(X=(mX, nX), D=(mD, nD))


def qmin_chain(links):
    if not links:
        return None
    off = np.sqrt(np.array([float(x[2]) for x in links]))
    ev = eigvalsh_tridiagonal(np.zeros(len(links) + 1), off)
    a = np.abs(ev); thr = 1e-10 * a.max()
    return a[a > thr].min()


def H(K): return sum(Fr(1, i) for i in range(1, K + 1))
def H2(K): return sum(Fr(1, i * i) for i in range(1, K + 1))


def frob_E(cs):
    """Lemma-2 E for an even chain given link c^2 list (Fractions); returns ||E||_F^2 (Fraction)."""
    K = (len(cs) + 1) // 2
    tot = Fr(0)
    for j in range(1, K + 1):
        e2 = 1 / cs[2 * j - 2]
        tot += e2
        for i in range(j + 1, K + 1):
            e2 = e2 * cs[2 * i - 3] / cs[2 * i - 2]     # times c_{2(i-1)}^2 / c_{2(i-1)+1}^2
            tot += e2
    return tot


def smin_even(cs):
    if not cs:
        return math.inf
    off = np.sqrt(np.array([float(x) for x in cs]))
    ev = eigvalsh_tridiagonal(np.zeros(len(cs) + 1), off)
    return np.abs(ev).min()


stats = dict(chains=0, even=0, odd=0, viol=0)
viol_msgs = []
gmin = (math.inf, None); min_ratio = (math.inf, None)
BOUND = 6 * math.sqrt(2) / math.pi


def V(msg):
    stats['viol'] += 1
    if len(viol_msgs) < 20:
        viol_msgs.append(msg)


def check(a2, a3, a4, a5):
    global gmin, min_ratio
    S, links, mn = chain(a2, a3, a4, a5)
    if len(S) < 2:
        return
    stats['chains'] += 1
    (mX, nX), (mD, nD) = mn['X'], mn['D']
    m, n = max(mX, mD), min(nX, nD)
    N = len(S)
    tag = (1, a2, a3, a4, a5)
    # Lemma 5
    if (mX - mD) % 2 == 0 or (nX - nD) % 2 == 0: V(('parity', tag))
    if [l for _, l, _ in links] != list(range(m + 1, n)): V(('labels', tag))
    if N != n - m: V(('N', tag))
    if any(links[i][0] == links[i + 1][0] for i in range(len(links) - 1)): V(('alt', tag))
    ownm = 'X' if mX == m else 'D'; ownn = 'X' if nX == n else 'D'
    if links[0][0] == ownm or links[-1][0] == ownn: V(('ends', tag))
    if m < 1: V(('m>=1', tag))
    cs = [x[2] for x in links]
    q = qmin_chain(links)
    if q < gmin[0]: gmin = (q, tag)
    if N % 2 == 0:
        stats['even'] += 1
        K = N // 2
        O = links[0][0]; E = 'D' if O == 'X' else 'X'
        mO, nO = mn[O]; mE, nE = mn[E]
        if not (all(links[i][0] == O for i in range(0, len(links), 2)) and mE == m and nE == n
                and mO <= m - 1 and nO >= n + 1): V(('L6 even', tag))
        for t in range(1, K):
            L_ = m + 2 * t
            r = cs[2 * t - 1] / cs[2 * t]
            f1 = Fr(n + L_, n + L_ + 2); f2 = Fr(L_ - m, L_ - m + 2)
            if not (r <= f1 * f2 <= Fr(t, t + 1)): V(('even ratio', tag, t))
        for j in range(1, K + 1):
            if cs[2 * j - 2] < 4 * n * j * j: V(('odd link bound', tag, j))
        fe = frob_E(cs)
        fb = (H(K) ** 2 + H2(K)) / (8 * n)
        if fe > fb: V(('frob even', tag))
        bound = math.sqrt(1 / float(fb))
        if abs(q - smin_even(cs)) > 1e-9 * q: V(('lemma2 q', tag))
        if q * (1 + 1e-12) < math.sqrt(1 / float(fe)): V(('1/||E||_F > q', tag))
    else:
        stats['odd'] += 1
        K = (N - 1) // 2
        Q = links[0][0]; Pp = 'D' if Q == 'X' else 'X'
        mQ, nQ = mn[Q]; mP, nP = mn[Pp]
        if not (Q == ownn and Pp == ownm and mQ <= m - 1 and nP >= n + 1): V(('L6 odd', tag))
        rho = [cs[2 * t - 1] / cs[2 * t] for t in range(1, K)]
        if any(rho[i] >= rho[i + 1] for i in range(len(rho) - 1)): V(('rho monotone', tag))
        tau = sum(1 for x in rho if x <= 1); r = tau + 1
        if not all(x <= 1 for x in rho[:tau]): V(('tau prefix', tag))
        # left chain: sites 1..2r, links 1..2r-1
        left = cs[:2 * r - 1]
        for j in range(1, r + 1):
            l = m + 2 * j - 1
            if left[2 * j - 2] < 2 * (r - j + 1) * (n + l) * j * j: V(('left link', tag, j))
        fL = frob_E(left)
        if fL > Fr(1) * sum(Fr(1, 2 * (n + m + 2 * j - 1) * j * j) for j in range(1, r + 1)): V(('frobL', tag))
        if float(fL) > math.pi ** 2 / (12 * (n + m + 1)) + 1e-15: V(('frobL pi', tag))
        # right chain: sites 2r+2..N, links 2r+2..2K, read reversed
        right = cs[2 * r + 1:][::-1]
        for t in range(r + 1, K + 1):
            l = m + 2 * t
            if cs[2 * t - 1] < 2 * (K - t + 1) * (n + m + 3) * t * (m + 1): V(('right link', tag, t))
        fR = frob_E(right) if right else Fr(0)
        if fR > H(K) / (4 * (n + 4)): V(('frobR', tag))
        bound = min(BOUND, math.sqrt(4 * (n + 4) / float(H(K))))
        sL = smin_even(left); sR = smin_even(right)
        if q * (1 + 1e-12) < min(sL, sR): V(('row deletion', tag))
    if q < bound * (1 - 1e-12): V(('final bound', tag, q, bound))
    if q / bound < min_ratio[0]: min_ratio = (q / bound, tag, q, bound)


# (1) closed-form chain vs signed operator
mx = 0; ncmp = 0
for a2, a3, a4, a5 in itertools.product(range(1, SMALL + 1), repeat=4):
    if (1 + a2 + a3 + a4 + a5) % 2: continue
    S, A = bc_operator((1, a2, a3, a4, a5), 2.0, 2.0)
    if len(S) < 2: continue
    q_op, k_op = qvals(A)
    _, links, _ = chain(a2, a3, a4, a5)
    off = np.sqrt([float(x[2]) for x in links])
    ev = np.sort(np.abs(eigvalsh_tridiagonal(np.zeros(len(links) + 1), off)))
    full = np.sort(np.abs(np.linalg.eigvals(A)))
    mx = max(mx, np.abs(ev - full).max() / full.max()); ncmp += 1
P(f'(1) closed-form j1=1/2 chain vs signed (b,c) operator, 2j<= {SMALL}: {ncmp} tuples, max rel err {mx:.2e}')

# (2) exhaustive
for a2 in range(1, M + 1):
    for a3 in range(a2, M + 1):
        for a4 in range(1, M + 1):
            for a5 in range(a4, M + 1):
                if (1 + a2 + a3 + a4 + a5) % 2: continue
                check(a2, a3, a4, a5)
P(f'(2) exhaustive 2j_2..2j_5 <= {M} (j2<=j3, j4<=j5): chains {stats["chains"]}, even {stats["even"]}, odd {stats["odd"]}, violations {stats["viol"]}')
P(f'    global min q_min = {gmin[0]:.10f} at doubled spins {gmin[1]}; min q/bound = {min_ratio}')
rng = random.Random(2026)
c0 = dict(stats)
for _ in range(RANDOM):
    while True:
        a = [rng.randint(1, BIG) for _ in range(4)]
        if (1 + sum(a)) % 2 == 0: break
    check(*a)
P(f'    random {RANDOM} with 2j <= {BIG}: chains {stats["chains"]-c0["chains"]}, violations {stats["viol"]-c0["viol"]}')
for m_ in viol_msgs: P('    VIOL', m_)
# (3) constants
best_e = (math.inf, None); best_o = (math.inf, None)
Hc = [0.0]; H2c = [0.0]
for i in range(1, 200001):
    Hc.append(Hc[-1] + 1 / i); H2c.append(H2c[-1] + 1 / i / i)
for n in range(4, 400001):
    Ke = (n - 1) // 2
    if Ke <= 200000:
        v = math.sqrt(8 * n / (Hc[Ke] ** 2 + H2c[Ke]))
        if v < best_e[0]: best_e = (v, n)
    Ko = (n - 2) // 2
    if 1 <= Ko <= 200000:
        v = math.sqrt(4 * (n + 4) / Hc[Ko])
        if v < best_o[0]: best_o = (v, n)
P(f'(3) even-N final bound min over 4<=n<=4e5 (K=(n-1)//2): {best_e}; odd-N right bound min (K=(n-2)//2): {best_o};'
  f' 6sqrt2/pi = {BOUND:.6f}')
open(outf, 'w', encoding='utf-8').write('\n'.join(out) + '\n')

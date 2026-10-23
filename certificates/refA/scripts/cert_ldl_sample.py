"""Exact re-certification of a random sample of the finite certificate [FC] (and of the j1 = 5/2 recheck [RR]) with own
code and a different exact matrix from the authors' Gram-matrix LDL.

For a tuple, W (closed form, refcore) is bipartite with block B (rows E, cols O).  Pick a spanning forest and set
theta_k^2 rational with theta_i theta_k x_e = 1 on tree edges; then Bt = Theta_E B Theta_O is rational (non-tree edges
are checked to give exact rational squares).  By the Schur complement and Sylvester's law,
    Z(eps) := [[-Theta_E^2, Bt], [Bt^T, -eps Theta_O^2]]  =  diag(Theta_E, Theta_O) [[-I, B], [B^T, -eps]] diag(Theta_E, Theta_O)
has inertia  (n_E + n_-(B^T B - eps), n_0(B^T B - eps), n_+(B^T B - eps)).
Certified (every nonzero sigma(B)^2 > eps, i.e. every nonzero |eig 8T| > 8 sqrt(eps)) iff n_0 = 0 and
n_-(B^T B - eps) = dim ker B (= n_O - rank Bt, exact via flint).  eps = (45579607/80000000)^2 exactly.
Z is factorised by a banded LDL^T in python-flint fmpq in grid order (sites sorted by (b, c)); a zero pivot aborts
(reported, none expected).  The sample: per level, random nonempty tuples from the authors' jsonl files plus the
per-level minimiser; also j1 = 5/2 levels from [RR].  The exact verdict and kerT are compared with the jsonl records.
"""
import sys, json, random, time, math
from fractions import Fraction as Fr
from multiprocessing import Pool
import flint
import numpy as np
from refcore import region, links_closed_form, colour

EPS = Fr(45579607, 80000000) ** 2
FC = "../../fc/outputs/cert_M%03d.jsonl"
RR = "../../rr/outputs/cert52_M%03d.jsonl"


def fq(x): return flint.fmpq(x.numerator, x.denominator)


def exact_square_root(q):
    a, b = math.isqrt(q.numerator), math.isqrt(q.denominator)
    if a * a == q.numerator and b * b == q.denominator: return Fr(a, b)
    return None


def certify_exact(tj_doubled, eps=EPS):
    t0 = time.time()
    s, _, _, rect, R = region(tj_doubled)
    links = links_closed_form(tj_doubled)
    sites = sorted(R)
    idx = {p: i for i, p in enumerate(sites)}
    col = [colour(s, b, c) for (b, c) in sites]
    adj = [[] for _ in sites]
    for (p, q), w2 in links.items():
        adj[idx[p]].append((idx[q], w2)); adj[idx[q]].append((idx[p], w2))
    th2 = [None] * len(sites)
    for root in range(len(sites)):
        if th2[root] is not None: continue
        th2[root] = Fr(1); stack = [root]
        while stack:
            i = stack.pop()
            for k, w2 in adj[i]:
                if th2[k] is None:
                    th2[k] = 1 / (w2 * th2[i]); stack.append(k)
    bt = {}
    for (p, q), w2 in links.items():
        i, k = idx[p], idx[q]
        r = exact_square_root(w2 * th2[i] * th2[k])
        assert r is not None, "non-rational cycle product"
        bt[(i, k)] = r
    n0 = sum(1 for c in col if c == 0); n1 = len(col) - n0
    Ocol = 0 if n0 <= n1 else 1
    O = [i for i in range(len(sites)) if col[i] == Ocol]; E = [i for i in range(len(sites)) if col[i] != Ocol]
    nE, nO = len(E), len(O)
    if nO == 0: return dict(t=tj_doubled, skip=True)
    # exact rank of Bt (E x O)
    pE = {i: a for a, i in enumerate(E)}; pO = {i: a for a, i in enumerate(O)}
    Mq = flint.fmpq_mat(nE, nO)
    for (i, k), v in bt.items():
        if i in pE: Mq[pE[i], pO[k]] = fq(v)
        else: Mq[pE[k], pO[i]] = fq(v)
    rank = Mq.rank()
    zB = nO - rank
    # Z in grid order: diagonal -theta^2 on E, -eps theta^2 on O; off-diagonal Bt
    n = len(sites)
    A = [dict() for _ in range(n)]
    for i in range(n):
        A[i][i] = fq(-th2[i]) if col[i] != Ocol else fq(-eps * th2[i])
    for (i, k), v in bt.items():
        a, b = min(i, k), max(i, k)
        A[a][b] = fq(v)
    neg = pos = 0
    for k in range(n):
        piv = A[k].get(k, flint.fmpq(0))
        if piv == 0: return dict(t=tj_doubled, zero_pivot=True)
        if piv < 0: neg += 1
        else: pos += 1
        row = sorted((j, v) for j, v in A[k].items() if j > k and v != 0)
        for t, (j, vj) in enumerate(row):
            f = vj / piv
            Aj = A[j]
            for (l, vl) in row[t:]:
                Aj[l] = Aj.get(l, flint.fmpq(0)) - f * vl
        A[k] = None
    negBB = neg - nE
    ok = (negBB == zB)
    return dict(t=tj_doubled, nE=nE, nO=nO, kerT=nE + nO - 2 * rank, zB=zB, negBB=negBB, cert=ok, secs=time.time() - t0)


def load(fn):
    return [json.loads(l) for l in open(fn)]


def to_doubled(rec):
    return tuple(rec['t'])


if __name__ == '__main__':
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    random.seed(20261021)
    jobs = []
    levels = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 38]
    for M in levels:
        recs = [r for r in load(FC % M) if not r.get('skip')]
        smp = random.sample(recs, min(per, len(recs)))
        mn = min(recs, key=lambda r: r['qmin'] if r['qmin'] is not None else 9e9)
        if mn not in smp: smp.append(mn)
        jobs += [('FC', M, r) for r in smp]
    for M in (40, 50, 60, 70):
        recs = [r for r in load(RR % M) if not r.get('skip')]
        smp = random.sample(recs, min(max(2, per // 3), len(recs)))
        mn = min(recs, key=lambda r: r['qmin'])
        if mn not in smp: smp.append(mn)
        jobs += [('RR', M, r) for r in smp]
    # the two j1 = 7/2 tuples of [RR] sec. 3 (records transcribed from outputs/s72_two_tuples.txt)
    jobs.append(('RR72', 39, dict(t=[7, 35, 36, 37, 39], nE=137, nO=137, kerT=2, cert=True)))
    jobs.append(('RR72', 39, dict(t=[7, 35, 37, 38, 39], nE=140, nO=138, kerT=2, cert=True)))
    t0 = time.time()
    with Pool(8) as p:
        res = p.map(certify_exact, [to_doubled(r) for _, _, r in jobs], chunksize=1)
    lines = []; agree = 0; disagree = []; zp = 0; dimmax = 0
    for (src, M, rec), out in zip(jobs, res):
        if out.get('zero_pivot'): zp += 1; lines.append(f"{src} level {M} {rec['t']}: zero pivot"); continue
        same_ker = out['kerT'] == rec['kerT'] and out['nE'] == rec['nE'] and out['nO'] == rec['nO']
        same = same_ker and out['cert'] == rec['cert']
        agree += same
        if not same: disagree.append((src, M, rec, out))
        dimmax = max(dimmax, out['nE'] + out['nO'])
        lines.append(f"{src} level {M:2d} {str(rec['t']):24s} dim {out['nE']+out['nO']:5d}  kerT {out['kerT']:3d} (file {rec['kerT']:3d})  "
                     f"exact certified {out['cert']} (file {rec['cert']})  {out['secs']:.1f}s")
    lines.append(f"sample size {len(jobs)} (levels {levels} from [FC]; 40, 50, 60, 70 from [RR] at j1 = 5/2); "
                 f"agreement with the files on (nE, nO, kerT, certified): {agree}; disagreements: {len(disagree)}; zero pivots: {zp}; "
                 f"largest dim {dimmax}; wall {time.time()-t0:.0f}s")
    for d in disagree: lines.append(f"DISAGREE: {d}")
    # negative/positive controls: bracket the per-level minimum q of the file to +-1e-6 exactly
    ctl = []
    for M in (2, 3, 8, 16, 24, 32, 38):
        recs = [r for r in load(FC % M) if not r.get('skip')]
        mn = min(recs, key=lambda r: r['qmin'])
        q = Fr(mn['qmin']).limit_denominator(10 ** 12)
        ctl.append((M, mn['t'], mn['qmin'], ((q - Fr(1, 10 ** 6)) / 8) ** 2, ((q + Fr(1, 10 ** 6)) / 8) ** 2))
    with Pool(8) as p:
        lo = p.starmap(certify_exact, [(tuple(c[1]), c[3]) for c in ctl])
        hi = p.starmap(certify_exact, [(tuple(c[1]), c[4]) for c in ctl])
    for c, a, b in zip(ctl, lo, hi):
        lines.append(f"control level {c[0]} minimiser {c[1]} q_file={c[2]:.9f}: exact certified at q-1e-6: {a['cert']} (must be True); at q+1e-6: {b['cert']} (must be False)")
    print("\n".join(lines))
    open("../outputs/cert_ldl_sample_out.txt", "w").write("\n".join(lines) + "\n")

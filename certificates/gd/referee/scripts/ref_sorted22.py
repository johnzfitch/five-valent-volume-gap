"""Referee: exhaustive sorted-convention survey of the 30 orbit-3 patterns, 2j_max <= M,
with my own signed operator (normal form (4,2) after relabelling by the pattern).
Compares per-(pattern, 2j_max) minima and running minima with the author's JSONL (read as data).
Usage: python ref_sorted22.py M AUTHOR_JSONL OUTFILE [WORKERS]
"""
import sys, os, itertools, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool

PATS = [(k, pr) for k in range(1, 6) for pr in itertools.combinations([i for i in range(1, 6) if i != k], 2)]


def work(task):
    m, a1 = task
    from ref_ops import bc_operator, qvals
    best = {}
    for J in itertools.combinations_with_replacement(range(a1, m + 1), 3):
        sj = (a1,) + tuple(J) + (m,)
        if sum(sj) % 2:
            continue
        for pat in PATS:
            k, (p, q) = pat
            rest = [i for i in range(1, 6) if i not in (k, p, q)]
            Jn = (sj[k - 1], sj[p - 1], sj[q - 1], sj[rest[0] - 1], sj[rest[1] - 1])
            S, A = bc_operator(Jn, 4.0, 2.0)
            if len(S) < 2:
                continue
            qq, kk = qvals(A, rel=1e-13)
            if qq.size == 0:
                continue
            if pat not in best or qq[0] < best[pat][0]:
                best[pat] = (float(qq[0]), sj)
    return (m, a1), best


if __name__ == '__main__':
    M = int(sys.argv[1]); JS = sys.argv[2]; outf = sys.argv[3]
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    t0 = time.time()
    with Pool(W) as pool:
        tasks = sorted([(m, a1) for m in range(2, M + 1) for a1 in range(1, m + 1)], key=lambda t: (-t[0], t[1]))
        parts = pool.map(work, tasks, chunksize=1)
    res = {}
    for (m, a1), best in parts:
        d = res.setdefault(m, {})
        for pat, v in best.items():
            if pat not in d or v[0] < d[pat][0]:
                d[pat] = v
    recs = [json.loads(l) for l in open(JS, encoding='utf-8')]
    auth = {((r['pattern'][0], tuple(r['pattern'][1])), r['m']): r['q_min'] for r in recs}
    out = []
    mx = 0; n = 0; missing = 0
    for m, best in res.items():
        for pat, (v, sj) in best.items():
            a = auth.get((pat, m))
            if a is None:
                missing += 1; continue
            mx = max(mx, abs(a - v) / v); n += 1
    out.append(f'exhaustive sorted survey 2j_max<= {M}, 30 patterns: {n} (pattern, 2j_max) minima compared,'
               f' max rel diff to author {mx:.1e}, missing in author file {missing}; wall {time.time()-t0:.0f}s with {W} processes')
    out.append('pattern: running min over 2j_max<=22 (mine) | author runmin over <=34 from its per-m records | ratio')
    for pat in PATS:
        rm = min(res[m][pat][0] for m in res if pat in res[m])
        a34 = min(v for (p_, m_), v in auth.items() if p_ == pat)
        out.append(f'  {pat}: {rm:.4e} | {a34:.4e} | {a34/rm:.3g}')
    open(outf, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('\n'.join(out))

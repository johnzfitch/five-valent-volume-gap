"""Ladder case j2 = 1/2.  In the Heron-Stark normal form T0 (rungs h23 = 1, rails = Stark h45), order the
two colour classes by c: e_c, o_c.  Claim: the bipartite block of T0 equals the window
   P_rows (H45 + I) P_cols   (rows/cols = sets of c carrying an E/O site),
and classify the windows: 'nested' (one window's H-neighbourhood lies in the other) vs 'shifted'.
Also records lam(T0) per class and parity of N45.
Usage: python ladder_windows.py M
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import sites, grid
from explore_T0 import T0, nzmin


def stark_full(n2, m2):
    """Full Stark matrix on c = m..n-1 (doubled inputs n2 = 2n, m2 = 2m); returns list of doubled c and H."""
    cs = list(range(m2, n2 - 1, 2))
    N = len(cs)
    H = np.zeros((N, N))
    for k in range(1, N):
        C = cs[k] / 2; n = n2 / 2; m = m2 / 2
        H[k - 1, k] = H[k, k - 1] = np.sqrt((n * n - C * C) * (C * C - m * m) / (4 * C * C - 1))
    return cs, H


def analyse(tj):
    s2, j2, j3, j4, j5 = tj
    assert j2 == 1
    S, W0 = T0(tj)
    if len(S) < 2:
        return None
    l0, _ = nzmin(W0, S)
    if l0 is None:
        return None
    cs, H = stark_full(j4 + j5 + 2, abs(j4 - j5))
    pos = {c: i for i, c in enumerate(cs)}
    beta_lo = min(b for b, c in S)
    # colour: ((b+c)/2) parity
    E = sorted([(c, (b, c)) for (b, c) in S if ((b + c) // 2) % 2 == 0])
    O = sorted([(c, (b, c)) for (b, c) in S if ((b + c) // 2) % 2 == 1])
    rows = [pos[c] for c, _ in E]; cols = [pos[c] for c, _ in O]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        return ('dup', tj)
    idx = {p: i for i, p in enumerate(S)}
    Bt = W0[np.ix_([idx[p] for _, p in E], [idx[p] for _, p in O])]
    ok = None
    for sg in (1, -1):
        Bw = (H + sg * np.eye(len(cs)))[np.ix_(rows, cols)]
        if np.allclose(np.abs(Bw), np.abs(Bt), atol=1e-12):
            ok = sg
    rs, cs_ = set(rows), set(cols)
    def nbhd(X):
        out = set()
        for x in X:
            for y in (x - 1, x, x + 1):
                if 0 <= y < len(cs): out.add(y)
        return out
    if nbhd(cs_) <= rs: kind = 'nested_cols'
    elif nbhd(rs) <= cs_: kind = 'nested_rows'
    else: kind = 'shifted'
    contiguous = (max(rows) - min(rows) + 1 == len(rows)) and (max(cols) - min(cols) + 1 == len(cols))
    return (kind, ok, contiguous, len(cs) % 2, s2 % 2, l0)


if __name__ == '__main__':
    M = int(sys.argv[1])
    stats = {}
    for s2 in range(1, M + 1):
        for j3 in range(1, M + 1):
            for j4 in range(1, M + 1):
                for j5 in range(j4, M + 1):
                    tj = (s2, 1, j3, j4, j5)
                    if sum(tj) % 2: continue
                    r = analyse(tj)
                    if r is None: continue
                    if r[0] == 'dup': print('dup', tj); continue
                    kind, ok, contig, N45par, spar, l0 = r
                    key = (kind, ok, contig, 'N45 odd' if N45par else 'N45 even', 's half' if spar else 's int')
                    st = stats.setdefault(key, [0, 9e9, None])
                    st[0] += 1
                    if l0 < st[1]: st[1], st[2] = l0, tj
    for k, v in sorted(stats.items(), key=str):
        print(k, 'count', v[0], 'min lam0 %.6f' % v[1], v[2])

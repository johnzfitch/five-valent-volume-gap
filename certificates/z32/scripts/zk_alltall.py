"""Existence of an all-tall ordered pairing (every zigzag imbalance in {0,+1}) for half-integer j1.
Imbalances from exact site counts (combinatorial).  For 2j1 = 5 the family argument of [OR] Lemma 4'
reduces the claim for all spins to the tuples with 2j <= 28; we check 2j <= M.
Usage: python zk_alltall.py 2s M out.txt
"""
import sys, collections


def imbalances(tj):
    s2, a, b_, c_, d_ = tj
    K = s2 // 2 + 1
    rows = [0] * K; cols = [0] * K
    for B in range(abs(a - b_), a + b_ + 1, 2):
        for C in range(abs(c_ - d_), c_ + d_ + 1, 2):
            if abs(B - C) <= s2 <= B + C and (B + C + s2) % 2 == 0:
                e = (B - C + s2) // 2          # = d + s
                if e % 2 == 0:
                    rows[e // 2] += 1
                else:
                    cols[(e - 1) // 2] += 1
    return [rows[k] - cols[k] for k in range(K)], sum(rows) + sum(cols)


def pairings6(tj):
    s, a, b, c, d = tj
    return [(s, a, b, c, d), (s, c, d, a, b), (s, a, c, b, d), (s, b, d, a, c), (s, a, d, b, c), (s, b, c, a, d)]


if __name__ == '__main__':
    s2, M, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    tot = ok = 0; bad = []; pat = collections.Counter()
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    tj = (s2, a, b, c, d)
                    if sum(tj) % 2:
                        continue
                    imb, nsites = imbalances(tj)
                    if nsites < 2:
                        continue
                    tot += 1
                    found = False
                    for P in pairings6(tj):
                        im, _ = imbalances(P)
                        if min(im) >= 0:
                            found = True; pat[tuple(im)] += 1; break
                    if found:
                        ok += 1
                    else:
                        bad.append(tj)
    L = [f'2j1={s2}, all tuples with 2j<={M}: {tot}; with an all-tall ordered pairing: {ok}; without: {len(bad)}']
    if bad:
        L.append('first tuples without: ' + ' '.join(map(str, bad[:30])))
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

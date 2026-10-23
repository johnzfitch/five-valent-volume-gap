"""[Z] Lemma 10 and the 2j1 = 7 exceptions, from rb_core's site sets (independent).
For half-integer j1 = s, zigzag k has rows on d = -s + 2k and columns on d = -s + 1 + 2k (k = 0 .. s-1/2).
A pairing is all-tall if every zigzag has #rows - #cols in {0, 1}.  All 6 ordered pairings of legs 2-5 are tried
(pair order and which pair is first matter; swapping inside a pair does not change the site set).
Usage: python rb_alltall.py > ../outputs/rb_alltall_out.txt"""
import sys, os, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C


def alltall(tj):
    s2 = tj[0]
    S = C.sites_of(tj)
    cnt = {}
    for (B, Cc) in S:
        cnt[B - Cc] = cnt.get(B - Cc, 0) + 1                  # doubled d
    for k in range((s2 + 1) // 2):
        r = cnt.get(-s2 + 4 * k, 0); c = cnt.get(-s2 + 2 + 4 * k, 0)
        if r - c not in (0, 1):
            return False
    return True


def run(s2, M):
    n = nofail = 0; fails = []
    for sp in itertools.combinations_with_replacement(range(1, M + 1), 4):
        if (s2 + sum(sp)) % 2: continue
        if C.dim_inv((s2,) + sp) < 1: continue
        n += 1
        a, b, c, d = sp
        ok = False
        for (p, q) in (((a, b), (c, d)), ((c, d), (a, b)), ((a, c), (b, d)), ((b, d), (a, c)), ((a, d), (b, c)), ((b, c), (a, d))):
            if alltall((s2,) + p + q):
                ok = True; break
        if not ok: fails.append(sp)
    return n, fails


if __name__ == '__main__':
    t0 = time.time()
    n, f = run(5, 40)
    print(f"2j1=5: {n} tuples (multisets, 2j<=40, dim Inv>=1): without an all-tall ordered pairing: {len(f)} {f[:5]}")
    n, f = run(7, 30)
    print(f"2j1=7: {n} tuples (2j<=30): without an all-tall ordered pairing: {len(f)}; first {f[:6]}")
    n20 = sum(1 for x in f if max(x) <= 20)
    print(f"        of which with 2j<=20: {n20}  ([Z]: 29 with 2j<=20, 49 with 2j<=30)  ({time.time()-t0:.0f}s)")

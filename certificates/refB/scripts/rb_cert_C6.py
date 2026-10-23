"""Exhaustive exact certificates for the part of [LJ] Theorem C that rests on [FC]: every tuple of class C6 (three of
legs 2-5 with 2j <= 6) with 2j_max <= 38, target q >= 4.5579607 (statement (i); statement (ii) is implied there since
4.922951 j1 - 1370 < 0).  The three bounded legs and j5 are placed as (t1 t2 | t3 j5); the S4 symmetry of [DC] Prop. 1(d)
(proved there, tested in rb_minima) makes the placement irrelevant.  Usage: python rb_cert_C6.py > ../outputs/rb_cert_C6_out.txt"""
import sys, os, time, itertools
from fractions import Fraction as Fr
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_cert as RC
import rb_core as C

Q0 = Fr(45579607, 10 ** 7)


def job(tj):
    return tj, RC.certify(tj, Q0)[0]


if __name__ == '__main__':
    t0 = time.time()
    T = set()
    for t in itertools.combinations_with_replacement(range(1, 7), 3):
        for s2 in range(1, 39):
            for j5 in range(1, 39):
                tj = (s2, t[0], t[1], t[2], j5)
                if sum(tj) % 2 == 0 and C.dim_inv(tj) >= 2:
                    T.add(tj)
    T = sorted(T)
    st = {}; bad = []
    with Pool(8) as pool:
        for tj, r in pool.imap_unordered(job, T, chunksize=20):
            st[r] = st.get(r, 0) + 1
            if r not in ('certified', 'trivial'): bad.append((tj, r))
    print(f"C6 with 2j_max <= 38: {len(T)} tuples (dim Inv >= 2), exact target 4.5579607: {st}; failures {len(bad)} {bad[:10]}  ({time.time()-t0:.0f}s)")

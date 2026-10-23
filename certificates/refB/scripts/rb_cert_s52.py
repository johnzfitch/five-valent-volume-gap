"""Exact certificates (rb_cert) for j1 = 5/2 on ALL tuples with legs 2j <= MAX: targets 7.69 ([Z]: q >= 7.695 on 2j <= 40)
and 4.5579607 ([FJ] Thm 5.1 lower range, via [FC]).  Usage: python rb_cert_s52.py MAX > ../outputs/rb_cert_s52_out.txt"""
import sys, os, time
from fractions import Fraction as Fr
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_cert as RC
import rb_cert_run as RR


def job(tj):
    return tj, RC.certify(tj, Fr(769, 100))[0]


if __name__ == '__main__':
    M = int(sys.argv[1]); t0 = time.time()
    T = list(RR.all_tuples(5, M))
    st = {}; bad = []
    with Pool(8) as pool:
        for tj, r in pool.imap_unordered(job, T, chunksize=20):
            st[r] = st.get(r, 0) + 1
            if r not in ('certified', 'trivial'): bad.append((tj, r))
    print(f"j1=5/2, all {len(T)} tuples with legs 2j<={M} (sorted within pairs): target 7.69 -> {st}; failures {bad[:10]}  ({time.time()-t0:.0f}s)")

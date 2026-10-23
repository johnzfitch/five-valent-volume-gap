"""Exhaustive exact certificates (rb_cert) for one j1 on ALL tuples with legs 2j <= MAX (sorted within pairs and pairs ordered;
the S4 symmetry is not used).  Usage: python rb_cert_all.py 2J1 MAX TARGET_NUM TARGET_DEN"""
import sys, os, time
from fractions import Fraction as Fr
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_cert as RC
import rb_minima as RM

TGT = None


def job(tj):
    return tj, RC.certify(tj, TGT)[0]


def init(t):
    global TGT
    TGT = t


if __name__ == '__main__':
    s2, M, a, b = (int(x) for x in sys.argv[1:5]); tgt = Fr(a, b); t0 = time.time()
    T = RM.tuples(s2, M)
    st = {}; bad = []
    with Pool(8, initializer=init, initargs=(tgt,)) as pool:
        for tj, r in pool.imap_unordered(job, T, chunksize=20):
            st[r] = st.get(r, 0) + 1
            if r not in ('certified', 'trivial'): bad.append((tj, r))
    print(f"2j1={s2}: all {len(T)} tuples with legs 2j<={M} (j2<=j3, j4<=j5, (j2,j3)<=(j4,j5)), exact target q >= {float(tgt)}: {st}; failures {len(bad)} {bad[:10]}  ({time.time()-t0:.0f}s)", flush=True)

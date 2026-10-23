"""validate the triple elimination against brute force: spectrum of i*qhat_T (all triples incl. last leg)."""
import numpy as np, itertools
from rs_core import *
from hv_core import spin_mats
def brute_triple(Jd, T):
    sig = {T: 1}
    return brute_Qsigma_spectrum(Jd, sig)[0]
for Jd in [(1,1,1,1,2),(1,2,2,2,1),(2,2,2,2,2),(1,1,2,3,3)]:
    V = Vertex(Jd); Q = qhats(V)
    for T in itertools.combinations(range(5),3):
        sb = np.sort(np.abs(brute_triple(Jd,T)))
        # brute uses i*4 J.(JxJ) normalisation = |qhat|
        sc = np.sort(np.linalg.svd(Q[T],compute_uv=False))
        assert len(sb)==len(sc), (Jd,T,len(sb),len(sc))
        d = np.max(np.abs(sb-sc))
        if d>1e-9: print('MISMATCH',Jd,T,d)
    print(Jd,'dim',V.dim,'all 10 triples match brute force')

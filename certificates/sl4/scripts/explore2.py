import numpy as np, time
from vs_core import *
from scipy.linalg import eigh_tridiagonal
best={}
for J in sorted_tuples(80):
    c2=chain2_float(J); d=len(c2)+1
    if d<4: continue
    dg,o2=M_tridiag(list(c2))
    w=eigh_tridiagonal(np.array(dg),np.sqrt(np.array(o2)),eigvals_only=True)
    j4=J[3]/2; mu=j4*(j4+1)/4
    F=mu*np.sum(1/w); r=w[0]/mu
    if d not in best or F>best[d][0]: best[d]=(F,r,J)
for d in sorted(best): print(d, 'max mu*tr %.4f  (lam2/mu there %.3f)'%best[d][:2], best[d][2])

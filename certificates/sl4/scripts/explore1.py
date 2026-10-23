import numpy as np, time
from vs_core import *
t=time.time()
worst_r={}; worst_F={}
for J in sorted_tuples(60):
    c2=chain2_float(J); d=len(c2)+1
    if d<4: continue
    dg,o2=M_tridiag(list(c2))
    m=len(dg); M=np.diag(dg)
    for i in range(m-1): M[i,i+1]=M[i+1,i]=np.sqrt(o2[i])
    # gauge sign irrelevant for eigenvalues; actual BtB offdiag = +c c
    w=np.linalg.eigvalsh(M)
    j4=J[3]/2; mu=j4*(j4+1)/4
    r=w[0]/mu; F=mu*np.sum(1/w)
    key=(d%2, J[3])
    if key not in worst_r or r<worst_r[key][0]: worst_r[key]=(r,J)
    if key not in worst_F or F>worst_F[key][0]: worst_F[key]=(F,J)
for par in (0,1):
    print('parity',par)
    for J4 in range(3,61,3):
        if (par,J4) in worst_r: print(J4, 'min lam2/mu %.4f'%worst_r[(par,J4)][0], worst_r[(par,J4)][1], ' max mu*trMinv %.4f'%worst_F[(par,J4)][0], worst_F[(par,J4)][1])
print(time.time()-t)

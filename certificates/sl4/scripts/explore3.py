import numpy as np
from vs_core import *
best={}
for J in sorted_tuples(60):
    c2=chain2_float(J); d=len(c2)+1
    if d<4: continue
    c=np.sqrt(c2)
    j4=J[3]/2; mu=j4*(j4+1)/4
    S=np.sum(1/c); So=np.sum(1/c[0::2]); Se=np.sum(1/c[1::2])
    rp=max(1,np.max(c[1:]/c[:-1])) if len(c)>1 else 1; rm=max(1,np.max(c[:-1]/c[1:])) if len(c)>1 else 1
    P1=mu*rp*rm*S*S/4      # Lemma-1 style
    P2=mu*S*S/4            # if the ratio factor could be removed
    for key,val in (('P1',P1),('P2',P2)):
        kk=(key,min(d,12))
        if kk not in best or val>best[kk][0]: best[kk]=(val,J)
for k in sorted(best): print(k, '%.4f'%best[k][0], best[k][1])

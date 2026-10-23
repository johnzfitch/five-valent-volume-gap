import numpy as np
from vs_core import *
worst=9; fails=[]
for J in sorted_tuples(80):
    j=[x/2 for x in J]; kd=kdata(J); d=kd[2]
    if d<2: continue
    A1=j[0]+j[1]+1; B1=abs(j[0]-j[1]); A2=j[2]+j[3]+1; B2=abs(j[2]-j[3])
    kmin=kd[0]/2; kmax=kd[1]/2; M=max(B1,B2); beta=min(B1,B2)
    ks=np.arange(kmin+1,kmax+1e-9,1.0)
    f=(ks*ks-beta*beta)*(A2*A2-ks*ks)
    mu=j[3]*(j[3]+1)/4; lh2=1.0 if d%2==0 else 4.0
    if len(f)==1: bound=f[0]/16
    else:
        fmin=min(f[0],f[-1]); f1=f[0]; f2=f[1]
        endc=min(1.0,f[-1]/f[-2])
        if np.all(np.diff(f)>=0): bound=f1/16
        else: bound=(f1/16)*min(1,np.sqrt(fmin/f1))*min(1,np.sqrt(fmin/f2))*endc**2
    r=bound*lh2/mu
    if r<worst: worst=r; wJ=J
    if r<1-1e-12: fails.append((J,d,round(r,4)))
print('min crude-bound/mu',worst,wJ,'fails',len(fails)); print(fails[:20])

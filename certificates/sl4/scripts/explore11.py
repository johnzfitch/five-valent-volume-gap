import numpy as np
from vs_core import *
def proxy(f):
    L=len(f); F=lambda s: f[s-1]
    if L==1: return F(1)/16
    # a_min lower bound
    cand=[1.0]
    for t in range(3,L+1,2): cand.append(np.sqrt(F(t)/F(1)))
    if L%2==0: cand.append(np.sqrt(F(L-1)/F(1))*F(L)/F(L-1))
    amin=min(cand)
    cand=[1.0]
    for t in range(2,L+1,2): cand.append(np.sqrt(F(t)/F(2)))
    if L%2==1 and L>=3: cand.append(np.sqrt(F(L-1)/F(2))*F(L)/F(L-1))
    bmin=F(1)/16*min(cand)
    return amin*bmin
worst=9; fails=[]
for J in sorted_tuples(80):
    j=[x/2 for x in J]; kd=kdata(J); d=kd[2]
    if d<2: continue
    A1=j[0]+j[1]+1; B1=abs(j[0]-j[1]); A2=j[2]+j[3]+1; B2=abs(j[2]-j[3])
    kmin=kd[0]/2; kmax=kd[1]/2; M=max(B1,B2); beta=min(B1,B2)
    ks=np.arange(kmin+1,kmax+1e-9,1.0)
    f=(ks*ks-beta*beta)*(A2*A2-ks*ks)
    mu=j[3]*(j[3]+1)/4; lh2=1.0 if d%2==0 else 4.0
    r=proxy(f)*lh2/mu
    if r<worst: worst=r; wJ=J
    if r<1-1e-12: fails.append((J,d,round(r,4)))
print('min proxy/mu',worst,wJ,'fails',len(fails)); print(fails[:20])

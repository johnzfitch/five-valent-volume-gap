import numpy as np, collections
from vs_core import *
stats=collections.Counter(); ex={}
for J in sorted_tuples(80):
    j=[x/2 for x in J]; kd=kdata(J); d=kd[2]
    if d<3: continue
    A1=j[0]+j[1]+1; B1=abs(j[0]-j[1]); A2=j[2]+j[3]+1; B2=abs(j[2]-j[3])
    kmin=kd[0]/2; kmax=kd[1]/2; M=max(B1,B2); beta=min(B1,B2)
    ks=np.arange(kmin+1,kmax+1e-9,1.0)
    f=(ks*ks-beta*beta)*(A2*A2-ks*ks); L=len(f)
    mu=j[3]*(j[3]+1)/4
    c1=f[-1]>=f[0]; c2=f[-1]>=f[1]; c3=f[-1]**2>=f[-2]*f[1] if L>=3 else True
    c0=f[1]>=f[0]
    key=(c0,c1,c2,c3)
    stats[key]+=1
    marg=f[0]/16/mu
    if key not in ex or marg<ex[key][0]: ex[key]=(marg,J,d)
for k,v in stats.items(): print(k,v,'min f1/(16mu)=%.3f'%ex[k][0],ex[k][1],'d',ex[k][2])

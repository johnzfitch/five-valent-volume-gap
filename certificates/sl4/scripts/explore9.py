import numpy as np
from vs_core import *
def lemS(g):
    # g: link multipliers g_1..g_{n-1}; returns sqrt(min a * min b) (forward orientation)
    g=np.asarray(g,float); n=len(g)+1
    a=[1.0]; b=[]
    # a_i for i=0.. : a_i = prod_{l<=i} (g_{2l}/g_{2l-1})^2 ; b_i = g_1^2 prod_{l=1}^{i-1}(g_{2l+1}/g_{2l})^2
    la=0.0; amin=1.0
    for l in range(1,n):
        if 2*l-1>=1 and 2*l<=n-1:
            la+=2*(np.log(g[2*l-1])-np.log(g[2*l-2])); amin=min(amin,np.exp(la))
        else: break
    lb=2*np.log(g[0]); bmin=np.exp(lb)
    for l in range(1,n):
        if 2*l+1<=n-1:
            lb+=2*(np.log(g[2*l])-np.log(g[2*l-1])); bmin=min(bmin,np.exp(lb))
        else: break
    return np.sqrt(amin*bmin)
worst=9; worstJ=None; fails=[]
for J in sorted_tuples(60):
    j=[x/2 for x in J]; kd=kdata(J); d=kd[2]
    if d<2: continue
    A1=j[0]+j[1]+1; B1=abs(j[0]-j[1]); A2=j[2]+j[3]+1; B2=abs(j[2]-j[3])
    kmin=kd[0]/2; kmax=kd[1]/2
    M=max(B1,B2); beta=min(B1,B2)
    ks=np.arange(kmin+1,kmax+1e-9,1.0)
    w=np.sqrt((ks*ks-beta*beta)*(A2*A2-ks*ks))/4
    lam_h=1.0 if d%2==0 else 2.0
    bnd=max(lemS(w),lemS(w[::-1]))*lam_h
    mu=j[3]*(j[3]+1)/4
    r=bnd**2/mu
    if r<worst: worst=r; worstJ=J
    if r<1-1e-12: fails.append((J,d,r))
print('min bound^2/mu =',worst,worstJ,' fails:',len(fails)); print(fails[:15])

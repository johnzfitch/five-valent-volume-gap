import numpy as np
from scipy.linalg import eigh_tridiagonal
from vs_core import *
# 1. hydrogen chain h(n,m): links l=m+1..n-1, h^2=(n^2-l^2)(l^2-m^2)/(4l^2-1); eigenvalues should be n1-n2, n1+n2=n-m-1
for (n,m) in [(3,0),(5,0),(5,2),(6,1),(7.5,0.5),(8.5,1.5),(10,3.0),(4.5,0.5)]:
    ls=np.arange(m+1,n-1+1e-9,1.0)
    h=np.sqrt((n*n-ls*ls)*(ls*ls-m*m)/(4*ls*ls-1))
    w=np.sort(eigh_tridiagonal(np.zeros(len(h)+1),h,eigvals_only=True))
    print(n,m,np.round(w,10))
# 2. monotone regime vs not; lossy bound w_min^2/w_max in non-monotone cases
cnt=0;nonmono=0;fail_lossy=[];minmarg_nm=9
for J in sorted_tuples(60):
    J1,J2,J3,J4=J
    j=[x/2 for x in J]; A1=j[0]+j[1]+1; B1=abs(j[0]-j[1]); A2=j[2]+j[3]+1; B2=abs(j[2]-j[3])
    kd=kdata(J); kmin=kd[0]/2; kmax=kd[1]/2; d=kd[2]
    if d<2: continue
    M=max(B1,B2); beta=min(B1,B2)
    ks=np.arange(kmin+1,kmax+1e-9,1.0)
    f=(ks*ks-beta*beta)*(A2*A2-ks*ks)
    cnt+=1
    mu=j[3]*(j[3]+1)/4
    if np.all(np.diff(f)>=0): 
        assert f[0]/16>=mu*(1-1e-12), J
        continue
    nonmono+=1
    lossy=(f.min()/16)**2/(f.max()/16)   # (w_min^2/w_max)^2
    l2=lam2_float(J)
    minmarg_nm=min(minmarg_nm,l2/mu)
    if lossy<mu: fail_lossy.append((J,d,lossy/mu,l2/mu))
print('tuples',cnt,'non-monotone',nonmono,'min lam2/mu among non-monotone',minmarg_nm)
print('lossy bound fails in',len(fail_lossy)); print(fail_lossy[:10]); 
if fail_lossy: print('max d among fails',max(x[1] for x in fail_lossy))

import numpy as np
from vs_core import *
def R1(c):  # c: even-length-site chain c_1..c_{2m-1} (len odd), returns |u|^2 |v|^2 rank-one bound on ||B^-1||^2
    n=len(c)+1; m=n//2
    C=lambda s: c[s-1]
    if m==1: return 1/C(1)**2
    u2=sum(1/C(2*i) for i in range(1,m))+C(2*m-2)/C(2*m-1)**2
    v2=sum(C(2*j)/C(2*j-1)**2 for j in range(1,m))+1/C(2*m-2)
    return u2*v2
best={}
for J in sorted_tuples(60):
    c2=chain2_float(J); d=len(c2)+1
    if d<4: continue
    c=np.sqrt(c2); j4=J[3]/2; mu=j4*(j4+1)/4
    if d%2==0: val=mu*R1(c)
    else: val=mu*min(R1(c[:-1]),R1(c[1:][::-1]) if False else R1(c[1:]))
    key=min(d,14)
    if key not in best or val>best[key][0]: best[key]=(val,J)
for k in sorted(best): print(k,'%.4f'%best[k][0],best[k][1])

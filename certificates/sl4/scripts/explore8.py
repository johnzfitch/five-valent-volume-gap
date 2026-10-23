import numpy as np
from vs_core import *
def pairing_bound(j, P):
    (a,b),(c,d)=P
    A=[j[a]+j[b]+1, j[c]+j[d]+1]; B=[abs(j[a]-j[b]), abs(j[c]-j[d])]
    As,Ab=min(A),max(A); M,beta=max(B),min(B)
    if M>As-1: return None
    ks=np.arange(M+1,As-1+1e-9,1.0)
    if len(ks)==0: return None
    f=(ks*ks-beta*beta)*(Ab*Ab-ks*ks)
    return bool(np.all(np.diff(f)>=0)), f[0]/16
PAIRS=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
uncovered=[]; n=0
for J in sorted_tuples(60):
    j=[x/2 for x in J]; d=kdata(J)[2]
    if d<2: continue
    n+=1
    mu=j[3]*(j[3]+1)/4
    ok=False
    for P in PAIRS:
        r=pairing_bound(j,P)
        if r and r[0] and r[1]>=mu*(1-1e-12): ok=True;break
    if not ok: uncovered.append((J,d))
print(n,'uncovered',len(uncovered)); print(uncovered[:20]); 
import collections; print(collections.Counter(d for _,d in uncovered).most_common(12))

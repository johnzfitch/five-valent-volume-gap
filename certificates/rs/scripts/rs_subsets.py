import numpy as np, itertools
from rs_core import *
trip=list(itertools.combinations(range(5),3))
res={}
M=10
for J5 in range(1,M+1):
 for J4 in range(1,J5+1):
  for J3 in range(1,J4+1):
   for J2 in range(1,J3+1):
    for J1 in range(1,J2+1):
     if (J1+J2+J3+J4+J5)%2: continue
     V=Vertex((J1,J2,J3,J4,J5))
     if V.dim<2: continue
     Q=qhats(V); A={T:absm(m) for T,m in Q.items()}
     full=sum(A.values()); wf=np.linalg.eigvalsh(full); kf=int(np.sum(wf<1e-8*wf.max()))
     for r in (2,3):
      for sub in itertools.combinations(trip,r):
        S=sum(A[t] for t in sub); w=np.linalg.eigvalsh(S); tol=1e-8*max(w.max(),1)
        k=int(np.sum(w<tol)); nz=w[w>tol]
        key=sub
        ok=(k==kf)
        cur=res.get(key,[True,1e9])
        cur[0]=cur[0] and ok
        if ok and len(nz): cur[1]=min(cur[1],nz[0]/ (J5/2+1))
        res[key]=cur
good=[(v[1],k) for k,v in res.items() if v[0]]
good.sort(reverse=True)
print('subsets with the same kernel as Q_RS for all tuples 2j<=%d:'%M,len(good))
for g in good[:12]: print(' min lambda/(jmax+1)=%.3f'%g[0], g[1])
pairs=[(v[1],k) for k,v in res.items() if v[0] and len(k)==2]
print('pairs with full kernel match:',len(pairs)); pairs.sort(reverse=True); print(pairs[:10])
bad=[k for k,v in res.items() if not v[0] and len(k)==2]
print('pairs failing:',len(bad), bad[:10])

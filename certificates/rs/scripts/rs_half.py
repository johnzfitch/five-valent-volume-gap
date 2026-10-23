"""Tuples with a spin-1/2 leg (leg 1, 0-based 0): subsets of the 6 triples containing leg 1.
Report: kernel dims of each such |qhat|, which subsets reproduce ker Q_RS for all tuples, and min nonzero eig of
sum of their range projectors (the angle quantity)."""
import numpy as np, itertools, json
from rs_core import *
def rangeproj(A):
    w,U=np.linalg.eigh(1j*A); tol=1e-9*max(1,np.abs(w).max()); Uk=U[:,np.abs(w)>tol]; return (Uk@Uk.conj().T).real
T1=[t for t in itertools.combinations(range(5),3) if 0 in t]
M=30
stats={}; maxker=0; nt=0
for J5 in range(1,M+1):
 for J4 in range(1,J5+1):
  for J3 in range(1,J4+1):
   for J2 in range(1,J3+1):
     Jd=(1,J2,J3,J4,J5)
     if sum(Jd)%2: continue
     V=Vertex(Jd)
     if V.dim<1: continue
     nt+=1
     Q=qhats(V); A={T:absm(m) for T,m in Q.items()}
     wf=np.linalg.eigvalsh(sum(A.values())); kf=int(np.sum(wf<1e-9*max(1,wf.max())))
     P={t:rangeproj(Q[t]) for t in T1}
     for t in T1: maxker=max(maxker,V.dim-int(round(np.trace(P[t]))))
     for r in (2,3,6):
      for sub in itertools.combinations(T1,r):
        S=sum(P[t] for t in sub); w=np.linalg.eigvalsh(S); k=int(np.sum(w<1e-8)); nz=w[w>1e-8]
        st=stats.setdefault(sub,[True,9.0])
        st[0]=st[0] and (k==kf)
        if k==kf and len(nz): st[1]=min(st[1],nz[0])
print('tuples with j1=1/2, 2jmax<=%d:'%M,nt,' max kernel dim of a leg-1 triple:',maxker)
ok=sorted([(v[1],k) for k,v in stats.items() if v[0]],reverse=True)
print('subsets reproducing ker Q_RS everywhere:',len(ok))
for v,k in ok[:10]: print('  min lambda+(sum P) = %.4f'%v,k)

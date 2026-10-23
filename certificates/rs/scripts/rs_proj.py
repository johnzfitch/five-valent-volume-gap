import numpy as np, itertools, sys
from rs_core import *
def rangeproj(A):
    w,U=np.linalg.eigh(1j*A); tol=1e-9*max(1,np.abs(w).max()); Uk=U[:,np.abs(w)>tol]; return (Uk@Uk.conj().T).real
M=int(sys.argv[1]) if len(sys.argv)>1 else 12
worst=9;wj=None;minpair={}
for J5 in range(1,M+1):
 for J4 in range(1,J5+1):
  for J3 in range(1,J4+1):
   for J2 in range(1,J3+1):
    for J1 in range(1,J2+1):
     if (J1+J2+J3+J4+J5)%2: continue
     V=Vertex((J1,J2,J3,J4,J5))
     if V.dim==0: continue
     Q=qhats(V); P={T:rangeproj(A) for T,A in Q.items()}
     S=sum(P.values()); w=np.linalg.eigvalsh(S); nz=w[w>1e-9]
     if len(nz) and nz[0]<worst: worst=nz[0]; wj=(J1,J2,J3,J4,J5,V.dim,len(w)-len(nz))
print('min nonzero eig of sum of range projectors:',worst,wj)

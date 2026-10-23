import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))
from hv_core import Vertex
def casimir(V, legs, spins2):
    d=V.dim; M=np.eye(d)*sum((s/2)*(s/2+1) for s in spins2)
    for a in range(len(legs)):
        for b in range(a+1,len(legs)): M=M+2*V.dot(legs[a],legs[b])
    return M
lab=lambda U,M,i: round((-1+np.sqrt(1+4*(U[:,i]@M@U[:,i])))/2*2)/2
for n in (5,6):
    V=Vertex((1,1,n,n,n+1,n+1))
    Xr=V.qhat(0,1,2)/4; Yr=V.qhat(1,3,4)/4      # A = 2 qhat123 - 2 qhat245 = 8 (Xr - Yr) as real matrices
    C13=casimir(V,[0,2],[1,n]); H=casimir(V,[1,3,4],[1,n,n+1]); E=casimir(V,[3,4],[n,n+1])
    w,U=np.linalg.eigh(C13+np.pi*H+np.e**2*E)
    keys=[]
    for i in range(V.dim):
        c=lab(U,C13,i); h=lab(U,H,i); e=lab(U,E,i)
        r=-1 if c<n/2 else 1; ie=int(e-0.5); side='L' if abs(h-(e-0.5))<1e-9 else 'R'
        keys.append((r,ie,side))
    idx={k:i for i,k in enumerate(keys)}
    Xb=U.T@Xr@U; Yb=U.T@Yr@U
    sg=np.ones(V.dim)
    # gauge: make Y(r,ie,R ; r,ie+1,L) > 0 by fixing signs left to right per row
    for r in (-1,1):
        for ie in range(n+1):
            for side in ('L','R'):
                k=(r,ie,side)
                if k not in idx: continue
                if side=='L' and (r,ie-1,'R') in idx:
                    a=idx[(r,ie-1,'R')]; b=idx[k]
                    if sg[a]*sg[b]*Yb[a,b]<0: sg[b]=-sg[b]
    D=np.diag(sg); Xg=D@Xb@D; Yg=D@Yb@D
    print('n',n,'Y signs all positive:',all(Yg[idx[(r,ie,'R')],idx[(r,ie+1,'L')]]>0 for r in (-1,1) for ie in range(n) if (r,ie,'R') in idx and (r,ie+1,'L') in idx))
    for ie in range(n+1):
        rows=[(-1,ie,'L'),(-1,ie,'R')]; cols=[(1,ie,'L'),(1,ie,'R')]
        M=[[ (Xg[idx[a],idx[b]] if a in idx and b in idx else 0) for b in cols] for a in rows]
        print('  e=%d+1/2'%ie, np.round(np.array(M),5))

# compare: gauge-fixed true K = Xg - Yg vs model
from ladder import model, states
n=6
V=Vertex((1,1,n,n,n+1,n+1)); Xr=V.qhat(0,1,2)/4; Yr=V.qhat(1,3,4)/4
Ktrue=Xr-Yr
print('true |spec|*8:', np.sort(np.linalg.svd(8*Ktrue,compute_uv=False))[:6])
for sY in ((1,1),(-1,-1)):
    for sx in (1,-1):
        K,S=model(n,sY,ctx=np,signX=sx)
        print(sY,sx,np.sort(np.linalg.svd(8*K,compute_uv=False))[:6])

import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))
from hv_core import Vertex
from ladder import model
def casimir(V, legs, spins2):
    d=V.dim; M=np.eye(d)*sum((s/2)*(s/2+1) for s in spins2)
    for a in range(len(legs)):
        for b in range(a+1,len(legs)): M=M+2*V.dot(legs[a],legs[b])
    return M
lab=lambda U,M,i: round((-1+np.sqrt(1+4*(U[:,i]@M@U[:,i])))/2*2)/2
n=6
V=Vertex((1,1,n,n,n+1,n+1)); Xr=V.qhat(0,1,2)/4; Yr=V.qhat(1,3,4)/4
C13=casimir(V,[0,2],[1,n]); H=casimir(V,[1,3,4],[1,n,n+1]); E=casimir(V,[3,4],[n,n+1])
w,U=np.linalg.eigh(C13+np.pi*H+np.e**2*E)
keys=[]
for i in range(V.dim):
    c=lab(U,C13,i); h=lab(U,H,i); e=lab(U,E,i)
    keys.append((-1 if c<n/2 else 1,int(e-0.5),'L' if abs(h-(e-0.5))<1e-9 else 'R'))
K,S=model(n,(1,1),ctx=np)
idxm={s:k for k,s in enumerate(S)}
Xb=U.T@Xr@U; Yb=U.T@Yr@U
for nm,Mt in (('X',Xb),('Y',Yb)):
    for a in range(V.dim):
        for b in range(a+1,V.dim):
            if abs(Mt[a,b])>1e-9:
                mv=abs(K[idxm[keys[a]],idxm[keys[b]]])
                if abs(abs(Mt[a,b])-mv)>1e-9: print(nm,keys[a],keys[b],'true',round(abs(Mt[a,b]),6),'model',round(mv,6))

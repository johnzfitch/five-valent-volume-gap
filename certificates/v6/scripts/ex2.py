"""Confirm the ladder picture: joint eigenbasis of C13^2, G^2=(J1+J2+J3)^2, E^2=(J4+J5)^2 in the full recoupling space."""
import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))
from hv_core import Vertex
def casimir(V, legs, spins2):
    d=V.dim; M=np.eye(d)*sum((s/2)*(s/2+1) for s in spins2)
    for a in range(len(legs)):
        for b in range(a+1,len(legs)):
            M=M+2*V.dot(legs[a],legs[b])
    return M
for n in (3,4,5):
    # leg order as given: 1,2,3,4,5 (0-based 0..4), spins (1,1,n,n,n+1 | n+1)
    Jd=(1,1,n,n,n+1,n+1); V=Vertex(Jd)
    A=2*V.qhat(0,1,2)-2*V.qhat(1,3,4)
    Xr=V.qhat(0,1,2)/(-4); Yr=V.qhat(1,3,4)/(-4)   # real antisym: qhat = -4 i J.(JxJ) -> stored real matrix
    C13=casimir(V,[0,2],[1,n]); G=casimir(V,[0,1,2],[1,1,n]); E=casimir(V,[3,4],[n,n+1]); H=casimir(V,[1,3,4],[1,n,n+1])
    # joint diagonalisation: generic combination
    w,U=np.linalg.eigh(C13+np.pi*G+np.e**2*E)
    lab=lambda M: np.round([ (-1+np.sqrt(1+4*(U[:,i]@M@U[:,i])))/2 for i in range(U.shape[1])],6)
    c=lab(C13); g=lab(G); e=lab(E)
    Xb=U.T@Xr@U; Yb=U.T@Yr@U
    print('n',n,'dim',V.dim)
    nzX=[(c[i],g[i],e[i],c[j],g[j],e[j],round(abs(Xb[i,j]),6)) for i in range(len(c)) for j in range(i+1,len(c)) if abs(Xb[i,j])>1e-9]
    print(' X links (c,g,e)-(c,g,e) |w|:',nzX[:8], ' predicted |x| =',np.sqrt(n*(n+2))/4)
    w2,U2=np.linalg.eigh(C13+np.pi*H+np.e**2*E)
    c2=np.round([(-1+np.sqrt(1+4*(U2[:,i]@C13@U2[:,i])))/2 for i in range(V.dim)],6)
    h2=np.round([(-1+np.sqrt(1+4*(U2[:,i]@H@U2[:,i])))/2 for i in range(V.dim)],6)
    e2=np.round([(-1+np.sqrt(1+4*(U2[:,i]@E@U2[:,i])))/2 for i in range(V.dim)],6)
    Yb2=U2.T@Yr@U2
    nzY=[(c2[i],h2[i],e2[i],e2[j],round(abs(Yb2[i,j]),6), round(np.sqrt(((n+1.5)**2-(max(e2[i],e2[j]))**2)*(max(e2[i],e2[j])**2-0.25))/4,6)) for i in range(V.dim) for j in range(i+1,V.dim) if abs(Yb2[i,j])>1e-9]
    print(' Y links (c,h,e,e\') |w| pred:',nzY[:6])
    zeroY=[(c2[i],h2[i],e2[i]) for i in range(V.dim) if np.all(np.abs(Yb2[i])<1e-9)]
    print(' Y zero rows:',zeroY)

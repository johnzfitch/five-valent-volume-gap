import numpy as np
from stark import stark
def data(N,m):
    H,g=stark(N,m); lam,U=np.linalg.eigh(H)
    v0=U[0,:]; vN=U[-1,:]
    G=lambda w,z: np.sum(w/(lam-z))
    X=lambda z: np.sum(v0*vN/(lam-z))
    return lam,v0,vN,G(v0**2,2),G(vN**2,2),X(0),X(2)
for N in [2,4,6,10,20,40]:
    for m in [0,0.5,2,10,1e3]:
        lam,v0,vN,G0,GN,X0,X2=data(N,m)
        print(N,m,'ratio (X0+X2)^2/(G0 GN)=%.4f'%((X0+X2)**2/(G0*GN)), 'G0=%.3g GN=%.3g'%(G0,GN))

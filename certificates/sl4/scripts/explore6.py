import numpy as np
from fractions import Fraction as Fr
from scipy.linalg import eigh_tridiagonal
def lam2(c2):
    n=len(c2)+1
    w=eigh_tridiagonal(np.zeros(n),np.sqrt(np.array(c2,float)),eigvals_only=True)
    w=np.abs(w); w=w[w>1e-9*w.max()]; return w.min()**2
def probe(J1,J2,B2x2):
    # doubled spins; probe chain ĉ^2 = F1(k)(k^2-B2^2)/(4k^2-1), k from kmin+1 to j1+j2
    A1=(J1+J2+2)/2; B1=abs(J1-J2)/2; B2=B2x2/2
    kmin=max(B1,B2); kmax=(J1+J2)/2
    ks=np.arange(kmin+1,kmax+1e-9,1.0)
    return [(A1*A1-k*k)*(k*k-B1*B1)*(k*k-B2*B2)/(4*k*k-1) for k in ks]
worst={}
for J2 in range(1,61):
  for J1 in range(1,J2+1):
    for B in range(0,J1+J2+1):
      if (J1+J2-B)%2: continue
      c2=probe(J1,J2,B)
      d=len(c2)+1
      if d<4: continue
      # minimal j4 = j2 + B2/2 (j3=j2)
      j4=(J2+B)/2; j1=J1/2; j2=J2/2
      Phi=4*j4*(j4+1)/((j2+j4+1)**2-(j1+j2)**2)
      l2=lam2(c2) if d%2==0 else None
      if d%2==1:
          l2=max(lam2(c2[1:]),lam2(c2[:-1]))
      val=l2/Phi
      key=min(d,9)
      if key not in worst or val<worst[key][0]: worst[key]=(val,(J1,J2,B))
for k in sorted(worst): print(k, 'min lamhat^2/Phi* = %.4f'%worst[k][0], worst[k][1])

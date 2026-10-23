import numpy as np
from scipy.linalg import eigh_tridiagonal
rng=np.random.default_rng(1)
def lam(c):
    n=len(c)+1; J=eigh_tridiagonal(np.zeros(n),c,eigvals_only=True); return np.min(np.abs(J[np.abs(J)>1e-12*np.max(np.abs(J))]))
bad=0;tot=0;worst=1
for trial in range(200000):
    n=rng.integers(3,12)
    c=np.exp(rng.normal(0,1.5,n-1))
    r=np.cumprod(np.exp(np.abs(rng.normal(0,0.7,n-1))))   # increasing >= 1
    r=r/r[0]*np.exp(abs(rng.normal(0,0.5)))
    l0=lam(c); l1=lam(c*np.sqrt(r))
    tot+=1
    if l1<l0*(1-1e-10): bad+=1; worst=min(worst,l1/l0)
print('increasing multipliers: violations',bad,'of',tot,'worst ratio',worst)

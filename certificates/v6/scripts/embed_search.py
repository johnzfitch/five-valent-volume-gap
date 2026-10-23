"""Find integer tangent vectors e1..e6 (no three coplanar) realising BR's valence-6 sigma=(2,0,0,0,0,0,0,0,-2,0)
(BR ordering 123,124,125,134,135,145,234,235,245,345; sigma(IJK)=eps(IJK)-eps(IJ6)+eps(IK6)-eps(JK6))."""
import itertools, numpy as np
from fractions import Fraction
rng=np.random.default_rng(7)
tr=list(itertools.combinations(range(5),3))
target=(2,0,0,0,0,0,0,0,-2,0)
def det(a,b,c): return int(round(np.linalg.det(np.array([a,b,c],dtype=float))))
def detx(a,b,c):
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
found=None
for t in range(2000000):
    V=[tuple(int(v) for v in rng.integers(-4,5,3)) for _ in range(6)]
    eps={}
    ok=True
    for T in itertools.combinations(range(6),3):
        d=detx(*[V[i] for i in T])
        if d==0: ok=False;break
        eps[T]=1 if d>0 else -1
    if not ok: continue
    sig=tuple(eps[(I,J,K)]-eps[(I,J,5)]+eps[(I,K,5)]-eps[(J,K,5)] for (I,J,K) in tr)
    if sig==target: found=V;break
print('trials',t,'found',found)
if found:
    V=found
    print('all 20 triple determinants:',[detx(*[V[i] for i in T]) for T in itertools.combinations(range(6),3)])

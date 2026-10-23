import numpy as np, itertools, collections, sys
from s1 import Qmat
M=int(sys.argv[1])
fam={'lin':lambda b,beta: b-beta, 'sqrt':lambda b,beta: np.sqrt(np.maximum(b*b-beta*beta,0)+1e-300)}
cov=collections.Counter(); fails=[]
for a,b,c,d in itertools.product(range(1,M+1),repeat=4):
    if a>b or c>d or (a,b)>(c,d) or (2+a+b+c+d)%2 or (min(a,b)+min(c,d))%2 or a==1 or c==1: continue
    n1,m1,n2,m2=(a+b)/2+1,abs(a-b)/2,(c+d)/2+1,abs(c-d)/2
    if (m1-m2)%1 or max(m1,m2)>min(n1,n2)-1: continue
    Q,bs=Qmat(n1,m1,n2,m2)
    ident=(n1,m1)==(n2,m2)
    if ident and m1>0: cov['identical m>0 (kernel; skip)']+=1; continue
    lo=bs[0]; found=None
    for name,f in fam.items():
        for beta in np.linspace(-lo, lo-1e-6, 41) if name=='lin' else np.linspace(0, lo-1e-6, 41):
            g=f(bs,beta)
            if g.min()<=0: continue
            if (Q@g-g).min()>=-1e-9: found=(name,round(beta/lo,3) if lo else 0); break
        if found: break
    if found: cov[found[0]]+=1
    else: cov['none']+=1; fails.append((n1,m1,n2,m2))
print(dict(cov)); print(fails[:20])

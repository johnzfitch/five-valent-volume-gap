import numpy as np, itertools, collections, sys
from s1 import Qmat
M=int(sys.argv[1])
cov=collections.Counter(); fails=[]
def cands(n1,m1,n2,m2,lo,hi):
    mn,mx=min(m1,m2),max(m1,m2); Nn,Nx=min(n1,n2),max(n1,n2)
    As=sorted({0.0,max(mn-1,0),max(mn-.5,0),mn if mn<lo else max(mn-.5,0)})
    Bs=sorted({Nx,Nx+.5,Nn+.5,Nn+1,Nx+1})
    for A in As:
        for B in Bs:
            if A<lo and B>hi: yield (A,B)
for a,b,c,d in itertools.product(range(1,M+1),repeat=4):
    if a>b or c>d or (a,b)>(c,d) or (2+a+b+c+d)%2 or (min(a,b)+min(c,d))%2 or a==1 or c==1: continue
    n1,m1,n2,m2=(a+b)/2+1,abs(a-b)/2,(c+d)/2+1,abs(c-d)/2
    if (m1-m2)%1 or max(m1,m2)>min(n1,n2)-1: continue
    Q,bs=Qmat(n1,m1,n2,m2); lo,hi=bs[0],bs[-1]
    if (n1,m1)==(n2,m2) and m1>0: cov['identical m>0']+=1; continue
    found=None
    for A,B in cands(n1,m1,n2,m2,lo,hi):
        g=np.sqrt((B*B-bs*bs)*(bs*bs-A*A))
        if (Q@g-g).min()>=-1e-9: found=(A,B); break
    if found: cov['ok']+=1
    else: cov['none']+=1; fails.append((n1,m1,n2,m2))
print(dict(cov)); print(fails[:15])

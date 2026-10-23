"""Closed-form ladder model for the valence-6 BR family sigma=(2,0,0,0,0,0,0,0,-2,0), 2j=(1,1,n,n,n+1,n+1).
Basis (row r in {-,+}, e in {1/2..n+1/2}, side in {L: h=e-1/2, R: h=e+1/2}); allowed: row - needs 1<=h<=n,
row + needs 0<=h<=n+1.  K = X - s_Y * Y (real antisymmetric), spectrum of A = 8 * |spec(K)| (A = -8i(X-Y)).
X = x * sum_e (u_+(e) u_-(e)^T - u_-(e) u_+(e)^T), x = sqrt(n(n+2))/4,
   u_-(e) = alpha_-(e) L_- + beta_-(e) R_-, u_+(e) = alpha_+(e) L_+ + beta_+(e) R_+  (alpha_+ <= 0); alpha_+^2=(n+3/2-e)/(2n+4), beta_+^2=(n+5/2+e)/(2n+4).
Y = sum_{r,h=1..n} y_h (|r,h-1/2,R><r,h+1/2,L| - transpose), y_h = sqrt(((n+3/2)^2-(h+1/2)^2)((h+1/2)^2-1/4))/4.
The sign pattern (sY, and the relative sign of the two rows' Y) is fixed by comparison with the full operator.
"""
import numpy as np, mpmath as mp

def states(n):
    S=[]
    for r in (-1,1):
        for i in range(n+1):
            e=i+0.5
            for side,h in (('L',e-0.5),('R',e+0.5)):
                if r==-1 and 1<=h<=n: S.append((r,i,side))
                if r==1 and 0<=h<=n+1: S.append((r,i,side))
    return S

def model(n, sgnY=(1,1), ctx=mp, signX=1):
    S=states(n); idx={s:k for k,s in enumerate(S)}; D=len(S)
    K=ctx.matrix(D,D) if ctx is mp else np.zeros((D,D))
    x=ctx.sqrt(n*(n+2))/4
    for i in range(n+1):
        e=ctx.mpf(i)+ctx.mpf(1)/2 if ctx is mp else i+0.5
        am=ctx.sqrt((n+e+1.5)*(e-0.5)/(n*(2*e+1))); bm=ctx.sqrt((n+0.5-e)*(e+1.5)/(n*(2*e+1)))
        ap=-ctx.sqrt((n+1.5-e)/(2*n+4)); bp=ctx.sqrt((n+2.5+e)/(2*n+4))
        um={}; up={}
        if (-1,i,'L') in idx: um[idx[(-1,i,'L')]]=am
        if (-1,i,'R') in idx: um[idx[(-1,i,'R')]]=bm
        up[idx[(1,i,'L')]]=ap; up[idx[(1,i,'R')]]=bp
        for a,va in up.items():
            for b,vb in um.items():
                K[a,b]+=signX*x*va*vb; K[b,a]-=signX*x*va*vb
    for r,sg in zip((-1,1),sgnY):
        for h in range(1,n+1):
            k=h+0.5
            y=ctx.sqrt(((n+1.5)**2-k*k)*(k*k-0.25))/4
            a=idx[(r,h-1,'R')]; b=idx[(r,h,'L')]
            K[a,b]+=sg*y; K[b,a]-=sg*y
    return K,S

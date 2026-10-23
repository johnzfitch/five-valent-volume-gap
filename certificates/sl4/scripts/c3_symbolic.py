"""C3 (f(K)^2 >= f(K-1) f(M+2)) as a polynomial inequality; exact sympy analysis."""
import sympy as sp
K,M,b,E=sp.symbols('K M b E',real=True)
p=(2*K-1)*((K-1)**2-(M+2)**2-E)
q=(K**2-(M+2)**2)*E
T=(K**2-b**2)*(E+(M+2)**2-b**2)
F=sp.expand(p*q-T*(p-q))
print('deg in E:',sp.degree(F,E))
# region: 0<=b<=M, K>=M+3, Emin=2K(M+1)-2M-3+b^2 <= E <= Emax=(K-1)^2-(M+2)^2 (p>=0)
x,y,z,w,s=sp.symbols('x y z w s',nonnegative=True)
# b = y, M = y + z, K = M + 3 + x, E = Emin + w  (w>=0), and also need E<=Emax
sub={b:y, M:y+z, K:y+z+3+x}
Emin=sp.expand((2*K*(M+1)-2*M-3+b**2).subs(sub))
Emax=sp.expand(((K-1)**2-(M+2)**2).subs(sub))
G=sp.expand(F.subs(sub).subs(E,Emin+w))
P=sp.Poly(G,x,y,z,w)
neg=[(m,c) for m,c in zip(P.monoms(),P.coeffs()) if c<0]
print('terms',len(P.coeffs()),'negative-coefficient terms:',len(neg))
print(neg[:20])
gap=sp.expand(Emax-Emin); print('Emax-Emin =',sp.factor(gap))

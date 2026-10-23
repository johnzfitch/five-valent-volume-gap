"""q_min of the ladder model (mpmath, high precision) vs the sign-independent transfer bound B(n) (odd n)."""
import mpmath as mp
from ladder import model
mp.mp.dps=80
def coeffs(n):
    n=mp.mpf(n); x=mp.sqrt(n*(n+2))/4
    ap=lambda i: mp.sqrt((n+1-i)/(2*n+4)); bp=lambda i: mp.sqrt((n+3+i)/(2*n+4))
    am=lambda i: mp.sqrt((n+2+i)*i/(n*(2*i+2))); bm=lambda i: mp.sqrt((n-i)*(i+2)/(n*(2*i+2)))
    y=lambda j: mp.sqrt((n+1-j)*(n+2+j)*j*(j+1))/4
    def M1(i):  # W_i -> U_{i+1}: magnitudes A,B,C
        return (y(i+1)/(x*ap(i+1)), x*am(i+1)*bm(i)/(y(i+1)*bm(i+1)), y(i+1)/(x*ap(i+1)*bm(i+1)))
    def M2(i):  # U_i -> W_{i+1}
        return (y(i+1)/(x*am(i+1)), x*ap(i+1)*bp(i)/(y(i+1)*bp(i+1)), y(i+1)/(x*am(i+1)*bp(i+1)))
    return x,y,M1,M2
def Bn(n):
    x,y,M1,M2=coeffs(n)
    fro=lambda m: mp.sqrt(m[0]**2+m[1]**2+m[2]**2)
    lead=fro(M1(0))*fro(M2(1))
    det_rest=mp.mpf(1); grow=mp.mpf(1); fac=mp.mpf(1)
    for i in range(2,n-2,2):     # N(i)=M2(i+1)M1(i), i even, 2..n-3
        a1,b1,c1=M1(i); a2,b2,c2=M2(i+1)
        det_rest*=a1*b1*a2*b2
        grow*=c1*c2*(1-b2*a1/(c2*c1)-2*b1/c1)
        fac*=(1-2*x**2/y(i+1)**2-x**2/y(i+2)**2)
    yn=y(n)
    B=8*mp.sqrt(3)*(yn+mp.sqrt(2)*x**2/yn)*lead*det_rest/grow
    return B
if __name__=='__main__':
  for n in range(7,62,4):
    K,S=model(n,(1,1),ctx=mp)
    sv=mp.svd_r(K,compute_uv=False)
    q=8*min(sv)
    print(n,'q_min=%s'%mp.nstr(q,8),'bound=%s'%mp.nstr(Bn(n),8),'ratio=%s'%mp.nstr(Bn(n)/q,6),flush=True)

"""Referee check R5: theta_d = sum_{s=1}^{d-1} 1/(s sqrt((d-s) min(s,d-s))) <= theta_3 = 1/sqrt2 + 1/2.

Rigorous part: interval arithmetic (mpmath.iv, 40 digits) for 2 <= d <= 19, which is all that is
needed because the analytic tail bound zeta(3/2) sqrt(2/d) + (2/d)(1+ln(d/2)) is decreasing in d
and is evaluated here in interval arithmetic at d = 20 (zeta(3/2) enclosed by mpmath at 40 digits).
Also: re-derivation check of the tail bound against direct values for 20 <= d <= 2000, and the
float maximum over 2 <= d <= 10^5 (to compare with the author's E5).
"""
import sys, math
import mpmath as mp
from mpmath import iv
import numpy as np
iv.dps = 40; mp.mp.dps = 40
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

def theta_iv(d):
    t = iv.mpf(0)
    for s in range(1, d):
        t += 1/(iv.mpf(s)*iv.sqrt(iv.mpf((d-s)*min(s, d-s))))
    return t
th3 = 1/iv.sqrt(iv.mpf(2)) + iv.mpf(1)/2
ok = True; rows = []
for d in range(2, 20):
    th = theta_iv(d)
    strict = th.b < th3.a if d != 3 else True
    ok &= strict
    rows.append(f"d={d}: [{mp.nstr(th.a,12)}, {mp.nstr(th.b,12)}]")
log("R5 interval values: " + "; ".join(rows))
log(f"R5 theta_3 = [{mp.nstr(th3.a,15)}, {mp.nstr(th3.b,15)}]; theta_d < theta_3 strictly for all 2<=d<=19, d!=3: {ok}")
z = mp.zeta(1.5)
zi = iv.mpf([z - mp.mpf(10)**-35, z + mp.mpf(10)**-35])
tail20 = zi*iv.sqrt(iv.mpf(2)/20) + (iv.mpf(2)/20)*(1+iv.log(iv.mpf(10)))
log(f"R5 tail bound at d=20 = [{mp.nstr(tail20.a,12)}, {mp.nstr(tail20.b,12)}] < theta_3: {tail20.b < th3.a}")
viol = 0
for d in range(20, 2001):
    s = np.arange(1, d); t = d-s
    th = float(np.sum(1/(s*np.sqrt(t*np.minimum(s, t)))))
    tb = float(z)*math.sqrt(2/d) + (2/d)*(1+math.log(d/2))
    viol += th > tb
log(f"R5 direct theta_d <= tail bound for 20<=d<=2000: violations {viol}")
mx = max((float(np.sum(1/(np.arange(1,d)*np.sqrt((d-np.arange(1,d))*np.minimum(np.arange(1,d), d-np.arange(1,d)))))), d) for d in range(2, 100001))
log(f"R5 float max over 2<=d<=1e5: {mx[0]:.15f} at d={mx[1]}")
open(sys.argv[1] if len(sys.argv) > 1 else "r5_out.txt", "w").write("\n".join(out)+"\n")

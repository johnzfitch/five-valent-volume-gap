"""Referee check R2 (exact, CPU).  Independent of the author's code.

R2a  Own enumeration of sorted admissible doubled tuples 1<=J1<=J2<=J3<=J4<=N, admissibility
     decided directly from k_min <= k_max (not from a triangle shortcut).  Counts for N=60.
R2b  Theorem B finite step, by a DIFFERENT exact method from the author's LDL pivots:
     characteristic polynomial P_d(q) from the three-term recurrence with exact rational c_s^2,
     written as R(mu) with mu = q^2 (P_d = R(q^2) or q R(q^2)); sympy exact real-root counting
     of R on the closed interval [0, 3/16].  Covers every sorted tuple with 2 j_max <= NB
     (default 22; Theorem B needs 2 j_max <= 21).
R2c  Exact rational checks of Lemma 2 (log-concavity), Lemma 3 (ratios <= 6 and <= 4),
     Lemma 4 (a),(b) and Proposition C for 2 j_max <= NL (default 40), plus the margins:
     largest observed ratios and smallest observed slack.
"""
import sys, time
from fractions import Fraction as F
import sympy as sp

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 22
NL = int(sys.argv[2]) if len(sys.argv) > 2 else 40
OUT = sys.argv[3] if len(sys.argv) > 3 else "r2_out.txt"
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

def tuples(N):
    for J4 in range(1, N+1):
        for J3 in range(1, J4+1):
            for J2 in range(1, J3+1):
                for J1 in range(1, J2+1):
                    if (J1+J2+J3+J4) % 2: continue
                    kmin2 = max(abs(J1-J2), abs(J3-J4)); kmax2 = min(J1+J2, J3+J4)
                    if kmax2 >= kmin2:
                        yield (J1, J2, J3, J4)

def chain(T):
    j1, j2, j3, j4 = (F(x, 2) for x in T)
    kmin = max(abs(j1-j2), abs(j3-j4)); kmax = min(j1+j2, j3+j4)
    d = int(kmax-kmin)+1
    c2 = []
    for s in range(1, d):
        k = kmin+s
        c2.append(((j1+j2+1)**2-k*k)*(k*k-(j1-j2)**2)*((j3+j4+1)**2-k*k)*(k*k-(j3-j4)**2)/(16*(4*k*k-1)))
    return c2, d, kmin

# R2a counts
t0 = time.time()
cnt = cnt2 = cnt23 = 0
for T in tuples(60):
    c2, d, _ = chain(T)
    cnt += 1; cnt2 += d >= 2; cnt23 += d in (2, 3)
log(f"R2a sorted admissible tuples with 2j_max<=60: {cnt}; with d>=2: {cnt2}; with d in (2,3): {cnt23}  ({time.time()-t0:.1f}s)")

# R2b Theorem B certificate by root counting
q, mu = sp.symbols("q mu")
t0 = time.time()
nb = 0; bad = []; eq = []
mu0 = sp.Rational(3, 16)
for T in tuples(NB):
    c2, d, _ = chain(T)
    if d < 2: continue
    P0, P1 = sp.Poly(1, q, domain="QQ"), sp.Poly(q, q, domain="QQ")
    for v in c2:
        P0, P1 = P1, P1*sp.Poly(q, q, domain="QQ") - P0*sp.Rational(v.numerator, v.denominator)
    coeffs = P1.all_coeffs()[::-1]  # ascending in q
    if d % 2:  # P = q R(q^2)
        assert all(c == 0 for c in coeffs[0::2])
        rc = coeffs[1::2]
    else:
        assert all(c == 0 for c in coeffs[1::2])
        rc = coeffs[0::2]
    R = sp.Poly(list(reversed(rc)), mu, domain="QQ")
    assert R.eval(0) != 0
    nroots = R.count_roots(0, mu0)
    nb += 1
    if nroots:
        (eq if R.eval(mu0) == 0 and nroots == 1 else bad).append((T, nroots))
log(f"R2b root-count certificate, every sorted tuple with d>=2 and 2j_max<={NB}: {nb} tuples; "
    f"tuples with a root mu=q^2 in [0,3/16]: {bad + eq}; of which exactly one root at mu=3/16: {eq}  ({time.time()-t0:.1f}s)")

# R2c lemma checks
t0 = time.time()
nlc = nrat = nlow = 0; viol = []
maxup = maxdown = F(0); minlc = None
nc = 0; eqC = []; violC = []
for T in tuples(NL):
    c2, d, kmin = chain(T)
    jm = F(T[3], 2)
    for i in range(1, len(c2)-1):
        nlc += 1
        r = c2[i]*c2[i]/(c2[i-1]*c2[i+1])
        minlc = r if minlc is None or r < minlc else minlc
        if r < 1: viol.append(("LC", T, i))
    for i in range(len(c2)-1):
        nrat += 1
        u = c2[i+1]/c2[i]; dn = c2[i]/c2[i+1]
        maxup = max(maxup, u); maxdown = max(maxdown, dn)
        if u > 6 or dn > 4: viol.append(("ratio", T, i))
    for i, v in enumerate(c2):
        s = i+1; t = d-s; k = kmin+s
        nlow += 1
        if 16*v < s*s*t*t*k*k: viol.append(("4a", T, i))
        if 96*v < s*t*min(s, t)*k*(jm+1)**2: viol.append(("4b", T, i))
    if d == 2:
        nc += 1
        if c2[0] < jm*(jm+1)/4: violC.append(T)
        if c2[0] == jm*(jm+1)/4: eqC.append(T)
    elif d == 3:
        nc += 1
        if c2[0]+c2[1] <= jm*(jm+1)/4: violC.append(T)
log(f"R2c 2j_max<={NL}: log-concavity checks {nlc} (min (a_k^2)^2/(a_(k-1)^2 a_(k+1)^2) = {float(minlc):.6f}), "
    f"ratio checks {nrat} (max up a_(k+1)^2/a_k^2 = {float(maxup):.6f}, max down = {float(maxdown):.6f}), lower-bound checks {nlow}; violations {len(viol)} {viol[:5]}")
log(f"R2c Prop C (lambda^2 = c1^2 for d=2, c1^2+c2^2 for d=3) on {nc} tuples: violations {len(violC)}; equality cases {len(eqC)}, "
    f"all (1/2,1/2,j,j): {all(t[0]==1 and t[1]==1 and t[2]==t[3] for t in eqC)}  ({time.time()-t0:.1f}s)")
open(OUT, "w").write("\n".join(out)+"\n")

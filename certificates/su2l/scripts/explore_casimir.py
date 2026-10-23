import numpy as np, sys
from ladder import blocks, grams
S = int(sys.argv[1]); legs = tuple(int(x) for x in sys.argv[2].split(','))
sites, W, V, M = blocks(S, legs)
Gp, Gm, Cp, Cm = grams(S, V, M)
s = S/2
print("dims", [len(v) for v in V])
for k in range(S+1):
    D = Cp[k]-Cm[k]
    off = np.abs(D - np.diag(np.diag(D))).max()
    dg = np.diag(D)
    print(f"k={k} d={k-s:+.1f} |V|={len(V[k])} offdiag(Cp-Cm)={off:.2e} diag defect range [{dg.min():.3g},{dg.max():.3g}]  lmin Cp={np.linalg.eigvalsh(Cp[k]).min():.4g} lmin Cm={np.linalg.eigvalsh(Cm[k]).min():.4g}  s(s+1)/4={s*(s+1)/4}")
ev, U = np.linalg.eigh(W)
order = np.argsort(np.abs(ev))
nz = [i for i in order if abs(ev[i])>1e-9]
psi = U[:, nz[0]]
print("lambda", abs(ev[nz[0]]))
# Casimir expectation using symmetric C = (Cp+Cm)/2, block-diagonal
tot=0; tot2=0
for k in range(S+1):
    x = psi[V[k]]
    C = (Cp[k]+Cm[k])/2
    tot += x@C@x; tot2 += x@C@C@x
print("<C>", tot, "spread", np.sqrt(max(tot2-tot**2,0)), " s(s+1)/4", s*(s+1)/4, " (s+1)(s+2)/4", (s+1)*(s+2)/4)

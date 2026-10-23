"""lambda(W0) (best pairing/class) and q along families with fixed j1 and growing legs."""
import sys, math
from t1core import qmin, valid
from w0core import pairings, W0_block, lam, covered

fams = {
    'open (s; n-1/2, n, n, n)': lambda n2: (n2 - 1, n2, n2, n2),
    'eq (s; n,n,n,n)': lambda n2: (n2, n2, n2, n2),
    'gap (s; n, n+1, n+2, n+3)': lambda n2: (n2, n2 + 2, n2 + 4, n2 + 6),
}
for S in [5, 7, 9, 13, 21]:
    for name, f in fams.items():
        for n2 in [10, 20, 40, 80]:
            legs = f(n2)
            if not valid(S, *legs):
                legs = (legs[0] + 1,) + legs[1:]
            q, dk, n = qmin(S, *legs)
            best = 0
            for p1, p2 in pairings(*legs):
                for col in (0, 1):
                    B0, c_, r_ = W0_block(S, p1, p2, col)
                    l = lam(B0)
                    if l is not None:
                        best = max(best, l)
            s = S / 2
            print(f'2s={S:3d} {name:28s} legs={legs} dim={n:5d} type={covered(S,*legs):10s} q={q:9.3f} '
                  f'lamW0={best:.4f} lamW0*(s+1)={best*(s+1):.3f}', flush=True)

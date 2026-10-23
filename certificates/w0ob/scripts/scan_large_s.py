"""lambda(W0) (best of the three pairings) along bulk families at large 2j1, to see where the infimum goes.
Families (doubled legs): A = (2n-1, 2n, 2n, 2n), B = (2n, 2n+1, 2n+2, 2n+2), C = (2n, 2n, 2n, 2n+1).
Usage: python scan_large_s.py"""
import time
from w0sparse import lam_sparse, pairings

fams = {'A (n-1/2,n,n,n)': lambda N: (N - 1, N, N, N),
        'B (n,n+1/2,n+1,n+1)': lambda N: (N, N + 1, N + 2, N + 2),
        'C (n,n,n,n+1/2)': lambda N: (N, N, N, N + 1)}
for S in (41, 81, 161):
    for name, f in fams.items():
        for N in (128, 256, 512, 1024):
            legs = f(N)
            if (S + sum(legs)) % 2:
                continue
            t = time.time()
            best = 0; sz = 0
            for p in pairings(*legs):
                l, Nn = lam_sparse(S, p, dense_max=1500)
                sz = Nn
                if l is not None and l > best:
                    best = l
            print(f'2j1={S:4d} {name:22s} legs={legs} dim={sz:6d} best lambda={best:.5f} '
                  f'lambda*(j1+1)={best*(S/2+1):.2f}  ({time.time()-t:.1f}s)', flush=True)

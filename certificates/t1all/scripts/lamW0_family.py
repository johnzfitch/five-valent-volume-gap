"""Best-pairing lambda(W0) along (j1; n-1/2, n, n, n) and (j1; n,n,n,n)-type families to large n.
Usage: python lamW0_family.py S NMAXDOUBLED"""
import sys, numpy as np
from t1core import valid
from w0core import pairings, W0_block, lam

S = int(sys.argv[1]); NM = int(sys.argv[2])
n2 = 16
while n2 <= NM:
    for name, legs in (('(n-1/2,n,n,n)', (n2 - 1, n2, n2, n2)), ('(n,n,n,n)', (n2, n2, n2, n2))):
        if not valid(S, *legs):
            legs = (legs[0] + 1,) + legs[1:]
            if not valid(S, *legs):
                continue
        best = 0
        for p1, p2 in pairings(*legs):
            for col in (0, 1):
                B0, c_, r_ = W0_block(S, p1, p2, col)
                l = lam(B0)
                if l is not None:
                    best = max(best, l)
        print(f'2j1={S} {name:14s} legs(doubled)={legs} lambda(W0)={best:.5f} lambda*(j1+1)={best*(S/2+1):.4f}', flush=True)
    n2 *= 2

"""Exploration 2: families at larger spin.  q, lam(W0), lam(KG), lam(KF), Phi = min_E phi * min_O phi, bG.
float only (exploration)."""
import sys, os, math
import numpy as np
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_core import *
from oc_explore1 import weights, mm

fams = {
    'A (3;a,a+1,a+1,a+1)': lambda a: (3, a, a + 1, a + 1, a + 1),
    'B (3;a,a,a,a+1)?': lambda a: (3, a, a + 1, a + 2, a + 2) ,
    'C (5;a,a+1,a+1,a+1)': lambda a: (5, a, a + 1, a + 1, a + 1),
    'D (a/2 ; a,a+1,a+1,a+1)': lambda a: ((a // 2) | 1, a, a + 1, a + 1, a + 1),
    'E (3;a,a+1,2a,2a+2)': lambda a: (3, a, a + 1, 2 * a, 2 * a + 2),
    'F (3;2,3,a,a)': lambda a: (3, 2, 3, a, a+1 if (a % 2) else a),
}
for name, f in fams.items():
    print(name)
    for a in [4, 8, 16, 32, 48, 64]:
        tj = f(a)
        if sum(tj) % 2:
            tj = (tj[0], tj[1], tj[2], tj[3], tj[4] + 1)
        if not is_open(tj):
            print('  ', tj, 'not open'); continue
        S, P, O = weights(tj)
        lT = lam(tj, 'T'); l0 = lam(tj, 'K'); lG = lam(tj, 'KG'); lF = lam(tj, 'KF')
        Phi = mm(S, P, tj[0])
        print(f"   {tj} dim={len(S)} q={8*lT:.4f} lamW0={l0:.4f} lamKG={lG:.4f} lamKF={lF:.4f} Phi={Phi:.3f} bG={2*math.sqrt(Phi)*lG:.4f}")

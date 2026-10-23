"""Exploration: sign and size of Lambda = diag(N^T N - (4/3) D0^T D0) (full row sets) per regime,
orientation n1<n2.  Usage: python z32_lambda.py M
"""
import sys, os, collections
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z32_core import Z32, pairings6, tuples32
from z32_cover import label

if __name__ == '__main__':
    M = int(sys.argv[1])
    kap = Fr(4, 3)
    st = {}
    for tj in tuples32(M):
        for P in pairings6(tj):
            z = Z32(P)
            if z.P[0] > z.P[2] or not z.C0:
                continue
            lab = label(z)
            for b in z.C0:
                nn = sum(w for (r, c), w in z.N.items() if c == b)
                dd = sum(w for (r, c), w in z.D0.items() if c == b)
                lam = nn - kap * dd
                if lam > 0:
                    rel = float(lam / dd)
                    pos = float((b - z.C0[0]) / max(1, z.C0[-1] - z.C0[0]))
                    cur = st.get(lab)
                    if cur is None or rel > cur[0]:
                        st[lab] = (rel, float(lam), pos, P, str(b))
    for k, v in sorted(st.items()):
        print(k, f'max Lambda+/diag {v[0]:.4f} (Lambda {v[1]:.1f}, rel pos {v[2]:.2f}) at {v[3]} b={v[4]}')

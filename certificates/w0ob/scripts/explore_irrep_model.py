"""Idealised su(2) wall model: A(S,s) = compression of J_x (spin S) to |M| <= s.
g(S,s) = smallest nonzero singular value.  Usage: python explore_irrep_model.py"""
import numpy as np


def A(S2, s2):
    """S2 = 2S, s2 = 2s (same parity). Rows/cols M = -s..s."""
    S = S2 / 2; s = s2 / 2
    Ms = np.arange(-s, s + 0.5, 1.0)
    n = len(Ms)
    a = np.zeros((n, n))
    for i in range(n - 1):
        M = Ms[i]
        l = 0.5 * np.sqrt(max(0.0, (S - M) * (S + M + 1)))
        a[i, i + 1] = a[i + 1, i] = l
    return a


def g(S2, s2):
    ev = np.abs(np.linalg.eigvalsh(A(S2, s2)))
    nz = ev[ev > 1e-10]
    return nz.min()


for s2 in [3, 5, 7, 9, 13, 21, 41, 81, 161, 321]:
    vals = [(g(S2, s2), S2) for S2 in range(s2, s2 + 400, 2)]
    m = min(vals)
    print(f'2s={s2:3d}: g(s,s)={g(s2,s2):.4f}  min_S g={m[0]:.5f} at 2S={m[1]}  g(s+1)={g(s2+2,s2):.4f} g(s+2)={g(s2+4,s2):.4f} g(s+5)={g(s2+10,s2):.4f}  min*(s+1)={m[0]*(s2/2+1):.3f}')

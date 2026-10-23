"""Is the Casimir Q = 2(E1E2 + F1F2) - (b+1/2)(c+1/2), compressed to R, approximately conserved by the
lowest nonzero mode of W0?  For the low mode psi (unit) report
   mean <Q>, std of Q, and ||[Q_R, W0] psi||, together with the spacing of the idealised Casimir
   S(S+1)/2 near S = s (which is s + 1 between S = s and s + 1).
Two cases: (a) an integer-s tuple with both pairs m = 1/2 (the exact su(2) x su(2) case, where
[Q,K] = 0 except at the bottom labels b = 1/2, c = 1/2, and the corner cut u >= s breaks it);
(b) the extremal half-integer family (n-1/2, n | n, n) (m = 1/2 and m = 0).
Usage: python test_casimir_sector.py"""
import numpy as np
import scipy.sparse as sp
from w0sparse import build, h2


def Qmat(S, legs, sites):
    a2, a3, a4, a5 = legs
    n1 = a2 + a3 + 2; m1 = abs(a2 - a3); n2 = a4 + a5 + 2; m2 = abs(a4 - a5)
    idx = {x: i for i, x in enumerate(sites)}
    I = []; J = []; V = []
    for (b, c), i in idx.items():
        I.append(i); J.append(i); V.append(-(b / 2 + .5) * (c / 2 + .5))
        k = idx.get((b + 2, c + 2))
        if k is not None:
            w = 2 * np.sqrt(h2(n1, m1, b + 2) * h2(n2, m2, c + 2))
            I += [i, k]; J += [k, i]; V += [w, w]
    return sp.csr_matrix((V, (I, J)), shape=(len(sites), len(sites)))


for S, legs in [(4, (39, 40, 41, 42)), (6, (79, 80, 79, 80)), (5, (63, 64, 64, 64)), (5, (255, 256, 256, 256)), (21, (127, 128, 128, 128))]:
    sites, W = build(S, *legs)
    Q = Qmat(S, legs, sites)
    Wd = W.toarray() if W.shape[0] < 6000 else None
    ev, vec = np.linalg.eigh(Wd)
    a = np.abs(ev); o = [i for i in np.argsort(a) if a[i] > 1e-7]
    psi = vec[:, o[0]]
    Qp = Q @ psi
    mean = psi @ Qp; std = np.sqrt(max(0, Qp @ Qp - mean ** 2))
    C = (Q @ W - W @ Q)
    cpsi = np.linalg.norm(C @ psi)
    s = S / 2
    print(f'2s={S} legs={legs} lambda={a[o[0]]:.4f}: <Q>={mean:.2f} std(Q)={std:.2f} '
          f'||[Q,W0]psi||={cpsi:.2f}  ||[Q,W0]||_max-entry={abs(C).max():.1f}  idealised spacing near S=s: {s+1:.1f}', flush=True)

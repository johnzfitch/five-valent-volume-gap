"""How far real W0 is from the exact-ladder class of Theorem A (W0 frame).
For each tuple (pairing (a2 a3 | a4 a5), doubled spins):
  lambda(W0), the lowest nonzero |eig| mode psi;
  C = (C+ + C-)/2 on interior diagonals (C+ on the bottom wall, C- on the top wall);
  <C> and spread of C on psi, in units of the sector spacing (s+1)/2 at S = s;
  interior defect Delta = C+ - C- (diagonal): max |Delta| over sites with u <= 2s+2 ('band end') and u > 2s+2;
  relative operator defect max_d ||(G-_d)^{+1/2} Delta_d (G-_d)^{+1/2}|| on range(G-_d)."""
import numpy as np
from ladder import blocks, grams

TUPLES = [(5, (39, 40, 40, 40)), (5, (127, 128, 128, 128)), (9, (63, 64, 64, 64)), (21, (63, 64, 64, 64)),
          (5, (20, 24, 30, 31)), (9, (17, 17, 33, 34)), (7, (40, 41, 42, 42)), (6, (60, 62, 62, 64))]

print("2s legs | lambda | <C> - s(s+1)/4 and spread, in sector units | max|Delta| band end / bulk | rel. op. defect")
for S, legs in TUPLES:
    sites, W, V, M = blocks(S, legs)
    Gp, Gm, Cp, Cm = grams(S, V, M)
    s = S / 2
    ev, U = np.linalg.eigh(W)
    order = [i for i in np.argsort(np.abs(ev)) if abs(ev[i]) > 1e-9]
    lam = abs(ev[order[0]]); psi = U[:, order[0]]
    m1 = m2 = 0.0
    for k in range(S + 1):
        C = Cp[k] if k == 0 else (Cm[k] if k == S else (Cp[k] + Cm[k]) / 2)
        x = psi[V[k]]
        m1 += x @ C @ x; m2 += x @ C @ C @ x
    unit = (s + 1) / 2
    spread = np.sqrt(max(m2 - m1 ** 2, 0))
    end = bulk = 0.0; rel = 0.0
    for k in range(1, S):
        D = np.diag(Cp[k] - Cm[k])
        for t, i in enumerate(V[k]):
            b, c = sites[i]; u = (b + c) / 2
            if u <= 2 * s + 2: end = max(end, abs(D[t]))
            else: bulk = max(bulk, abs(D[t]))
        w, Q = np.linalg.eigh(Gm[k])
        keep = w > 1e-9 * max(1, w.max())
        if keep.any():
            R = Q[:, keep] / np.sqrt(w[keep])
            rel = max(rel, np.abs(np.linalg.eigvalsh(R.T @ np.diag(D) @ R)).max())
    print(f"{S:2d} {legs} | {lam:.4f} | {(m1 - s*(s+1)/4)/unit:7.3f} +- {spread/unit:6.3f} | {end:9.3g} / {bulk:8.3g} | {rel:8.3g}")

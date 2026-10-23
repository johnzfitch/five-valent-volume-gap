"""Exact-rational check of the congruence lemma behind Theorem B.
For 1 <= 2s <= 61 and rational kappa' > kappa > s(s-1)/4 (incl. kappa = s(s+1)/4):
  rho_i^2 = (kappa' - c_i)/(kappa - c_i), c_i = d_i(d_i+1)/4, d_i = -s+i  (i = 0..2s-1)
  sigma_0^2 = 1, sigma_{j+1}^2 = rho_j^2 / sigma_j^2
Claims checked exactly:  (a) min_{even j} sigma_j^2 * min_{odd j} sigma_j^2 = rho_mid^2 >= 1, with rho_mid the
link with |d+1/2| minimal (the middle link; for integer s either of the two middle links, equal by symmetry);
(b) both colour subsequences are unimodal with the minima at the positions stated in the proof;
(c) at kappa0 = s(s+1)/4 the squared links equal (s-d)(s+d+1)/4 (J_x(s)).
Float part: every singular value of the bipartite block is nondecreasing in kappa (Ostrowski consequence)."""
from fractions import Fraction as Fr
import random, numpy as np
random.seed(7)
bad = 0; cnt = 0
for S2 in range(1, 62):
    s = Fr(S2, 2)
    dlist = [-s + i for i in range(S2)]
    c = [d * (d + 1) / 4 for d in dlist]
    k0 = s * (s + 1) / 4
    assert all(k0 - ci == (s - d) * (s + d + 1) / 4 for ci, d in zip(c, dlist))
    kmin = s * (s - 1) / 4
    pairs = [(k0, k0 + Fr(random.randint(1, 10**6), random.randint(1, 1000))) for _ in range(40)]
    for _ in range(40):
        a = kmin + Fr(random.randint(1, 10**5), random.randint(1, 10**4))
        b = a + Fr(random.randint(1, 10**5), random.randint(1, 10**4))
        pairs.append((a, b))
    for k, k2 in pairs:
        rho2 = [(k2 - ci) / (k - ci) for ci in c]
        sig2 = [Fr(1)]
        for r in rho2: sig2.append(r / sig2[-1])
        ev = sig2[0::2]; od = sig2[1::2]
        # middle link: minimal |d+1/2|
        imid = min(range(S2), key=lambda i: abs(dlist[i] + Fr(1, 2)))
        lhs = min(ev) * min(od)
        cnt += 1
        if lhs != rho2[imid] or rho2[imid] < 1:
            bad += 1
        # unimodality (nonincreasing then nondecreasing)
        for seq in (ev, od):
            i = seq.index(min(seq))
            if any(seq[t+1] > seq[t] for t in range(i)) or any(seq[t+1] < seq[t] for t in range(i, len(seq)-1)):
                bad += 1
print(f"exact congruence checks: {cnt} (kappa, kappa') pairs over 2s = 1..61, failures: {bad}")
# float: all singular values monotone
def Bsv(S2, k):
    s = S2/2; d = -s + np.arange(S2); l = np.sqrt(k - d*(d+1)/4)
    J = np.diag(l, 1) + np.diag(l, -1)
    rows = list(range(0, S2+1, 2)); cols = list(range(1, S2+1, 2))
    return np.linalg.svd(J[np.ix_(rows, cols)], compute_uv=False)
viol = 0
for S2 in range(1, 62):
    s = S2/2; k0 = s*(s+1)/4
    ks = k0 + np.concatenate([[0], np.geomspace(1e-4, 1e5, 120)])
    prev = None
    for k in ks:
        sv = np.sort(Bsv(S2, k))
        if prev is not None and np.any(sv < prev - 1e-10*np.maximum(1, prev)): viol += 1
        prev = sv
    sv0 = np.sort(Bsv(S2, k0))
    expect = np.arange(1 if S2 % 2 == 0 else 0.5, s + 0.1, 1.0)
    assert np.allclose(sv0, expect), (S2, sv0[:3])
print(f"float: singular values of B(kappa) nondecreasing on a 121-point kappa grid, 2s = 1..61: violations {viol}; "
      f"B(kappa0) singular values = {{1/2,...,s}} (half-integer) / {{1,...,s}} (integer): ok")

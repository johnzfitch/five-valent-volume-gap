"""Numerical sanity check (not part of the proof): the closed-form tail bounds of tail_bounds.py dominate the box
quantities F0, F1, W (computed by ibox.py's outward-rounded routines, on the CPU) at selected n1, for every a.
Usage: python tail_validate.py n [n ...]"""
import sys, torch
from ibox import *
import tail_bounds as tb

def box_at(name, n0):
    n_all, a_all = pairs(name, n0, 'cpu')
    sel = (n_all == n0); n, a = n_all[sel], a_all[sel]
    one = torch.ones_like(n); z = 0 * one; half = H * one
    if name == 'R1':
        m2w = a - K; mu = torch.where(((a + 0.5) - torch.round(a + 0.5)).abs() < 1e-9, z, half)
        F0c, F0t, Wc, Wt = CR_block(a, n - 1, lambda c: q0_dn(n + K, m2w, c), lambda c: p0_up(n, a, c), lambda c: Lam_up(c, n, a, n + K, mu))
        F1c, F1t, _, _ = CR_block(a, n - 1, lambda c: q1_dn(n + K, m2w, c), lambda c: p1_up(n, a, c), None)
        return a, dict(F0c=F0c, F0t=F0t, F1c=F1c, F1t=F1t, Wt=Wt, Wc=Wc)
    if name in ('R2+', 'R2-'):
        m2 = a + H if name == 'R2+' else a - H
        F0c, F0t, Wc, Wt = CR_block(a, n - 1, lambda c: q0_dn(n + K, m2, c), lambda c: p0_up(n, a, c), lambda c: Lam_up(c, n, a, n + K, m2))
        lo1 = a + 1 if name == 'R2+' else a
        F1 = RR_block(lo1, n - 1, lambda i: p1_dn(n, a, i), lambda i: q1_dn(n + K, m2, i))
        return a, dict(F0c=F0c, F0t=F0t, F1=F1, Wt=Wt, Wc=Wc)
    if name == 'R4':
        m2 = a; b0 = m2 - K
        F0 = RR_block(b0, n - 1, lambda i: p0_dn(n, b0, i), lambda i: q0_dn(n + K, m2, i))
        F1 = RR_block(m2 + H, n - 1, lambda i: p1_dn(n, b0, i), lambda i: q1_dn(n + K, m2, i))
        return a, dict(F0=F0, F1=F1)
    if name == 'R4c':
        F0 = RR_block(z, n - 1, lambda i: p0_dn(n, z, i), lambda i: q0_dn(n + K, half, i))
        F1 = RR_block(one, n - 1, lambda i: p1_dn(n, z, i), lambda i: q1_dn(n + K, half, i))
        return a, dict(F0=F0, F1=F1)
    if name == 'R1c':
        F0c, F0t, Wc, Wt = CR_block(half, n - 1, lambda c: q0_dn(n + K, z, c), lambda c: p0_up(n, half, c), lambda c: Lam_up(c, n, half, n + K, z))
        F1c, F1t, _, _ = CR_block(K * one, n - 1, lambda c: q1_dn(n + K, z, c), lambda c: p1_up(n, half, c), None)
        return a, dict(F0t=F0t, F1t=F1t, Wt=Wt, F0c=F0c, F1c=F1c, Wc=Wc)
    if name in ('Tp', 'Tm'):
        n2 = n + H if name == 'Tp' else n - H; hi0 = n - 2 if name == 'Tp' else n - 3
        F0c, F0t, Wc, Wt = RC_block(n, hi0, n2, lambda c: Lam_up(c, n, z, n2, half))
        F1 = RR_block(one, n - 1, lambda i: p1_dn(n, z, i), lambda i: q1_dn(n2, half, i))
        return a, dict(F0t=F0t, Wt=Wt, F1=F1, F0c=F0c, Wc=Wc)

# which box quantity each closed form is claimed to dominate
CLAIM = {'R2-': ('F0c', 'F1', 'Wt'), 'R2+': ('F0c', 'F1', 'Wt'), 'R1': ('F0c', 'F1c', 'Wt'),
         'R4': ('F0', 'F1', None), 'R4c': ('F0', 'F1', None), 'R1c': ('F0t', 'F1t', 'Wt'),
         'Tm': ('F0t', 'F1', 'Wt'), 'Tp': ('F0t', 'F1', 'Wt')}

if __name__ == '__main__':
    out = []
    for nn in sys.argv[1:]:
        n0 = float(nn)
        for name, (k0, k1, kw) in CLAIM.items():
            if name in ('R1c', 'Tm', 'Tp', 'R4c') and (n0 - int(n0)) != {'R1c': 0.5, 'Tm': 0.0, 'Tp': 0.0, 'R4c': 0.0}[name]:
                continue
            a, q = box_at(name, n0)
            worst = [0.0, 0.0, 0.0]
            for j in range(len(a)):
                F0b, F1b, Wb = tb.BOUNDS[name](n0, float(a[j]))
                worst[0] = max(worst[0], float(q[k0][j]) / F0b)
                worst[1] = max(worst[1], float(q[k1][j]) / F1b)
                if kw: worst[2] = max(worst[2], float(q[kw][j]) / Wb)
            line = f"n1={n0:8.1f} {name:4s}: max(box/bound)  F0 {worst[0]:.3e}  F1 {worst[1]:.3e}  W {worst[2]:.3e}   ({len(a)} values of a)"
            print(line, flush=True); out.append(line)
    open('../outputs/tail_validate.txt', 'a').write("\n".join(out) + "\n")

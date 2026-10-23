"""Exploration (not part of the proof): F0, F1, W (count and exact-product forms) at a fixed n1 for all a, using the
interval box code on the CPU, to see the slack available to the tail bounds."""
import sys, torch
import ibox
from ibox import *
def at_n(name, n0, dev='cpu'):
    n_all, a_all = pairs(name, n0, dev)
    sel = (n_all == n0)
    n, a = n_all[sel], a_all[sel]
    one = torch.ones_like(n); z = 0 * one; half = H * one
    out = {}
    if name == 'R1':
        m2w = a - K; mu = torch.where(((a + 0.5) - torch.round(a + 0.5)).abs() < 1e-9, z, half)
        F0c, F0t, Wc, Wt = CR_block(a, n - 1, lambda c: q0_dn(n + K, m2w, c), lambda c: p0_up(n, a, c), lambda c: Lam_up(c, n, a, n + K, mu))
        F1c, F1t, _, _ = CR_block(a, n - 1, lambda c: q1_dn(n + K, m2w, c), lambda c: p1_up(n, a, c), None)
        out = dict(F0c=F0c, F0t=F0t, Wc=Wc, Wt=Wt, F1c=F1c, F1t=F1t)
    elif name in ('R2+', 'R2-'):
        m2 = a + H if name == 'R2+' else a - H
        F0c, F0t, Wc, Wt = CR_block(a, n - 1, lambda c: q0_dn(n + K, m2, c), lambda c: p0_up(n, a, c), lambda c: Lam_up(c, n, a, n + K, m2))
        lo1 = a + 1 if name == 'R2+' else a
        F1 = RR_block(lo1, n - 1, lambda i: p1_dn(n, a, i), lambda i: q1_dn(n + K, m2, i))
        out = dict(F0c=F0c, F0t=F0t, Wc=Wc, Wt=Wt, F1=F1)
    return n, a, out
if __name__ == '__main__':
    name, n0 = sys.argv[1], float(sys.argv[2])
    n, a, out = at_n(name, n0)
    for key, v in out.items():
        vv = v * n
        i = int(torch.argmax(vv)); print(f"{name} n={n0}: max_a n*{key} = {float(vv[i]):.4f} at a={float(a[i])};  at a=min: {float(vv[0]):.4f}")

def G_profile(name, n0):
    n, a, o = at_n(name, n0)
    F0 = torch.minimum(o['F0c'], o['F0t'])
    W = torch.minimum(o['Wc'], o['Wt'])
    F1 = torch.minimum(o['F1c'], o['F1t']) if 'F1c' in o else o['F1']
    G = (4/3 + 0.95 + W) * F1
    i = int(torch.argmax(G)); j = int(torch.argmax(F0))
    print(f"{name} n={n0}: max_a G = {float(G[i]):.4f} at a={float(a[i])} (W={float(W[i]):.3f}, F1={float(F1[i]):.3e});  max F0 = {float(F0[j]):.3e} at a={float(a[j])}")

def G_count(name, n0):
    n, a, o = at_n(name, n0)
    F1 = o['F1c'] if 'F1c' in o else o['F1']
    G = (4/3 + 0.95 + o['Wc']) * F1
    i = int(torch.argmax(G))
    print(f"COUNT {name} n={n0}: max_a G = {float(G[i]):.4f} at a={float(a[i])} (Wc={float(o['Wc'][i]):.3f}, F1={float(F1[i]):.3e}); max F0c={float(o['F0c'].max()):.3e}")

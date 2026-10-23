"""Sanity check (not part of the proof): the closed-form tail functions of z32_certify_tail must dominate the
box quantities wherever both are defined.  For sampled n in [NLO, NHI] and every parameter a at that n:
   G_box(n,a) = (4/3 + theta + W_box) F1_box  <=  Gbar(n)      and   F0_box <= F0bar(n),
and for R2+ with a <= 60 also against the per-a function.
Usage: python z32_check_tail_vs_box.py NLO NHI device out.txt
"""
import sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z32_box_torch as zb
import z32_certify_tail as ct

if __name__ == '__main__':
    NLO, NHI, dev, out = float(sys.argv[1]), float(sys.argv[2]), torch.device(sys.argv[3]), sys.argv[4]
    lines = []
    fun = {'R1': ct.tail_R1, 'R2-': ct.tail_R2m, 'R4': ct.tail_R4, 'R1c': ct.tail_R1c, 'R4c': ct.tail_R4c,
           'Tp': lambda n: ct.tail_T(n, 'Tp'), 'Tm': lambda n: ct.tail_T(n, 'Tm')}
    series = {}
    for name in ('R1', 'R2-', 'R2+', 'R4', 'R1c', 'R4c', 'Tp', 'Tm'):
        n, a, F0, F1, W, s2 = zb.regime_box(name, NHI, dev)
        sel = n >= NLO
        n, a, F0, F1, W = n[sel].cpu(), a[sel].cpu(), F0[sel].cpu(), F1[sel].cpu(), W[sel].cpu()
        G = (ct.KAP + ct.THETA + W) * F1
        bad = 0; worst = 0.0
        for i in range(n.numel()):
            nn, aa = float(n[i]), float(a[i])
            if name == 'R2+':
                if aa <= 60:
                    if aa not in series:
                        series[aa] = ct.series_R2p(aa, Tmax=20000)
                    Gt, F0t = ct.tail_R2p_a(nn, aa, *series[aa])
                else:
                    Gt, F0t = ct.tail_R2p_large(nn, 60.0)
            else:
                Gt, F0t = fun[name](nn)
            r = max(float(G[i]) / Gt, float(F0[i]) / F0t)
            worst = max(worst, r)
            if r > 1 + 1e-9:
                bad += 1
        msg = f'{name:4s}: {n.numel()} parameter sets with {NLO} <= n <= {NHI}; violations {bad}; max box/tail ratio {worst:.4f}'
        print(msg); lines.append(msg)
    open(out, 'w').write('\n'.join(lines) + '\n')

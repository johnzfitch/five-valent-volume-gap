"""One-wall models: W0 compressed to {d <= s, u >= s} (only the wall d = s and the corner cut) and to
{d >= -s, u >= s} (only the wall d = -s and the corner), against the two-wall strip.  Prints the colours of
the cuts (corner colour 0; wall d = s colour 2(j4+j5) mod 2; wall d = -s colour 2(j2+j3) mod 2) and,
for the strip without the corner cut ({|d| <= s}, all u), the same numbers.  Usage: python test_onewall.py"""
import numpy as np
from test_localisation import region_W, lowest

for S, legs in [(5, (63, 64, 64, 64)), (5, (64, 63, 64, 64)), (9, (63, 64, 64, 64)), (21, (63, 64, 64, 64)), (7, (40, 41, 42, 42)), (6, (40, 42, 42, 42))]:
    a2, a3, a4, a5 = legs
    kp = (a4 + a5) % 2; km = (a2 + a3) % 2
    out = []
    for name, keep in [('strip', lambda b, c: abs(b - c) <= S <= b + c),
                       ('wall +s & corner', lambda b, c: (b - c) <= S <= b + c),
                       ('wall -s & corner', lambda b, c: (c - b) <= S <= b + c),
                       ('two walls, no corner cut', lambda b, c: abs(b - c) <= S)]:
        _, W = region_W(S, legs, keep)
        lam, _ = lowest(W, k=12)
        out.append(f'{name}: {lam:.4f}')
    print(f'2s={S} pairing {legs[:2]}|{legs[2:]} cut colours: corner 0, wall+s {kp}, wall-s {km}  ->  ' + '; '.join(out), flush=True)

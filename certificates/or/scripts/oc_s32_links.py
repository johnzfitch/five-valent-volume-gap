"""j1 = 3/2: exact check of the explicit link formulas (squares, in rationals) and float check of the
plaquette identity for the coupling block.

P1(L) = (n1^2-L^2)(L^2-m1^2), P2(L) = (n2^2-L^2)(L^2-m2^2).
zigzag 0: rows R_b=(b,b+3/2), cols C_b=(b,b+1/2);  zigzag 1: rows R'_b=(b,b-1/2), cols C'_b=(b,b-3/2).
  p_b  : R_b  - C_{b+1}  p_b^2  = 3 P1(b+1)(2b+5)/(16(2b+3))
  q_b  : R_b  - C_b      q_b^2  = 3 P2(b+3/2) b/(16(b+1))
  p'_b : R'_b - C'_{b+1} p'_b^2 = 3 P1(b+1)(2b-1)/(16(2b+1))
  q'_b : R'_b - C'_b     q'_b^2 = 3 P2(b-1/2)(b+1)/(16 b)
  n+_b : R'_b - C_b      n+_b^2 = P2(b+1/2)(2b+3)(2b-1)/(16 b(b+1))
  n-_b : R'_b - C_{b-1}  n-_b^2 = P1(b)(b^2-1)/(4b^2-1)
Plaquette identity (orientation A, D0 lower: R_b paired with C_{b+1}):
  (N D0^-1)[R'_b, R_{b-1}] = n+_b/p_{b-1} + (delta_b/p_{b-1}) (D0^-1)[C_{b-1}, R_{b-1}],
  (N D0^-1)[R'_b, R_j]      = (delta_b/p_{b-1}) (D0^-1)[C_{b-1}, R_j]  (j != b-1),  delta_b = n-_b p_{b-1} - n+_b q_{b-1};
orientation B (D0 upper: R_b paired with C_b):  replace n+/p by n-/q and delta/p by -delta/q, C_{b-1} by C_b.
Usage: python oc_s32_links.py M out.txt
"""
import sys, os, itertools
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_core import links, sites


def P(n, m, L):
    return (n * n - L * L) * (L * L - m * m)


def formulas(n1, m1, n2, m2):
    h = Fr(1, 2)
    return {
        'p': lambda b: 3 * P(n1, m1, b + 1) * (2 * b + 5) / (16 * (2 * b + 3)),
        'q': lambda b: 3 * P(n2, m2, b + 3 * h) * b / (16 * (b + 1)),
        'pp': lambda b: 3 * P(n1, m1, b + 1) * (2 * b - 1) / (16 * (2 * b + 1)),
        'qp': lambda b: 3 * P(n2, m2, b - h) * (b + 1) / (16 * b),
        'np': lambda b: P(n2, m2, b + h) * (2 * b + 3) * (2 * b - 1) / (16 * b * (b + 1)),
        'nm': lambda b: P(n1, m1, b) * (b * b - 1) / (4 * b * b - 1),
    }


if __name__ == '__main__':
    M, out = int(sys.argv[1]), sys.argv[2]
    h = Fr(1, 2)
    nlinks = 0; bad = 0; ntup = 0
    for tj in itertools.product(range(1, M + 1), repeat=4):
        P5 = (3,) + tj
        if sum(P5) % 2:
            continue
        S, Lk = links(P5, 'T')
        if len(S) < 2:
            continue
        ntup += 1
        j = [Fr(t, 2) for t in tj]
        n1, m1, n2, m2 = j[0] + j[1] + 1, abs(j[0] - j[1]), j[2] + j[3] + 1, abs(j[2] - j[3])
        f = formulas(n1, m1, n2, m2)
        idx = {(Fr(B, 2), Fr(C, 2)): i for i, (B, C) in enumerate(S)}
        W = {}
        for (i, k), w in Lk.items():
            W[frozenset((i, k))] = w
        for (b, c), i in idx.items():
            d = b - c
            checks = []
            if d == -Fr(3, 2):
                checks += [('p', (b + 1, c), b), ('q', (b, c - 1), b)]
            if d == h:
                checks += [('pp', (b + 1, c), b), ('qp', (b, c - 1), b), ('np', (b, c + 1), b), ('nm', (b - 1, c), b)]
            for name, other, bb in checks:
                kk = idx.get(other)
                if kk is None:
                    continue
                nlinks += 1
                if W[frozenset((i, kk))] != f[name](bb):
                    bad += 1
    L = [f"explicit j1=3/2 link formulas vs closed-form grid (exact rationals), all labelled tuples 2j_i<={M}: "
         f"{ntup} tuples, {nlinks} links, {bad} mismatches"]
    # plaquette identity, float, on the same tuples (pairing as given), both orientations where D0 square
    nid = 0; worst = 0.0
    for tj in itertools.product(range(1, M + 1), repeat=4):
        P5 = (3,) + tj
        if sum(P5) % 2:
            continue
        S, Lk = links(P5, 'T')
        if len(S) < 2:
            continue
        j = [Fr(t, 2) for t in tj]
        n1, m1, n2, m2 = j[0] + j[1] + 1, abs(j[0] - j[1]), j[2] + j[3] + 1, abs(j[2] - j[3])
        f = {k: (lambda g: (lambda b: float(g(b)) ** 0.5))(v) for k, v in formulas(n1, m1, n2, m2).items()}
        site = set((Fr(B, 2), Fr(C, 2)) for B, C in S)
        Rb = sorted(b for b, c in site if b - c == -Fr(3, 2))
        Cb = sorted(b for b, c in site if b - c == -h)
        Rp = sorted(b for b, c in site if b - c == h)
        if not Rb or not Cb or not Rp:
            continue
        for orient in ('A', 'B'):
            # choose square sub-zigzag: A pairs R_b with C_{b+1}; B pairs R_b with C_b
            rows = [b for b in Rb if ((b + 1) in Cb if orient == 'A' else b in Cb)]
            cols = [b + 1 if orient == 'A' else b for b in rows]
            if len(rows) < 2:
                continue
            ri = {b: i for i, b in enumerate(rows)}; ci = {b: i for i, b in enumerate(cols)}
            D = np.zeros((len(rows), len(cols)))
            for b in rows:
                if (b + 1) in ci: D[ri[b], ci[b + 1]] = f['p'](b)
                if b in ci: D[ri[b], ci[b]] = f['q'](b)
            if abs(np.linalg.det(D)) < 1e-300:
                continue
            Di = np.linalg.inv(D)
            for b in Rp:
                Nrow = np.zeros(len(cols))
                if b in ci: Nrow[ci[b]] += f['np'](b)
                if (b - 1) in ci: Nrow[ci[b - 1]] += f['nm'](b)
                lhs = Nrow @ Di
                pred = np.zeros(len(rows))
                if orient == 'A' and (b - 1) in ri and (b - 1) in ci and b in ci:
                    delta = f['nm'](b) * f['p'](b - 1) - f['np'](b) * f['q'](b - 1)
                    pred[ri[b - 1]] += f['np'](b) / f['p'](b - 1)
                    pred += (delta / f['p'](b - 1)) * Di[ci[b - 1], :]
                elif orient == 'B' and (b - 1) in ri and (b - 1) in ci and b in ci:
                    delta = f['nm'](b) * f['p'](b - 1) - f['np'](b) * f['q'](b - 1)
                    pred[ri[b - 1]] += f['nm'](b) / f['q'](b - 1)
                    pred += (-delta / f['q'](b - 1)) * Di[ci[b], :]
                else:
                    continue
                nid += 1
                worst = max(worst, np.abs(lhs - pred).max() / max(np.abs(lhs).max(), 1e-300))
    L.append(f"plaquette identity for N D0^-1 (orientations A and B, interior rows of zigzag 1): {nid} rows checked, "
             f"max relative deviation {worst:.2e}")
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

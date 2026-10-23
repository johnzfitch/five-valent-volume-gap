"""Exact check of Lemma M (j1 = 3/2): a pairing P is non-monotone (zigzag pattern (wide, tall)) iff
   |n1 - n2| = 1/2,  |m1 - m2| = 1/2,  and the integer one of m1, m2 is >= 1.
and of the consequence: every tuple with j1 = 3/2 has a monotone pairing.
Also checks the zigzag start/end rules used in the proof (start col <=> c_lo(d_k+1) < c_lo(d_k);
end row <=> d_k >= n1 - n2) on every zigzag of every tuple, for general half-integer s.
Exact (integer/Fraction) site enumeration.  Usage: python oc_lemma_s32.py M smax out.txt
"""
import sys, os, itertools
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_core import sites, pairings


def zigzags(P):
    """Return list over k of (rows c-list, cols c-list) for pairing P (doubled spins), exact."""
    s2 = P[0]; s = Fr(s2, 2)
    S = sites(P)
    K = s2 // 2 + 1
    Z = [([], []) for _ in range(K)]
    for B, C in S:
        d = Fr(B - C, 2); c = Fr(C, 2)
        t = d + s                      # integer 0..2s
        if t % 2 == 0:
            Z[int(t) // 2][0].append(c)
        else:
            Z[int(t - 1) // 2][1].append(c)
    return [(sorted(r), sorted(c)) for r, c in Z]


def imb_pattern(P):
    return [len(r) - len(c) for r, c in zigzags(P)]


def monotone(imb):
    w = False
    for x in imb:
        if x == -1:
            w = True
        if x == 1 and w:
            return False
    return True


def lemma_bad(P):
    s2, a, b, c, d = P
    n1, m1 = Fr(a + b, 2) + 1, Fr(abs(a - b), 2)
    n2, m2 = Fr(c + d, 2) + 1, Fr(abs(c - d), 2)
    mint = m1 if m1.denominator == 1 else m2
    return abs(n1 - n2) == Fr(1, 2) and abs(m1 - m2) == Fr(1, 2) and mint >= 1


def rules_ok(P):
    """start/end rules for each nonempty zigzag."""
    s2, a, b, c_, d_ = P
    s = Fr(s2, 2)
    n1, m1 = Fr(a + b, 2) + 1, Fr(abs(a - b), 2)
    n2, m2 = Fr(c_ + d_, 2) + 1, Fr(abs(c_ - d_), 2)
    bad = 0
    for k, (R, C) in enumerate(zigzags(P)):
        if not R or not C:
            continue
        dk = -s + 2 * k
        start_col = C[0] < R[0]
        end_row = R[-1] > C[-1]
        if end_row != (dk >= n1 - n2):
            bad += 1
        # structural: C[0] in {R[0]-1, R[0]}, C[-1] in {R[-1]-1, R[-1]}
        if C[0] not in (R[0] - 1, R[0]) or C[-1] not in (R[-1] - 1, R[-1]):
            bad += 1
        if (len(R) - len(C)) != (int(end_row) - int(start_col)):
            bad += 1
    return bad


if __name__ == '__main__':
    M, smax, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    L = []
    n = 0; mism = 0; nomono = 0; ex = []
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(b, M + 1):
                for d in range(c, M + 1):
                    tj = (3, a, b, c, d)
                    if sum(tj) % 2 or len(sites(tj)) < 2:
                        continue
                    n += 1; anymono = False
                    for P in pairings(tj):
                        imb = imb_pattern(P)
                        mono = monotone(imb)
                        anymono |= mono
                        if mono == lemma_bad(P):
                            mism += 1
                            if len(ex) < 5:
                                ex.append((P, imb))
                    nomono += (not anymono)
    L.append(f"Lemma M (j1=3/2), all tuples 2j2<=..<=2j5<={M}: {n} tuples x 3 pairings; criterion mismatches: {mism} {ex}; tuples without monotone pairing: {nomono}")
    nr = 0; badr = 0
    for s2 in range(1, smax + 1, 2):
        for tj in itertools.product(range(1, M // 2 + 1), repeat=4):
            P = (s2,) + tj
            if sum(P) % 2 or len(sites(P)) < 2:
                continue
            nr += 1; badr += rules_ok(P)
    L.append(f"zigzag start/end/interleave rules, all labelled tuples 2s<={smax} odd, 2j_i<={M//2}: {nr} tuples, violations: {badr}")
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

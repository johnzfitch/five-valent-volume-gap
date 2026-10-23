"""Referee: BR's Table 9 lists sigma_min as both (2,0,0,0,0,0,0,0,-2,0) and (2,0,0,0,0,0,0,0,+2,0).
Exact kernel and q_min for the '+' variant (A = 2 q123 + 2 q245) on the same spin family,
plus an explicit non-coplanar realisation of each variant (found by random integer search, re-verified here).
usage: python ref_sigma_plus.py NMAX out.txt
"""
import sys, itertools
from ref_full_exact import hw_space, Dop, commut_apply, add, restrict, spectrum_data, qmin_from, det3

REAL = {
    (2, 0, 0, 0, 0, 0, 0, 0, 2, 0): [(3, 1, 1), (1, -4, 4), (1, -2, 1), (-5, -5, -2), (-4, -5, -1), (-2, -6, -5)],
    (2, 0, 0, 0, 0, 0, 0, 0, -2, 0): [(1, -5, 0), (-5, 0, -4), (2, -6, 0), (2, 1, 4), (0, -1, 1), (5, -6, 4)],
}
ORDER = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]


def main():
    nmax = int(sys.argv[1]); out = open(sys.argv[2], "w", encoding="utf-8")
    for sig, T in REAL.items():
        eps = {}
        dets = []
        for c in itertools.combinations(range(6), 3):
            d = det3(*(T[i] for i in c)); dets.append(d); eps[c] = (d > 0) - (d < 0)
        s = tuple(eps[(I, J, K)] - eps[(I, J, 5)] + eps[(I, K, 5)] - eps[(J, K, 5)] for I, J, K in ORDER)
        out.write(f"tangents {T}: all 20 dets nonzero {all(dets)}, min|det| {min(map(abs, dets))}, sigma {s} == {sig}: {s == sig}\n")
    for n in range(1, nmax+1):
        tj = [1, 1, n, n, n+1]
        S, idx, N = hw_space(tj, n+1)
        D12 = Dop(tj, 0, 1, S, idx); D23 = Dop(tj, 1, 2, S, idx)
        D24 = Dop(tj, 1, 3, S, idx); D45 = Dop(tj, 3, 4, S, idx)
        EN = add(commut_apply(D12, D23, N), commut_apply(D24, D45, N))  # sigma245 = +2
        C = restrict(EN, N)
        odd, k, Rz = spectrum_data(C)
        line = f"sigma245=+2, n={n}: kernel dim {k}"
        if k < len(N):
            q, _ = qmin_from(Rz, 64 + 40*n)
            line += f", q_min = {q.str(12)}"
        print(line); out.write(line + "\n"); out.flush()


if __name__ == "__main__":
    main()

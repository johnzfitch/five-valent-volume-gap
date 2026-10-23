"""Reproduction of published numbers with this paper's own code.

R1  Bianchi-Haggard 1102.5439, table 'Loop gravity' column: V = (sqrt2/3) sqrt|q|,
    q eigenvalues of J1.(J2xJ3) (brute force from generators).
R2  Brunnemann-Rideout II (0706.0382) 7-vertex value lambda_V = sqrt(q_min) = 0.000597922
    for sigma of their Table 10 and spins (1,1,1,1,1,3/2,5/2) (brute force).
R3  BR II Table 5: minimising sorted spins of sigma_min = (4,2,2,0), exhaustive in the
    sorted convention, float64 grid (validated against brute force in validate.py).
R4  BR II Table 9: minimising sorted spins at valence 6 for sigma = (2,0,0,0,0,0,0,0,-2,0)
    (brute force), and their fit lambda_V ~ e^{3.72} e^{-1.215 * 2jmax}.
"""
import itertools, sys, time
from multiprocessing import Pool
import numpy as np
import mpmath as mp
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
import su2, grid


def r1(out):
    print("R1 Bianchi-Haggard volume table, V = (sqrt2/3) sqrt|q|:", file=out)
    for j in [(1, 1, 1, 1), (1, 1, 2, 2), (1, 1, 3, 3), (1, 2, 2, 3), (2, 2, 2, 2), (1, 1, 4, 4), (1, 2, 3, 4), (2, 2, 2, 4), (12, 12, 12, 14)]:
        H = su2.chain_4valent(j)
        legs, V = su2.invariant_space(list(j))
        qb = np.sort(np.abs(np.linalg.eigvals(su2.triple(legs, V, 0, 1, 2)))) if np.prod([x + 1 for x in j]) < 5000 else None
        ev = np.sort(np.linalg.eigvalsh(H))
        pos = ev[ev > 1e-9]
        vols = (np.sqrt(2) / 3) * np.sqrt(pos)
        zero = "0 and " if np.any(np.abs(ev) < 1e-9) else ""
        print(f"   2j={j}: V = {zero}{', '.join(f'{v:.3f}' for v in vols)}"
              + (f"   (brute force q agrees: {np.allclose(np.sort(np.abs(ev)), qb)})" if qb is not None else ""), file=out)


def r2(out):
    sig = (2, 2, 2, -4, 0, 0, 0, 0, -2, -2, 2, 2, 4, -2, 4, 4, 0, 0, 0, -2)
    tri = list(itertools.combinations(range(6), 3))   # 123 124 125 126 134 ... 456 (0-based)
    spins = [2, 2, 2, 2, 2, 3, 5]
    A = su2.sigma_operator(spins, sig, tri)
    ev = np.sort(np.abs(np.linalg.eigvalsh(1j * A)))
    nz = ev[ev > 1e-10 * ev.max()]
    q = nz.min()
    print(f"R2 BR 7-vertex, sigma of Table 10, 2j = {spins}: dim Inv = {A.shape[0]}, "
          f"kernel = {len(ev) - len(nz)}, q_min = {q:.10e}, lambda_V = sqrt(q_min) = {np.sqrt(q):.9f} (BR: 0.000597922)", file=out)


def _sorted_tuples(M):
    for t in itertools.combinations_with_replacement(range(1, M + 1), 4):
        tup = t + (M,)
        if sum(tup) % 2 == 0:
            yield tup


def _r3_job(tup):
    q, ker = grid.qmin(tup, alpha=4, beta=2, rel=1e-14)
    B = grid.bipartite(tup, 4, 2)
    smax = np.linalg.norm(B, 2) if B.size else 0
    return tup, q, ker, smax


def r3(out, Mmax=24, procs=8):
    # BR II Table 5 (p. 43), rows for 2jmax = 2..30, read from the page image and checked
    # at 200 dpi. The extracted text of the PDF drops and shifts rows; the image is authoritative.
    rows = ["1 1 2 2 2", "1 2 3 3 3", "1 3 4 4 4", "1 4 5 5 5", "1 5 6 6 6", "1 6 7 7 7", "1 7 8 8 8",
            "1 8 9 9 9", "3 8 9 10 10", "3 9 10 11 11", "3 10 11 12 12", "3 11 12 13 13", "3 12 13 14 14",
            "3 13 14 15 15", "5 13 14 16 16", "5 14 15 17 17", "5 15 16 18 18", "5 16 17 19 19",
            "5 17 18 20 20", "5 18 19 21 21", "5 19 20 22 22", "7 19 20 23 23", "7 20 21 24 24",
            "7 21 22 25 25", "7 22 23 26 26", "7 23 24 27 27", "7 24 25 28 28", "7 25 26 29 29",
            "9 25 26 30 30"]
    br = {M: tuple(int(x) for x in r.split()) for M, r in zip(range(2, 31), rows)}
    print(f"R3 BR Table 5, sigma = (4,2,2,0), exhaustive over sorted spins, 2jmax <= {Mmax}:", file=out)
    print("   2jmax  argmin (2j)            q_min        lambda_V=sqrt(q)  q/|A|    BR Table 5 row   agree", file=out)
    with Pool(procs) as pool:
        for M in range(2, Mmax + 1):
            t0 = time.time()
            res = [r for r in pool.map(_r3_job, list(_sorted_tuples(M)), chunksize=16) if r[1] is not None]
            tup, q, ker, smax = min(res, key=lambda r: r[1])
            ok = br.get(M) == tup
            print(f"   {M:5d}  {str(tup):22s} {q:.6e}  {np.sqrt(q):.6e}     {q / smax:.1e}  {str(br.get(M)):18s} {ok}"
                  f"   [{len(res)} tuples, {time.time() - t0:.0f}s]", file=out)
            out.flush()


def _v6_job(tup):
    tri = [(0, 1, 2), (1, 3, 4)]
    A = su2.sigma_operator(list(tup), (2, -2), tri)
    if A.shape[0] == 0:
        return tup, None, 0
    ev = np.sort(np.abs(np.linalg.eigvalsh(1j * A)))
    nz = ev[ev > 1e-12 * max(1, ev.max()) * len(ev)]
    return tup, (nz.min() if len(nz) else None), ev.max()


def r4(out, Mmax=8, procs=8):
    br9 = {1: (1, 1, 1, 1, 1, 1), 2: (1, 1, 1, 1, 2, 2), 3: (2, 2, 3, 3, 3, 3), 4: (1, 1, 3, 3, 4, 4), 5: (1, 1, 3, 3, 5, 5),
           6: (1, 1, 5, 5, 6, 6), 7: (1, 2, 2, 3, 7, 7), 8: (1, 1, 7, 7, 8, 8), 9: (1, 1, 7, 7, 9, 9),
           10: (1, 1, 9, 9, 10, 10), 11: (1, 1, 9, 9, 11, 11), 12: (1, 1, 11, 11, 12, 12), 13: (1, 1, 11, 11, 13, 13)}
    print(f"R4 BR Table 9 (valence 6, sigma = (2,0,0,0,0,0,0,0,-2,0), A = 2q123 - 2q245), exhaustive sorted, 2jmax <= {Mmax}:", file=out)
    print("   2jmax  argmin (2j)              q_min       lambda_V    BR fit e^{3.723-1.215*2jmax}  BR row  agree", file=out)
    with Pool(procs) as pool:
        for M in range(1, Mmax + 1):
            tups = [t + (M,) for t in itertools.combinations_with_replacement(range(1, M + 1), 5) if (sum(t) + M) % 2 == 0]
            res = [r for r in pool.map(_v6_job, tups, chunksize=4) if r[1] is not None]
            if not res:
                continue
            tup, q, mx = min(res, key=lambda r: r[1])
            fit = np.exp(3.723 - 1.215 * M)
            print(f"   {M:5d}  {str(tup):24s} {q:.5e}  {np.sqrt(q):.5e}  {fit:.3e}   {str(br9.get(M)):20s} {br9.get(M) == tup}", file=out)
            out.flush()


def rs_operator(spins2):
    """Q_RS = sum over all triples I<J<K<=N of |qhat_IJK| on Inv (De Pietri-Rovelli form)."""
    legs, V = su2.invariant_space(list(spins2))
    N = len(spins2)
    Q = np.zeros((V.shape[1], V.shape[1]), dtype=complex)
    for (I, J, K) in itertools.combinations(range(N), 3):
        A = su2.qhat(legs, V, I, J, K)          # anti-Hermitian
        w, U = np.linalg.eigh(1j * A)
        Q = Q + (U * np.abs(w)) @ U.conj().T
    return Q


def r5(out):
    print("R5 Rovelli-Smolin operator Q_RS = sum_T |qhat_T| (all triples), V = sqrt(Q_RS/32) (Kisielowski 2104.11010 Table 2):", file=out)
    for sp in [(1, 1, 1, 1), (2, 2, 2, 2, 2), (4, 4, 4, 4, 4)]:
        ev = np.linalg.eigvalsh(rs_operator(sp))
        vals = np.sqrt(np.clip(ev, 0, None) / 32)
        u, cnt = np.unique(np.round(vals, 6), return_counts=True)
        print(f"   2j={sp}: V = " + ", ".join(f"{a:.6f} (x{c})" for a, c in zip(u, cnt)), file=out)
    print("R5b RS spin-1/2 theorem spot check: min eigenvalue of Q_RS on tuples (1/2, j2..j5), 2j_i <= 5, dim Inv >= 2:", file=out)
    best = (1e9, None); n = 0
    for t in itertools.combinations_with_replacement(range(1, 6), 4):
        sp = (1,) + t
        if sum(sp) % 2:
            continue
        Q = rs_operator(sp)
        if Q.shape[0] < 2:
            continue
        n += 1
        lo = np.linalg.eigvalsh(Q).min()
        if lo < best[0]:
            best = (lo, sp)
    print(f"   {n} tuples, min eigenvalue {best[0]:.6f} at 2j = {best[1]} (bound 3 sqrt3 = {3 * np.sqrt(3):.6f}; trivial kernel throughout: {best[0] > 1e-9})", file=out)


if __name__ == "__main__":
    which = sys.argv[1]
    with open(sys.argv[2], "w") as f:
        if which == "r1": r1(f)
        if which == "r2": r2(f)
        if which == "r3": r3(f, int(sys.argv[3]))
        if which == "r4": r4(f, int(sys.argv[3]))
        if which == "r5": r5(f)

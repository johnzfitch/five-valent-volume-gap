"""Checks of the W2-campaign note [W2c] (own code, float):
 (S1a) for j1 = 1 (doubled 2), every pairing of every tuple with legs <= M: if a colour class is closed and N1+N2 is
       even, is that class the diagonal b = c?  (Prop. S1 asserts this.)  Counterexamples are listed.
 (S1b) Prop. S1's tridiagonal Q (gauge phi_b = (-1)^b psi_b) has the same spectrum as B^T B of the 2-D block K|C -> C'.
 (S1c) Theorem S1_0: j2 = j3, j4 = j5 (m1 = m2 = 0): lambda_min(Q) >= 1 + 1/(2 hi), and the two inequalities used in
       the proof (c_b - c_{b+1} >= b + 1/2 and (Qg)_b/g_b >= 1 + 1/(2b) for g_b = b), for n1 <= n2 <= NQ.
"""
import sys, math, itertools, time
import numpy as np


def h2(n, m, L):
    if n * n - L * L <= 0 or L * L - m * m <= 0: return 0.0     # link absent (outside m < L < n)
    return (n * n - L * L) * (L * L - m * m) / (4 * L * L - 1)


def Qmat(n1, m1, n2, m2, lo, hi):
    """Prop. S1 Q on b = lo..hi (undoubled; integer or half-integer labels as floats)"""
    bs = [lo + i for i in range(int(round(hi - lo)) + 1)]
    a = lambda l: h2(n1, m1, l) + h2(n2, m2, l)
    c = lambda l: 2 * math.sqrt(h2(n1, m1, l) * h2(n2, m2, l))
    n = len(bs); Q = np.zeros((n, n))
    for i, b in enumerate(bs):
        Q[i, i] = a(b) + a(b + 1)
        if i + 1 < n: Q[i, i + 1] = Q[i + 1, i] = -c(b + 1)
    return bs, Q


def grid_block(tj, k):
    S, A, B, C, D = tj
    bs = list(range(abs(A - B), A + B + 1, 2)); cs = list(range(abs(C - D), C + D + 1, 2))
    rect = set((b, c) for b in bs for c in cs)
    R = [(b, c) for b in bs for c in cs if abs(b - c) <= S <= b + c]; Rs = set(R)
    col = {p: ((p[0] + p[1] - S) // 2) % 2 for p in R}
    nb = lambda p: [q for q in ((p[0] + 2, p[1]), (p[0] - 2, p[1]), (p[0], p[1] + 2), (p[0], p[1] - 2)) if q in rect]
    cut = [p for p in R if any(q not in Rs for q in nb(p))]
    closed = [all(col[p] != kk for p in cut) and any(col[p] == kk for p in R) for kk in (0, 1)]
    n1, m1, n2, m2 = (A + B + 2) / 2, abs(A - B) / 2, (C + D + 2) / 2, abs(C - D) / 2
    Ck = [p for p in R if col[p] == k]; Co = [p for p in R if col[p] != k]; io = {p: i for i, p in enumerate(Co)}
    M = np.zeros((len(Co), len(Ck)))
    for j, p in enumerate(Ck):
        for q in nb(p):
            if q in io:
                if q[1] == p[1]: M[io[q], j] = math.sqrt(h2(n1, m1, max(p[0], q[0]) / 2))
                else: M[io[q], j] = math.sqrt(h2(n2, m2, max(p[1], q[1]) / 2))
    return closed, Ck, M, (n1, m1, n2, m2), len(bs), len(cs)


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    t0 = time.time(); cnt = 0; nondiag = []; specdiff = 0.0; ndiag = 0
    for legs in itertools.product(range(1, M + 1), repeat=4):
        tj = (2,) + legs
        if sum(tj) % 2: continue
        A, B, C, D = legs
        if (A, B) > (C, D) or A > B or C > D: continue
        for k in (0, 1):
            closed, Ck, Mb, (n1, m1, n2, m2), N1, N2 = grid_block(tj, k)
            if not closed[k] or (N1 + N2) % 2: continue
            cnt += 1
            diag = all(p[0] == p[1] for p in Ck)
            if not diag:
                nondiag.append((tj, k, Ck)); continue
            ndiag += 1
            bsd = sorted(p[0] for p in Ck)
            lo, hi = bsd[0] / 2, bsd[-1] / 2
            _, Q = Qmat(n1, m1, n2, m2, lo, hi)
            e1 = np.sort(np.linalg.eigvalsh(Q)); e2 = np.sort(np.linalg.eigvalsh(Mb.T @ Mb))
            specdiff = max(specdiff, float(np.max(np.abs(e1 - e2))))
    print(f"(S1a) j1 = 1, legs <= {M} (doubled; pairs unordered): {cnt} closed even-sum classes; {ndiag} are the diagonal; "
          f"{len(nondiag)} are NOT the diagonal")
    for x in nondiag[:6]:
        print(f"      e.g. tuple {x[0]}, closed colour {x[1]}, sites {x[2]}")
    print(f"(S1b) Prop. S1 Q vs 2-D block B^T B on the diagonal classes: max eigenvalue difference {specdiff:.2e}")
    worst = (9, None); worstc = (9, None); worstr = (9, None); viol = 0
    for n1 in range(2, NQ + 1):
        for n2 in range(n1, NQ + 1, 2):        # equal parity (m = 0 pairs: n = 2j + 1)
            hi = min(n1, n2) - 1
            bs, Q = Qmat(n1, 0, n2, 0, 1, hi)
            if len(bs) == 0: continue
            lm = np.linalg.eigvalsh(Q).min()
            r = lm / (1 + 1 / (2 * hi))
            if r < worst[0]: worst = (r, (n1, n2, lm))
            if lm < 1 + 1 / (2 * hi) - 1e-12: viol += 1
            c = lambda l: 2 * math.sqrt(h2(n1, 0, l) * h2(n2, 0, l))
            for b in range(1, hi + 1):
                d = c(b) - c(b + 1) - (b + 0.5)
                if d < worstc[0]: worstc = (d, (n1, n2, b))
            g = np.arange(1, hi + 1, dtype=float)
            rat = (Q @ g) / g - (1 + 1 / (2 * g))
            if rat.min() < worstr[0]: worstr = (rat.min(), (n1, n2))
    print(f"(S1c) n1 <= n2 <= {NQ}, equal parity, m1 = m2 = 0: lambda_min(Q) >= 1 + 1/(2hi) violations {viol}; "
          f"min ratio {worst[0]:.6f} at {worst[1]}")
    print(f"      min of c_b - c_(b+1) - (b + 1/2): {worstc[0]:.3e} at {worstc[1]};  min of (Qg)_b/g_b - (1 + 1/(2b)): {worstr[0]:.3e} at {worstr[1]}")
    print(f"total {time.time()-t0:.0f}s")

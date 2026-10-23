"""Checks of Theorem A (exact su(2) ladder => decomposition) and Theorem A-flat (flat T-ladder).
(1) Honest two-spin strip models: W = compression of J_x(j1') + J_x(j2') to |M| <= s (M = total J_z).
    (L) checked in exact rationals (diagonal entries; off-diagonal entries compared through their squares);
    wall diagonals checked; spectrum compared (float) with the union predicted by Theorems A and B.
(2) Random abstract exact ladders: sectors (kappa, interval, multiplicity) glued with random orthogonal
    frames on each V_d; (L) residual recomputed from the blocks; spectrum vs prediction.
(3) Random flat T-ladders: T = sum_d l_s(d)(N_d + N_d^T) with N_d^T N_d = N_{d-1} N_{d-1}^T; spectrum vs
    {|m_x| sqrt(g)} plus kernel."""
import numpy as np
from fractions import Fraction as Fr
rng = np.random.default_rng(3)


def Jlinks2(J2):
    """doubled spin; squared J_x links between mu and mu+1, mu = -j..j-1 (exact)"""
    j = Fr(J2, 2)
    return {(-j + i): (j - (-j + i)) * (j + (-j + i) + 1) / 4 for i in range(J2)}


def scalar_chain(lo, hi, kappa):
    """Jacobi on d = lo..hi, squared links kappa - d(d+1)/4"""
    n = int(round(hi - lo)) + 1
    d = lo + np.arange(n - 1)
    l = np.sqrt(np.maximum(kappa - d * (d + 1) / 4, 0))
    return np.diag(l, 1) + np.diag(l, -1)


def part1():
    nL = badL = badW = badspec = nmod = 0
    for S2 in range(1, 12):
        s = Fr(S2, 2)
        for J1 in range(0, 25):
            for J2 in range(0, 25):
                if (J1 + J2 - S2) % 2 or J1 + J2 < S2:
                    continue
                nmod += 1
                l1 = Jlinks2(J1); l2 = Jlinks2(J2)
                j1 = Fr(J1, 2); j2 = Fr(J2, 2)
                sites = [(-j1 + a, -j2 + b) for a in range(J1 + 1) for b in range(J2 + 1)
                         if abs(-j1 + a - j2 + b) <= s]
                idx = {x: i for i, x in enumerate(sites)}
                n = len(sites)
                W = np.zeros((n, n))
                nb = {i: [] for i in range(n)}
                for (m1, m2), i in idx.items():
                    for (y, w2) in [((m1 + 1, m2), l1.get(m1)), ((m1, m2 + 1), l2.get(m2))]:
                        if w2 is not None and y in idx:
                            k = idx[y]
                            W[i, k] = W[k, i] = float(w2) ** 0.5
                            nb[i].append((k, w2)); nb[k].append((i, w2))
                Md = lambda x: x[0] + x[1]
                for x, i in idx.items():
                    d = Md(x)
                    up = sum(w for k, w in nb[i] if Md(sites[k]) == d + 1)
                    dn = sum(w for k, w in nb[i] if Md(sites[k]) == d - 1)
                    if abs(d) < s:
                        nL += 1
                        if up - dn != -d / 2:
                            badL += 1
                        x2 = (x[0] + 1, x[1] - 1)
                        if x2 in idx:
                            pu = l1[x[0]] * l2[x[1] - 1]
                            pd = l2[x[1] - 1] * l1[x[0]]
                            if pu != pd:
                                badL += 1
                    elif d == s and dn - s / 2 < 0:
                        badW += 1
                    elif d == -s and up - s / 2 < 0:
                        badW += 1
                ev = np.sort(np.abs(np.linalg.eigvalsh(W))) if n else np.array([])
                pred = []
                for t in range(min(J1, J2) + 1):
                    S = Fr(J1 + J2, 2) - t
                    if S >= s:
                        ch = scalar_chain(float(-s), float(s), float(S * (S + 1) / 4))
                    else:
                        ch = scalar_chain(float(-S), float(S), float(S * (S + 1) / 4))
                    pred += list(np.abs(np.linalg.eigvalsh(ch)))
                pred = np.sort(pred)
                if len(pred) != len(ev) or (len(ev) and np.abs(pred - ev).max() > 1e-8):
                    badspec += 1
    return (f"(1) honest two-spin strips: {nmod} models (2s<=11, doubled spins<=24); exact (L) checks {nL}, "
            f"failures {badL}; wall-diagonal failures {badW}; spectrum vs Theorem A+B prediction failures {badspec}")


def rand_orth(n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def part2():
    fails = 0; minhalf = 9.0; minint = 9.0; T = 0
    for trial in range(400):
        S2 = int(rng.integers(1, 14)); s = S2 / 2
        dvals = [-s + k for k in range(S2 + 1)]
        sectors = []
        for _ in range(int(rng.integers(1, 6))):
            kap = s * (s + 1) / 4 + (float(rng.exponential(5)) if rng.random() < .8 else 0.0)
            sectors.append(('full', kap, int(rng.integers(1, 4))))
        Ss = [s - t for t in range(1, int(s) + 1) if s - t >= 0]
        for _ in range(int(rng.integers(0, 4))):
            if Ss:
                sectors.append(('hon', float(rng.choice(Ss)), int(rng.integers(1, 3))))
        Vd = {k: [] for k in range(S2 + 1)}
        for si, (typ, par, mult) in enumerate(sectors):
            lo, hi = (-s, s) if typ == 'full' else (-par, par)
            for k in range(S2 + 1):
                if lo - 1e-9 <= dvals[k] <= hi + 1e-9:
                    for c in range(mult):
                        Vd[k].append((si, c))
        Q = {k: rand_orth(len(Vd[k])) for k in Vd}
        blocks = []
        for k in range(S2):
            d = dvals[k]
            Mk = np.zeros((len(Vd[k + 1]), len(Vd[k])))
            for a, (si, c) in enumerate(Vd[k]):
                typ, par, mult = sectors[si]
                kap = par if typ == 'full' else par * (par + 1) / 4
                l2 = kap - d * (d + 1) / 4
                if (si, c) in Vd[k + 1] and l2 > 1e-12:
                    Mk[Vd[k + 1].index((si, c)), a] = np.sqrt(l2)
            blocks.append(Q[k + 1] @ Mk @ Q[k].T)
        off = np.cumsum([0] + [len(Vd[k]) for k in range(S2 + 1)])
        W = np.zeros((off[-1], off[-1]))
        for k in range(S2):
            W[off[k + 1]:off[k + 2], off[k]:off[k + 1]] = blocks[k]
            W[off[k]:off[k + 1], off[k + 1]:off[k + 2]] = blocks[k].T
        for k in range(1, S2):
            G = blocks[k].T @ blocks[k] - blocks[k - 1] @ blocks[k - 1].T + dvals[k] / 2 * np.eye(len(Vd[k]))
            if np.abs(G).max() > 1e-9:
                fails += 1
        ev = np.sort(np.abs(np.linalg.eigvalsh(W)))
        pred = []
        for typ, par, mult in sectors:
            ch = scalar_chain(-s, s, par) if typ == 'full' else scalar_chain(-par, par, par * (par + 1) / 4)
            pred += list(np.abs(np.linalg.eigvalsh(ch))) * mult
        pred = np.sort(pred)
        if len(pred) != len(ev) or np.abs(pred - ev).max() > 1e-8:
            fails += 1
        T += 1
        nz = ev[ev > 1e-9]
        if S2 % 2:
            minhalf = min(minhalf, ev.min())
        elif len(nz):
            minint = min(minint, nz.min())
    return (f"(2) random exact ladders: {T} models, failures {fails}; min |eig| (half-integer s) {minhalf:.6f}; "
            f"min nonzero |eig| (integer s) {minint:.6f}")


def part3():
    fails = 0
    for trial in range(300):
        S2 = int(rng.integers(1, 12)); s = S2 / 2
        g = rng.exponential(3, int(rng.integers(1, 6)))
        extra = [int(rng.integers(0, 3)) for _ in range(S2 + 1)]
        r = len(g)
        dims = [r + e for e in extra]
        Q = [rand_orth(n) for n in dims]
        N = []
        for k in range(S2):
            core = np.zeros((dims[k + 1], dims[k])); core[:r, :r] = np.diag(np.sqrt(g))
            N.append(Q[k + 1] @ core @ Q[k].T)
        off = np.cumsum([0] + dims)
        Tm = np.zeros((off[-1], off[-1]))
        for k in range(S2):
            d = -s + k; l = 0.5 * np.sqrt((s - d) * (s + d + 1))
            Tm[off[k + 1]:off[k + 2], off[k]:off[k + 1]] = l * N[k]
            Tm[off[k]:off[k + 1], off[k + 1]:off[k + 2]] = l * N[k].T
        ev = np.sort(np.abs(np.linalg.eigvalsh(Tm)))
        mx = np.abs(-s + np.arange(S2 + 1))
        pred = np.sort(np.concatenate([np.outer(mx, np.sqrt(g)).ravel(), np.zeros(sum(extra))]))
        if np.abs(pred - ev).max() > 1e-8:
            fails += 1
    return f"(3) random flat T-ladders: 300 models, spectrum = |m_x| sqrt(g) plus kernel: failures {fails}"


if __name__ == '__main__':
    for f in (part1, part2, part3):
        print(f(), flush=True)

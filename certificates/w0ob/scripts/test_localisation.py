"""Quantitative tests of the wall-model / localisation route for (RO), on the extremal family
(s; n-1/2, n, n, n) (pairing (n-1/2, n | n, n)):
 (1) d-profile of the lowest nonzero mode of W0 against the m_x = 1/2 eigenvector of J_x (spin s);
 (2) IMS cost: ||[W0, chi]|| for the linear cutoff chi(d) = (d+s)/(2s) (the smoothest d-partition
     with two pieces), and its square, which is what an IMS error term would be;
 (3) Schur complement onto the walls: lambda of W0 restricted to the interior diagonals |d| <= s-1
     (if the interior were gapped well above lambda(W0), a Schur complement could isolate the walls);
 (4) one-wall model: lambda of W0 on the half-strip d <= s (no wall at -s; region cut only by the rectangle).
Usage: python test_localisation.py"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from w0sparse import h2


def region_W(S, legs, keep):
    a2, a3, a4, a5 = legs
    n1 = a2 + a3 + 2; m1 = abs(a2 - a3); n2 = a4 + a5 + 2; m2 = abs(a4 - a5)
    sites = [(b, c) for b in range(m1, n1 - 1, 2) for c in range(m2, n2 - 1, 2) if keep(b, c)]
    idx = {x: i for i, x in enumerate(sites)}
    I = []; J = []; V = []
    for (b, c), i in idx.items():
        k = idx.get((b - 2, c))
        if k is not None:
            w = np.sqrt(h2(n1, m1, b)); I += [i, k]; J += [k, i]; V += [w, w]
        k = idx.get((b, c - 2))
        if k is not None:
            w = np.sqrt(h2(n2, m2, c)); I += [i, k]; J += [k, i]; V += [w, w]
    return sites, sp.csr_matrix((V, (I, J)), shape=(len(sites), len(sites)))


def lowest(W, k=10):
    if W.shape[0] < 1500:
        ev, vec = np.linalg.eigh(W.toarray())
    else:
        while True:
            ev, vec = spl.eigsh(W.tocsc(), k=k, sigma=1e-3, which='LM')
            a = np.abs(ev)
            if (a > 1e-7).sum() >= 2 or k > 400:
                break
            k = 2 * k + int((a <= 1e-7).sum())
    a = np.abs(ev); o = np.argsort(a)
    for i in o:
        if a[i] > 1e-7:
            return a[i], vec[:, i]
    return None, None


def jx_half(S):
    s = S / 2; Ms = np.arange(-s, s + 0.5)
    n = len(Ms); A = np.zeros((n, n))
    for i in range(n - 1):
        M = Ms[i]; A[i, i + 1] = A[i + 1, i] = 0.5 * np.sqrt((s - M) * (s + M + 1))
    ev, vec = np.linalg.eigh(A)
    i = np.argmin(np.abs(ev - 0.5))
    return Ms, vec[:, i] ** 2


if __name__ == '__main__':
    for S in (5, 9, 21):
        for n2x in (64, 256):
            legs = (n2x - 1, n2x, n2x, n2x)
            if (S + sum(legs)) % 2:
                legs = (n2x + 1, n2x, n2x, n2x)
            strip = lambda b, c: abs(b - c) <= S <= b + c
            sites, W = region_W(S, legs, strip)
            lam, v = lowest(W)
            d = np.array([(b - c) // 2 if (b - c) % 2 == 0 else (b - c) / 2 for b, c in sites], dtype=float)
            d = np.array([(b - c) / 2 for b, c in sites])
            prof = [(v[d == dd] ** 2).sum() for dd in np.arange(-S / 2, S / 2 + 0.5)]
            Ms, jx = jx_half(S)
            # (2) IMS commutator for chi(d) = (d+s)/(2s)
            chi = (d + S / 2) / S
            C = sp.diags(chi) @ W - W @ sp.diags(chi)
            cn = spl.svds(C.tocsc(), k=1, return_singular_vectors=False)[0] if C.shape[0] > 50 else np.linalg.norm(C.toarray(), 2)
            # (3) interior (walls removed)
            inter = lambda b, c: abs(b - c) <= S - 2 and S <= b + c
            _, Wi = region_W(S, legs, inter)
            lam_i, _ = lowest(Wi)
            # (4) one wall: d <= s only (and u >= s)
            onew = lambda b, c: (b - c) <= S and S <= b + c
            _, W1 = region_W(S, legs, onew)
            lam_1, _ = lowest(W1, k=20)
            print(f'2s={S} legs={legs}: lambda(W0)={lam:.4f}')
            print('   d-profile of low mode :', ' '.join('%.3f' % p for p in prof))
            print('   J_x(s) m_x=1/2 weights:', ' '.join('%.3f' % p for p in jx))
            print(f'   ||[W0,chi]|| = {cn:.2f}   (square {cn**2:.1f})   max link = {abs(W).max():.1f}')
            print(f'   interior strip |d|<=s-1: lambda = {lam_i:.4f};  one-wall half-strip d<=s: lambda = {lam_1:.4f}', flush=True)

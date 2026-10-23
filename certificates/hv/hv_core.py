"""Core routines: Ashtekar-Lewandowski volume operator at an N-valent gauge-invariant vertex.

Conventions (Brunnemann-Rideout 0706.0382, eqs. (2.26), (4.12)):
  Q_sigma = sum_{I<J<K<=N-1} sigma(IJK) * qhat_IJK ,  qhat_IJK = [(J_I+J_J)^2, (J_J+J_K)^2]
          = 4 [J_I.J_J, J_J.J_K] = -4 i J_I.(J_J x J_K)
  sigma(IJK) = eps(IJK) - eps(IJN) + eps(IKN) - eps(JKN),  eps = sgn det(tangents)
  V = l_P^3 sqrt(|Z Q_sigma|).  BR's printed lambda_V is sqrt(|eig of Q_sigma|) (sqrt|Z| l_P^3 omitted).

qhat is a real antisymmetric matrix in the chain basis; i*qhat is Hermitian, eigenvalues +-|.|.
We return the real antisymmetric matrix A = sum sigma * qhat ; |eig| = singular values of A.

Basis: chain coupling of legs 1..n (n = N-1): a_1 = j_1, a_k in a_{k-1} (x) j_k, a_n = L = j_N.
Spins are doubled integers throughout.
Reduced matrix elements in Edmonds' convention, <j||J||j> = sqrt(j(j+1)(2j+1)).
"""
from fractions import Fraction
from math import factorial, sqrt
import itertools
import numpy as np

# ------------------------------------------------------------------ exact 6j
_fact_cache = {}


def _f(n):
    v = _fact_cache.get(n)
    if v is None:
        v = factorial(n)
        _fact_cache[n] = v
    return v


def _tri(a, b, c):
    """doubled triangle check incl. integrality."""
    return (a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b


def _delta2(a, b, c):
    """Delta(abc)^2 as Fraction, doubled args."""
    return Fraction(_f((a + b - c) // 2) * _f((a - b + c) // 2) * _f((-a + b + c) // 2),
                    _f((a + b + c) // 2 + 1))


_sixj_cache = {}


def sixj_exact(a, b, c, d, e, f):
    """{a b c; d e f} (doubled args) returned as (S, P): value = S*sqrt(P), S Fraction, P Fraction>=0."""
    key = (a, b, c, d, e, f)
    r = _sixj_cache.get(key)
    if r is not None:
        return r
    if not (_tri(a, b, c) and _tri(a, e, f) and _tri(d, b, f) and _tri(d, e, c)):
        r = (Fraction(0), Fraction(0))
        _sixj_cache[key] = r
        return r
    P = _delta2(a, b, c) * _delta2(a, e, f) * _delta2(d, b, f) * _delta2(d, e, c)
    al = [(a + b + c) // 2, (a + e + f) // 2, (d + b + f) // 2, (d + e + c) // 2]
    be = [(a + b + d + e) // 2, (a + c + d + f) // 2, (b + c + e + f) // 2]
    S = Fraction(0)
    for t in range(max(al), min(be) + 1):
        num = _f(t + 1)
        den = 1
        for x in al:
            den *= _f(t - x)
        for x in be:
            den *= _f(x - t)
        S += Fraction((-1) ** t * num, den)
    r = (S, P)
    _sixj_cache[key] = r
    return r


def sixj(a, b, c, d, e, f):
    S, P = sixj_exact(a, b, c, d, e, f)
    return float(S) * sqrt(float(P))


# exact "signed square" representation: x = s*sqrt(r) is stored as the signed Fraction s*r
def ssq_from(S, P):
    """S*sqrt(P) -> signed square."""
    v = S * S * P
    return v if S >= 0 else -v


def ssq_float(x):
    return sqrt(float(x)) if x >= 0 else -sqrt(float(-x))


# ------------------------------------------------------------------ chain basis
def chains(Js, finals):
    """All chains (a_1..a_n), a_1 = J_1, for legs Js (doubled), with a_n in finals."""
    n = len(Js)
    out = []

    def rec(prefix):
        k = len(prefix)
        if k == n:
            if prefix[-1] in finals:
                out.append(tuple(prefix))
            return
        a = prefix[-1]
        j = Js[k]
        for b in range(abs(a - j), a + j + 1, 2):
            rec(prefix + [b])
    rec([Js[0]])
    return out


def reduced_J(Js, c, cp, I):
    """<c||J_I||cp> as a signed square (Fraction), I is 0-based leg index (0..n-1).
    Returns 0 if not connected."""
    n = len(Js)
    for k in range(I):
        if c[k] != cp[k]:
            return Fraction(0)
    one = 2
    jI = Js[I]
    # leg reduced element squared: j(j+1)(2j+1) with doubled: (J/2)(J/2+1)(J+1)
    val = Fraction(jI * (jI + 2) * (jI + 1), 4)
    if I >= 1:
        A = c[I - 1]
        J_, Jp = c[I], cp[I]
        # Edmonds 7.1.8: (-1)^{j1+j2+J+k} sqrt((2J+1)(2J'+1)) {j2 J j1; J' j2 k}
        ph = (A + jI + J_ + one) // 2
        S, P = sixj_exact(jI, J_, A, Jp, jI, one)
        if S == 0:
            return Fraction(0)
        t = ssq_from(S, P) * (J_ + 1) * (Jp + 1)
        if ph % 2:
            t = -t
        val = val * t
    for m in range(I + 1, n):
        a, ap = c[m - 1], cp[m - 1]
        jm = Js[m]
        J_, Jp = c[m], cp[m]
        # Edmonds 7.1.7: (-1)^{j1+j2+J'+k} sqrt((2J+1)(2J'+1)) {j1 J j2; J' j1' k}
        ph = (a + jm + Jp + one) // 2
        S, P = sixj_exact(a, J_, jm, Jp, ap, one)
        if S == 0:
            return Fraction(0)
        t = ssq_from(S, P) * (J_ + 1) * (Jp + 1)
        if ph % 2:
            t = -t
        val = val * t
    return val


class Vertex:
    """Gauge-invariant N-valent vertex, spins Jd (doubled, length N)."""

    def __init__(self, Jd):
        self.Jd = tuple(int(x) for x in Jd)
        self.N = len(self.Jd)
        self.legs = self.Jd[:-1]
        L = self.Jd[-1]
        self.L = L
        self.basis = chains(self.legs, {L})
        self.dim = len(self.basis)
        finals = {x for x in (L - 2, L, L + 2) if x >= 0}
        self.ext = chains(self.legs, finals)
        self.idx = {c: i for i, c in enumerate(self.basis)}
        self._red = {}
        self._dot = {}

    def red(self, I):
        """Reduced matrices of J_I as float, rows=basis (final L), cols=ext; also signed-square exact."""
        if I in self._red:
            return self._red[I]
        n = len(self.legs)
        R = np.zeros((self.dim, len(self.ext)))
        Rx = {}
        if not hasattr(self, "eidx"):
            self.eidx = {c: k for k, c in enumerate(self.ext)}
        for i, c in enumerate(self.basis):
            # neighbours: agree before level I, differ by 0 or +-1 (doubled +-2) from level I on
            for delta in itertools.product((-2, 0, 2), repeat=n - I):
                cp = tuple(c[:I]) + tuple(c[I + m] + delta[m] for m in range(n - I))
                k = self.eidx.get(cp)
                if k is None:
                    continue
                v = reduced_J(self.legs, c, cp, I)
                if v != 0:
                    Rx[(i, k)] = v
                    R[i, k] = ssq_float(v)
        self._red[I] = (R, Rx)
        return self._red[I]

    def dot(self, I, J):
        """Matrix of J_I . J_J on Inv (real symmetric), float64.
        Edmonds (7.1.6): <j|T.U|j> = (2j+1)^-1 sum_{j''} (-1)^{j-j''} <j||T||j''><j''||U||j>."""
        key = (min(I, J), max(I, J))
        if key in self._dot:
            return self._dot[key]
        RI, _ = self.red(I)
        RJ, _ = self.red(J)
        L = self.L
        ph = np.array([(-1) ** ((L - cp[-1]) // 2) for cp in self.ext], dtype=float)
        # <c||J_J||c''> for c'' in ext, c in basis -> need <c''||J_J||c> = (-1)^{c''_L - c_L} <c||J_J||c''> (real, Hermitian vector op)
        # Hermiticity of reduced elements: <a||T||b> = (-1)^{a-b} <b||T||a>^*  (Edmonds 5.5.? for J)
        # <c''||J_J||c'> = (-1)^{L''-L} <c'||J_J||c''> (Hermitian vector operator); this cancels the
        # (-1)^{L-L''} of (7.1.6), so M_{cc'} = sum_{c''} <c||J_I||c''><c'||J_J||c''> / (2L+1)
        M = RI @ RJ.T
        M = M / (L + 1)
        M = 0.5 * (M + M.T)
        self._dot[key] = M
        return M

    def qhat(self, I, J, K):
        """qhat_IJK = 4[J_I.J_J, J_J.J_K] (real antisymmetric), legs 0-based."""
        A = self.dot(I, J)
        B = self.dot(J, K)
        return 4.0 * (A @ B - B @ A)

    def Qsigma(self, sigma):
        """sigma: dict {(I,J,K) 0-based, I<J<K<=N-2 : value}."""
        M = np.zeros((self.dim, self.dim))
        for (I, J, K), s in sigma.items():
            if s:
                M += s * self.qhat(I, J, K)
        return M


# ------------------------------------------------------------------ sigma helpers
def triples(n):
    return list(itertools.combinations(range(n), 3))


def sigma_from_eps(eps, N):
    """eps: dict over 0-based triples of range(N) -> +-1/0. Returns sigma over triples of range(N-1).
    sigma(IJK) = eps(IJK) - eps(IJN) + eps(IKN) - eps(JKN)."""
    Nl = N - 1
    sig = {}
    for (I, J, K) in triples(Nl):
        sig[(I, J, K)] = eps[(I, J, K)] - eps[(I, J, Nl)] + eps[(I, K, Nl)] - eps[(J, K, Nl)]
    return sig


def eps_from_vectors(V):
    N = len(V)
    eps = {}
    for t in triples(N):
        d = np.linalg.det(np.array([V[t[0]], V[t[1]], V[t[2]]], dtype=float))
        eps[t] = int(np.sign(d)) if abs(d) > 1e-12 else 0
    return eps


def sigma_vec_BR(vals, N):
    """BR ordering: lexicographic triples of 1..N-1 (e.g. N=5: 123,124,134,234)."""
    ts = triples(N - 1)
    assert len(vals) == len(ts)
    return {t: v for t, v in zip(ts, vals)}


# ------------------------------------------------------------------ brute force
def spin_mats(J):
    j = J / 2.0
    ms = np.array([j - i for i in range(J + 1)])
    Jz = np.diag(ms).astype(np.complex128)
    Jp = np.zeros((J + 1, J + 1), dtype=np.complex128)
    for i in range(1, J + 1):
        m = ms[i]
        Jp[i - 1, i] = sqrt(j * (j + 1) - m * (m + 1))
    Jm = Jp.conj().T
    return (Jp + Jm) / 2, (Jp - Jm) / (2j), Jz


def brute_Qsigma_spectrum(Jd, sigma):
    """Spectrum of i*sum sigma qhat on SU(2)-invariant subspace of the full tensor product.
    qhat_IJK = -4 i J_I.(J_J x J_K) => i qhat = 4 J_I.(J_J x J_K)."""
    dims = [J + 1 for J in Jd]
    N = len(Jd)
    ops = []
    for leg, J in enumerate(Jd):
        mats = spin_mats(J)
        emb = []
        for M in mats:
            T = np.array([[1.0 + 0j]])
            for l2, dd in enumerate(dims):
                T = np.kron(T, M if l2 == leg else np.eye(dd))
            emb.append(T)
        ops.append(emb)
    tot = [sum(ops[l][a] for l in range(N)) for a in range(3)]
    C = sum(t @ t for t in tot)
    w, V = np.linalg.eigh(C)
    P = V[:, np.abs(w) < 1e-8]
    D = np.zeros_like(C)
    for (I, J, K), s in sigma.items():
        if not s:
            continue
        a, b, c = ops[I], ops[J], ops[K]
        trip = (a[0] @ (b[1] @ c[2] - b[2] @ c[1]) + a[1] @ (b[2] @ c[0] - b[0] @ c[2])
                + a[2] @ (b[0] @ c[1] - b[1] @ c[0]))
        D += 4 * s * trip
    R = P.conj().T @ D @ P
    return np.sort(np.linalg.eigvalsh((R + R.conj().T) / 2)), P.shape[1]


def spectrum_abs(A):
    """|eigenvalues| of real antisymmetric A, ascending (= singular values)."""
    return np.sort(np.linalg.svd(A, compute_uv=False))

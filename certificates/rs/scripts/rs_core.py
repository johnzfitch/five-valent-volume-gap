"""Rovelli-Smolin vertex operator Q_RS = sum over ALL triples T of the N legs of |qhat_T| on Inv(j_1..j_N),
qhat_IJK = [(J_I+J_J)^2,(J_J+J_K)^2] = -4i J_I.(J_J x J_K) (Brunnemann-Thiemann normalisation).
V_RS = const * sqrt(Q_RS) (De Pietri-Rovelli (7.15)). Uses the recoupling-basis code of
../../volume_gap_higher_valence_20261021/hv_core.py (imported read-only); triples containing the last leg N
are eliminated with J_N = -sum_{L<N} J_L:  qhat_{IJN} = -sum_{L != I,J, L<N} qhat_{IJL},
qhat_{INK} = qhat_{KIN}-type identities via antisymmetry.
"""
import sys, itertools, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))
from hv_core import Vertex, brute_Qsigma_spectrum

def absm(A):
    """|A| for real antisymmetric A (via the Hermitian matrix iA, avoiding the sqrt of rounding noise)."""
    w, U = np.linalg.eigh(1j * A)          # iA Hermitian; |A| = U |w| U^H (real)
    return ((U * np.abs(w)) @ U.conj().T).real

def qhats(V):
    """dict triple(0-based, I<J<K<=N-1) -> real antisymmetric matrix qhat on Inv."""
    N = V.N; n = N - 1
    Q = {}
    for t in itertools.combinations(range(n), 3):
        Q[t] = V.qhat(*t)
    # triples with last leg: (I,J,N-1)
    for I, J in itertools.combinations(range(n), 2):
        M = np.zeros((V.dim, V.dim))
        for L in range(n):
            if L in (I, J): continue
            a = tuple(sorted((I, J, L)))
            # sign of permutation (I,J,L) -> sorted
            perm = [I, J, L]; s = 1
            for x in range(3):
                for y in range(x+1, 3):
                    if perm[x] > perm[y]: s = -s
            M -= s * Q[a]
        Q[(I, J, n)] = M
    return Q

def Q_RS(V):
    return sum(absm(M) for M in qhats(V).values())

def rs_spectrum(Jd):
    V = Vertex(Jd)
    if V.dim == 0: return None, 0
    Q = Q_RS(V)
    w = np.linalg.eigvalsh(Q)
    return w, V.dim

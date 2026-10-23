"""Kernel dimension of the normal-form operator versus the bipartite imbalance |n_even - n_odd| of the grid,
all labelled tuples with 2j_i <= M, orbits 1, 2, 3.  Rank computed in exact rational arithmetic (python-flint
fmpq_mat) on the squared-weight-free form: W is similar to a matrix with entries sqrt(rational), so we use the
exact rank of B^T B? -- not rational either.  We therefore use float SVD with a 1e-9 relative threshold and report
the separation (largest 'zero' and smallest nonzero singular values)."""
import sys, os, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import W_matrix
from survey_grid import bip_block
M = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for orb, (al, be) in {1: (2, 0), 2: (2, 2), 3: (4, 2)}.items():
    extra = 0; tot = 0; ex_examples = []; sep = (0, 1e9)
    for J in itertools.product(range(1, M + 1), repeat=5):
        if sum(J) % 2: continue
        S, W = W_matrix(J, al, be)
        if len(S) == 0: continue
        B = bip_block(S, W)
        ne, no = B.shape
        s = np.linalg.svd(B, compute_uv=False) if B.size else np.zeros(0)
        smax = s.max() if s.size else 1.0
        z = (s < 1e-9 * smax).sum() + abs(ne - no)
        ker = z
        if s.size:
            zs = s[s < 1e-9 * smax]; nz = s[s >= 1e-9 * smax]
            if zs.size: sep = (max(sep[0], zs.max() / smax), sep[1])
            if nz.size: sep = (sep[0], min(sep[1], nz.min() / smax))
        tot += 1
        if ker > abs(ne - no):
            extra += 1
            if len(ex_examples) < 6: ex_examples.append((J, ne, no, int(ker)))
    print(f"orbit {orb}: {tot} tuples (2j_i<={M}); kernel > |n_e-n_o| in {extra}; examples (J, n_e, n_o, ker): {ex_examples}; "
          f"max zero sv/smax {sep[0]:.1e}, min nonzero sv/smax {sep[1]:.1e}")

# orbit 2: are the extra kernels exactly the pair-symmetric tuples {j2,j3} = {j4,j5}?  And is the excess then
# the dimension of the pair-swap-odd sector's imbalance?  Count.
sym_extra = nonsym_extra = sym_noextra = 0
for J in itertools.product(range(1, M + 1), repeat=5):
    if sum(J) % 2: continue
    S, W = W_matrix(J, 2, 2)
    if len(S) == 0: continue
    B = bip_block(S, W)
    ne, no = B.shape
    s = np.linalg.svd(B, compute_uv=False) if B.size else np.zeros(0)
    ker = (s < 1e-9 * (s.max() if s.size else 1)).sum() + abs(ne - no)
    sym = sorted(J[1:3]) == sorted(J[3:5])
    if ker > abs(ne - no):
        if sym: sym_extra += 1
        else: nonsym_extra += 1
    elif sym: sym_noextra += 1
print(f"orbit 2 extra kernel: pair-symmetric {sym_extra}, non-symmetric {nonsym_extra}; symmetric without extra kernel {sym_noextra}")

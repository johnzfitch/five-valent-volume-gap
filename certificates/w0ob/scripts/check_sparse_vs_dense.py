"""Cross-check of the shift-invert Lanczos value of lambda(W0) against a dense SVD of the bipartite block."""
from w0sparse import lam_sparse
for S, legs in [(41, (127, 128, 128, 128)), (81, (127, 128, 128, 128)), (21, (255, 256, 256, 256)), (6, (140, 142, 142, 144)), (5, (162, 163, 163, 163))]:
    a, N = lam_sparse(S, legs, dense_max=0)
    b, _ = lam_sparse(S, legs, dense_max=10**6)
    print(f'2j1={S} legs={legs} dim={N}: Lanczos {a:.12f}  dense {b:.12f}  rel.diff {abs(a-b)/b:.1e}')

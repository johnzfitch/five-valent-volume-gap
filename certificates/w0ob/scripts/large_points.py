"""Two further points of family A at 2n = 2048 (shift-invert Lanczos), recorded in outputs/scan_large_s_2048.txt."""
from w0sparse import lam_sparse
for S, legs in [(41, (2047, 2048, 2048, 2048)), (161, (2047, 2048, 2048, 2048))]:
    l, N = lam_sparse(S, legs, dense_max=0)
    print(f'2j1={S} legs={legs} dim={N} lambda(W0)={l:.5f} lambda*(j1+1)={l*(S/2+1):.2f}')

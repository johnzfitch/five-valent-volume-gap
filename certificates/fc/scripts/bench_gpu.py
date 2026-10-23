"""GPU vs CPU for the linear-algebra part of the certificate (A = B^T B, eigh, two congruence products), on all
tuples of one level. Building the exact rational data stays on the CPU in both cases (it is Python/flint work)."""
import time, numpy as np, collections, sys
from run_cert import tuples
from certcore import matrices
M = int(sys.argv[1])
mats = []
t0 = time.time()
for tj in tuples(M):
    D = matrices(tj)
    if D is None or D['nO'] == 0: continue
    mats.append(D['Bf'])
tbuild = time.time() - t0
t0 = time.time()
for B in mats:
    A = B.T @ B; w, X = np.linalg.eigh(A); Mc = X.T @ (A @ X)
tcpu = time.time() - t0
import torch
dev = torch.device('cuda')
groups = collections.defaultdict(list)
for B in mats: groups[B.shape].append(B)
torch.cuda.synchronize(); t0 = time.time()
for shp, lst in groups.items():
    Bt = torch.tensor(np.stack(lst), dtype=torch.float64, device=dev)
    A = Bt.transpose(1, 2) @ Bt; w, X = torch.linalg.eigh(A); Mc = X.transpose(1, 2) @ (A @ X)
    _ = Mc.cpu()
torch.cuda.synchronize(); tgpu = time.time() - t0
msg = (f"level {M}: {len(mats)} matrices in {len(groups)} shapes; building (CPU, exact data) {tbuild:.1f}s; "
       f"linear algebra numpy 1 process {tcpu:.1f}s; torch CUDA float64 batched by shape incl. transfers {tgpu:.1f}s "
       f"({torch.cuda.get_device_name(0)})")
print(msg); open(f"../outputs/bench_gpu_M{M}.txt", "w").write(msg + "\n")

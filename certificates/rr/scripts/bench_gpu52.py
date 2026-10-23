"""GPU vs CPU for the linear-algebra part of the full-matrix certificate on a bounded real workload:
the first 2000 j1 = 5/2 tuples of level 2j_max = 60 (exact data built on the CPU in both cases)."""
import time, collections, numpy as np, itertools
from certcore import matrices
def tuples52(M):
    for j2 in range(1, M + 1):
        for j3 in range(j2, M + 1):
            for j4 in range(j3, M + 1):
                j5 = M
                if (5 + j2 + j3 + j4 + j5) % 2: continue
                yield (5, j2, j3, j4, j5)
if __name__ == '__main__':
    tl = list(itertools.islice(tuples52(60), 2000))
    t0 = time.time(); mats = []
    for tj in tl:
        D = matrices(tj)
        if D is None or D['nO'] == 0: continue
        mats.append(D['Bf'])
    tb = time.time() - t0
    t0 = time.time()
    for B in mats:
        A = B.T @ B; w, X = np.linalg.eigh(A); Mc = X.T @ ((A - np.eye(len(A))) @ X)
    tc = time.time() - t0
    import torch
    dev = torch.device('cuda'); groups = collections.defaultdict(list)
    for B in mats: groups[B.shape].append(B)
    torch.cuda.synchronize(); t0 = time.time()
    for shp, lst in groups.items():
        Bt = torch.tensor(np.stack(lst), dtype=torch.float64, device=dev)
        A = Bt.transpose(1, 2) @ Bt; w, X = torch.linalg.eigh(A)
        I = torch.eye(A.shape[-1], dtype=torch.float64, device=dev)
        Mc = X.transpose(1, 2) @ ((A - I) @ X); _ = Mc.cpu()
    torch.cuda.synchronize(); tg = time.time() - t0
    sizes = [m.shape[1] for m in mats]
    msg = (f"j1=5/2, level 60, first {len(tl)} tuples: {len(mats)} blocks, nO in [{min(sizes)}, {max(sizes)}], {len(groups)} shapes; "
           f"exact build (CPU) {tb:.1f}s; linear algebra numpy 1 process {tc:.1f}s; torch CUDA float64 batched incl. transfers {tg:.1f}s "
           f"({torch.cuda.get_device_name(0)})")
    print(msg); open("../outputs/bench_gpu52.txt", "w").write(msg + "\n")

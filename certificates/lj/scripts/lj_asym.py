"""Large-s behaviour at fixed three legs t=(2j2,2j3,2j4) and fixed M = s - j5, via the exact pencil.
Reports lim q/(8(s+1)) vs nu(t;M), the offset q/8 - (s+1) nu, and kernel dimensions."""
import sys, os, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lj_pencil as P


def gap_and_ker(ev, rel=1e-8):
    a = np.abs(ev); thr = rel * max(1, a.max())
    nz = a[a > thr]
    return (nz.min() if nz.size else None), int(a.size - nz.size)


if __name__ == '__main__':
    K = int(sys.argv[1])
    print(" t(doubled)  M    dimV[M] kerUz  nu          | s=50: q/8-(s+1)nu  ker | s=400: q/8-(s+1)nu ker | s=3200 ...")
    worst_ratio = 0
    for t in itertools.combinations_with_replacement(range(1, K + 1), 3):
        o = P.three(t)
        for M2 in range(-sum(t), sum(t) + 1, 2):
            M = M2 / 2
            idx = np.where(np.abs(o['Mz'] - M) < 1e-9)[0]
            if M < 0 or idx.size == 0: continue
            evU = np.linalg.eigvalsh(o['Uz'][np.ix_(idx, idx)])
            nu, kU = gap_and_ker(evU)
            if nu is None: continue
            line = f"{t} {M:5.1f} {idx.size:5d} {kU:4d} {nu:.6f} |"
            for s in (50.0 + (M % 1), 400.0 + (M % 1), 3200.0 + (M % 1)):
                ev, herm, dim = P.pencil_spec(t, s, s - M)
                g, k = gap_and_ker(ev)
                line += f" {g - (s + 1) * nu:9.4f} {k:3d} |"
            print(line, flush=True)

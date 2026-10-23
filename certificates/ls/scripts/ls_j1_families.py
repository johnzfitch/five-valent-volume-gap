"""q/(2 j1) along families with j1 -> infinity (doubled spins), to test linear control by j1."""
import sys, os, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ls_core as L
def fix(t):
    t = list(t)
    if sum(t) % 2: t[-1] += 1
    return tuple(t)
FAM = {
 'K2 (J; 1/2, 1, 1, J-1/2)': lambda S: (S, 1, 2, 2, S - 1),
 'K1 (J; 1/2, J, 1, 1)': lambda S: (S, 1, S, 2, 2),
 'K3 (J; 1, 3/2, 3/2, J-1)': lambda S: (S, 2, 3, 3, S - 2),
 'R1 (J; J/2, J/2, J/2, J/2)': lambda S: fix((S, S // 2, S // 2, S // 2, S // 2)),
 'R2 (J; J/4 x4 +1)': lambda S: fix((S, S // 4 + 1, S // 4 + 1, S // 4 + 1, S // 4 + 1)),
 'R3 (J; J, J, 1/2, 1/2)': lambda S: fix((S, S, S, 1, 1)),
 'R4 (J; J, J, J, J)': lambda S: fix((S, S, S, S, S)),
 'R5 (J; 2J,2J,2J,2J)': lambda S: fix((S, 2 * S, 2 * S, 2 * S, 2 * S)),
 'R6 (J; J/2, J/2+1/2, 1, 1)': lambda S: fix((S, S // 2, S // 2 + 1, 2, 2)),
}
for name, f in FAM.items():
    print('==', name)
    for S in [9, 17, 33, 65, 129, 257, 513, 1025, 4097]:
        for SS in (S, S + 1):
            tj = f(SS)
            n = L.dim_inv(tj)
            if n < 2 or n > 40000: continue
            q, ker = L.qmin(tj)
            if q is None: continue
            print(f"  {tj} dim {n} q={q[0]:.4f} q/(2j1)={q[0]/tj[0]:.4f} q/(2jmax)={q[0]/max(tj):.4f}", flush=True)

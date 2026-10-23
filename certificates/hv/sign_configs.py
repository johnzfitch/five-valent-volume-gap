"""Realizable non-coplanar sign configurations for 5- and 6-valent vertices (Poisson sprinkling on S^2,
as in Brunnemann-Rideout 0706.0382 sec. 4.2.2), and the induced sigma-configurations (4.12).
Writes sigma5.json (list of sigma 4-tuples, BR order 123,124,134,234) and prints the counts
to compare with BR Table 1 (5-valent: 384 eps-configs, 171 sigma-configs; 6-valent: 23,808 and 16,413).
Usage: python sign_configs.py [n_samples5] [n_samples6]
"""
import sys
import json
import itertools
import numpy as np


def configs(N, nsamp, seed):
    rng = np.random.default_rng(seed)
    T = list(itertools.combinations(range(N), 3))
    Tl = list(itertools.combinations(range(N - 1), 3))
    idx = np.array(T)
    eps_set, sig_set = set(), set()
    batch = 200000
    done = 0
    while done < nsamp:
        V = rng.normal(size=(batch, N, 3))
        M = V[:, idx, :]                                  # (batch, nT, 3, 3)
        d = np.linalg.det(M)
        e = np.sign(d).astype(np.int8)
        for row in np.unique(e, axis=0):
            eps_set.add(tuple(int(x) for x in row))
        done += batch
    pos = {t: i for i, t in enumerate(T)}
    for e in eps_set:
        s = []
        for (I, J, K) in Tl:
            Nl = N - 1
            s.append(e[pos[(I, J, K)]] - e[pos[(I, J, Nl)]] + e[pos[(I, K, Nl)]] - e[pos[(J, K, Nl)]])
        sig_set.add(tuple(s))
    return eps_set, sig_set


def main():
    n5 = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    n6 = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000
    e5, s5 = configs(5, n5, 1)
    print(f"5-valent: {n5} sprinkles -> {len(e5)} eps-configs, {len(s5)} sigma-configs (BR Table 1: 384, 171)")
    nz = sorted(s for s in s5 if any(s))
    reps = sorted({max(s, tuple(-x for x in s)) for s in nz})
    print(f"          nonzero sigma up to overall sign: {len(reps)} (BR: 85)")
    with open("sigma5.json", "w") as f:
        json.dump(reps, f)
    chain_type = [s for s in reps if s[3] == 0 and s[1] == s[2] and s[0] != 0]
    print("          chain-type sigma (alpha, beta, beta, 0) realizable:", chain_type)
    chain_type2 = [s for s in reps if s[3] == 0 and s[1] == -s[2] and s[0] != 0]
    print("          (alpha, beta, -beta, 0):", chain_type2)
    e6, s6 = configs(6, n6, 2)
    print(f"6-valent: {n6} sprinkles -> {len(e6)} eps-configs, {len(s6)} sigma-configs (BR Table 1: 23,808, 16,413)")


if __name__ == "__main__":
    main()

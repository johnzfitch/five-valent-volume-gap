"""Finite-step congruence test for monotonicity of the scalar ladder in kappa.
rho_i = l_i(k')/l_i(k); solve sigma_0 = 1, sigma_{j+1} = rho_j / sigma_j;
condition: min_{even j} sigma_j * min_{odd j} sigma_j >= 1."""
import numpy as np
def cond(S2, k, k2):
    s = S2/2; d = -s + np.arange(S2)
    rho = np.sqrt((k2 - d*(d+1)/4)/(k - d*(d+1)/4))
    sig = [1.0]
    for r in rho: sig.append(r/sig[-1])
    sig = np.array(sig)
    return sig[0::2].min()*sig[1::2].min()
worst = 9
for S2 in range(1, 122):
    s = S2/2; k0 = s*(s+1)/4
    for g in np.concatenate([[0], np.geomspace(1e-6, 1e6, 60)]):
        for f in [1+1e-9, 1.001, 1.1, 2, 10, 1e3]:
            k = k0 + g; k2 = k*f
            c = cond(S2, k, k2)
            if c < worst: worst = c; arg = (S2, g, f)
print("min over grid of (min_even sigma)(min_odd sigma):", worst, "at", arg)

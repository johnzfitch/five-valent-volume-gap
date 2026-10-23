"""Theorem 1 numerics: q_min (ladder model, high precision) vs the sign-independent transfer bound B(n) and the
elementary bound B2(n) = 22700 ((n+1)sqrt(2n)/4 + 1) sqrt(12) (n+2)^{(n-3)/2} / ((n-2)!)^{3/2}, odd n.
Also checks the lemma hypotheses: y_j >= 3x (3<=j<=n-1), Phi_0 <= 30, S = sum_{j>=3} x^2/y_j^2 <= 4/3,
|det T_rest|^2 = (n+5)/(2n+2), and the two-chain kernel structure (trivial kernel for odd n, dim 2 for even n).
q_min: inverse iteration on the model K (mp.lu_solve), dps = 3n + 60.
"""
import mpmath as mp, math, time
from ladder import model
from bound import coeffs

def qmin(n):
    mp.mp.dps = 3 * n + 60
    K, S = model(n, (1, 1), ctx=mp)
    v = mp.matrix([mp.mpf(1) / (k + 1) for k in range(K.rows)])
    for _ in range(4):
        w = mp.lu_solve(K, v)
        lam = mp.norm(v) / mp.norm(w)
        v = w / mp.norm(w)
    return 8 * lam

def B(n):
    mp.mp.dps = 50
    x, y, M1, M2 = coeffs(n)
    fro = lambda m: mp.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2)
    phi0 = fro(M1(0)) * fro(M2(1))
    det_rest = mp.mpf(1); grow = mp.mpf(1)
    for i in range(2, n - 2, 2):
        a1, b1, c1 = M1(i); a2, b2, c2 = M2(i + 1)
        det_rest *= a1 * b1 * a2 * b2
        grow *= c1 * c2 * (1 - b2 * a1 / (c2 * c1) - 2 * b1 / c1)
    yn = y(n)
    hyp_y = min(y(j) / x for j in range(3, n)) if n > 3 else None
    S = sum(x ** 2 / y(j) ** 2 for j in range(3, n))
    val = 8 * mp.sqrt(3) * (yn + mp.sqrt(2) * x ** 2 / yn) * phi0 * det_rest / grow
    return val, phi0, det_rest, hyp_y, S

def B2(n):
    lg = (math.log(22700) + math.log((n + 1) * math.sqrt(2 * n) / 4 + 1) + 0.5 * math.log(12)
          + 0.5 * (n - 3) * math.log(n + 2) - 1.5 * math.lgamma(n - 1))
    return lg / math.log(10)

def main():
  lines = []
  t0 = time.time()
  for n in list(range(7, 32, 2)) + [41, 51, 61, 81]:
      q = qmin(n)
      b, phi0, dr, hy, S = B(n)
      lq = float(mp.log10(q)); lb = float(mp.log10(b)); lb2 = B2(n)
      okdet = abs(dr ** 2 - mp.mpf(n + 5) / (2 * n + 2)) < mp.mpf(10) ** -40
      lines.append(f"n={n:3d} j_max={(n+1)/2:5.1f}: log10 q_min = {lq:9.3f}, log10 B = {lb:9.3f}, log10 B2 = {lb2:9.3f};"
                   f" q<=B<=B2: {lq <= lb <= lb2}; min y_j/x = {float(hy):.3f}; Phi0 = {float(phi0):.3f}; S = {float(S):.4f};"
                   f" |detT_rest|^2=(n+5)/(2n+2): {okdet}; log q + 1.5 n ln n /ln10 = {lq + 1.5*n*math.log(n)/math.log(10):.3f}")
      print(lines[-1], flush=True)
  # Phi0 and S over a long range (no q needed)
  mp.mp.dps = 30
  pm = 0; sm = 0
  for n in list(range(7, 400, 2)) + [1001, 4001, 20001]:
      b, phi0, dr, hy, S = (None,) + B(n)[1:] if False else B(n)
      pm = max(pm, float(phi0)); sm = max(sm, float(S))
      assert float(hy) >= 3, n
  lines.append(f"over odd n in [7,399] and n = 1001, 4001, 20001: max Phi0 = {pm:.4f} (<= 30), max S = {sm:.4f} (<= 4/3), min_j y_j/x >= 3 always")
  # kernel structure
  for n in range(1, 13):
      mp.mp.dps = 40
      K, S_ = model(n, (1, 1), ctx=mp)
      sv = sorted(mp.svd_r(K, compute_uv=False))
      kern = sum(1 for s in sv if s < mp.mpf(10) ** -30)
      lines.append(f"kernel dim (40 digits) n={n}: {kern}")
  lines.append(f"wall time {time.time()-t0:.0f} s")
  print("\n".join(lines[-14:]))
  open("../outputs/v6_bound_output.txt", "w").write("\n".join(lines) + "\n")

if __name__ == '__main__':
    main()

"""[Z] beyond the box: the Corollary-5 bound with the box's forms (same formulas as rb_z32.regime_bound), evaluated in
binary64 at sampled parameters n1 = 4000 .. 30000 (two-parameter regimes) and n1 up to 3e5 (families), where the written
proof relies on the closed-form tails of z32_certify_tail.py.  This tests the tail's conclusion (8 sigma >= 3.7) at those
points; it is not a proof of the tail.  Usage: python rb_z32_large.py > ../outputs/rb_z32_large_out.txt"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_z32 as Z

if __name__ == '__main__':
    t0 = time.time()
    worst = (1e9, None)
    for name in ('R1', 'R2+', 'R2-', 'R4'):
        amin = {'R1': 1.5, 'R2+': 0.5, 'R2-': 1.0, 'R4': 1.5}[name]
        for n in (4000.0, 4000.5, 10000.0, 30000.5):
            alist = sorted(set([amin, amin + 1, amin + 2, amin + 5, amin + 20, math.floor(n / 10) + (n % 1) - (amin % 1 != n % 1) * 0.5,
                                math.floor(n / 2) + (n % 1), n - 2 if name != 'R4' else n + 0.5, n - 3 if name != 'R4' else n - 0.5]))
            for a in alist:
                if name != 'R4' and (a < amin or a > n - 2 or ((n - a) % 1) != 0): continue
                if name == 'R4' and (a < amin or a > n + 0.5 or ((n - a) % 1) != 0.5): continue
                v, s2 = Z.regime_bound(name, n, a)
                print(f"{name:4s} n1={n:8.1f} a={a:8.1f}: 8sigma bound {v:9.4f}", flush=True)
                if v < worst[0]: worst = (v, (name, n, a))
    for name in ('R1c', 'R4c', 'Tp', 'Tm'):
        for n in (100000.0, 300000.0):
            n_ = n + (0.5 if name == 'R1c' else 0.0)
            v, s2 = Z.regime_bound(name, n_, 0)
            print(f"{name:4s} n1={n_:9.1f}: 8sigma bound {v:9.4f}", flush=True)
            if v < worst[0]: worst = (v, (name, n_, 0))
    print(f"minimum over the samples: {worst[0]:.4f} at {worst[1]} (claim: >= 3.7)   ({time.time()-t0:.0f}s)")

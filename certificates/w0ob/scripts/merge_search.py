"""Merge the per-seed search outputs: per-2j1 minimum of the best-pairing lambda(W0) and the overall extremes,
separately for half-integer and integer j1.  Usage: python merge_search.py OUTDIR"""
import sys, re, glob, os
d = sys.argv[1]
per = {}; total = 0
for f in sorted(glob.glob(os.path.join(d, 'search_seed*.txt'))):
    txt = open(f).read().splitlines()
    total += int(re.search(r'samples=(\d+)', txt[0]).group(1))
    for line in txt:
        m = re.match(r'2j1=\s*(\d+) n=\s*(\d+) min best-lambda=([\d.]+) at (\(.*?\)) (\w+)', line)
        if m:
            S = int(m.group(1)); n = int(m.group(2)); l = float(m.group(3))
            c = per.setdefault(S, [0, 9e9, None])
            c[0] += n
            if l < c[1]:
                c[1] = l; c[2] = (m.group(4), m.group(5))
print(f'merged samples: {total}')
for S in sorted(per):
    n, l, at = per[S]
    print(f'2j1={S:3d} tuples={n:5d} min best-pairing lambda(W0)={l:.5f} lambda*(j1+1)={l*(S/2+1):.3f} at {at[0]} {at[1]}')
half = [(per[S][1], S) for S in per if S % 2]
integ = [(per[S][1], S) for S in per if S % 2 == 0]
print('half-integer j1: overall min lambda = %.5f at 2j1=%d; min lambda*(j1+1) = %.3f' % (min(half) + (min(per[S][1] * (S / 2 + 1) for S in per if S % 2),)))
print('integer j1     : overall min lambda = %.5f at 2j1=%d' % min(integ))

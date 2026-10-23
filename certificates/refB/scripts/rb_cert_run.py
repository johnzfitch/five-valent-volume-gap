"""Exact re-certification (rb_cert) of the claimed bounds: exhaustive at small spins, random samples at large spins.

Categories (doubled spins; legs 2-5 sorted within pairs only):
  Z32-all   : j1 = 3/2, all tuples with legs 2j <= ALLMAX, targets 3.7 ([Z] Thm A) and 7.18 ([Z] 2j<=40 claim)
  Z32-rand  : j1 = 3/2, random legs up to 2j <= 200, target 3.7
  Z32-r40   : j1 = 3/2, random legs with 2j <= 40, target 7.18
  P1-all    : j1 = 1, all tuples with legs 2j <= ALLMAX, target 4.558 ([LS] Thm P)
  P1-rand   : j1 = 1, random legs up to 2j <= 200, target 4.558
  S2-all    : j1 = 2, all tuples with legs 2j <= ALLMAX, target 4.5579607 ([FJ] Thm 4.1)
  S2-W2     : j1 = 2, random W2 tuples with 39 <= 2j_max <= 200, target 4.5579607 ([FJ] Thm 4.2)
  S2-nonW2  : j1 = 2, random non-W2 tuples, legs up to 200, target 7.8 ([DC] Cor. 10 as used by [FJ] Thm 4.1)
  S52-70    : j1 = 5/2, random tuples with 39 <= 2j_max <= 70, target 4.5579607 ([FJ] Thm 5.1)
  C6        : Theorem C class: three legs with 2j <= 6, fourth leg j5 = s - M, s up to 1500,
              target max(4.5579607, 4.922951 s - 1370)
  S72-two   : the two j1 = 7/2 tuples of [FJ] Sec. 6, target 4.5579607
Usage: python rb_cert_run.py ALLMAX NRAND > ../outputs/rb_cert_out.txt
"""
import sys, os, time, random, itertools
from fractions import Fraction as Fr
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_core as C
import rb_cert as RC

Q0 = Fr(45579607, 10 ** 7)


def all_tuples(s2, M):
    for a in range(1, M + 1):
        for b in range(a, M + 1):
            for c in range(1, M + 1):
                for d in range(c, M + 1):
                    if (s2 + a + b + c + d) % 2 == 0:
                        yield (s2, a, b, c, d)


def rand_legs(rng, L, mode):
    if mode == 0:
        return [rng.randint(1, L) for _ in range(4)]
    if mode == 1:                              # fat: all near one value
        base = rng.randint(1, L); return [max(1, base - rng.randint(0, 4)) for _ in range(4)]
    if mode == 2:                              # three small, one large
        big = rng.randint(1, L); return [rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4), big]
    big = rng.randint(1, L)                    # two small two large
    return [rng.randint(1, 4), rng.randint(1, 4), big, max(1, big - rng.randint(0, 6))]


def rand_tuple(rng, s2, L, cond=lambda t: True):
    while True:
        legs = rand_legs(rng, L, rng.randint(0, 3))
        rng.shuffle(legs)
        a, b, c, d = legs
        tj = (s2, min(a, b), max(a, b), min(c, d), max(c, d))
        if sum(tj) % 2 == 0 and C.dim_inv(tj) >= 2 and cond(tj):
            return tj


def w2(tj):
    a = tj[1:]
    return all(x % 2 == 0 for x in a) or all(x % 2 == 1 and x >= 3 for x in a)


def job(arg):
    cat, tj, qs = arg
    st, info = RC.certify(tj, qs)
    return cat, tj, qs, st, info


if __name__ == '__main__':
    ALLMAX = int(sys.argv[1]); NR = int(sys.argv[2])
    rng = random.Random(20261021)
    jobs = []
    for tj in all_tuples(3, ALLMAX):
        jobs.append(('Z32-all/3.7', tj, Fr(37, 10))); jobs.append(('Z32-all/7.18', tj, Fr(718, 100)))
    for tj in all_tuples(2, ALLMAX):
        jobs.append(('P1-all/4.558', tj, Fr(4558, 1000)))
    for tj in all_tuples(4, ALLMAX):
        jobs.append(('S2-all/q0', tj, Q0))
    for _ in range(NR):
        jobs.append(('Z32-rand/3.7', rand_tuple(rng, 3, 200), Fr(37, 10)))
        jobs.append(('Z32-r40/7.18', rand_tuple(rng, 3, 40), Fr(718, 100)))
        jobs.append(('P1-rand/4.558', rand_tuple(rng, 2, 200), Fr(4558, 1000)))
        jobs.append(('S2-W2/q0', rand_tuple(rng, 4, 200, lambda t: w2(t) and max(t[1:]) >= 39), Q0))
        jobs.append(('S2-nonW2/7.8', rand_tuple(rng, 4, 200, lambda t: not w2(t)), Fr(78, 10)))
        jobs.append(('S52-39..70/q0', rand_tuple(rng, 5, 70, lambda t: max(t[1:]) >= 39), Q0))
    # Theorem C class
    for _ in range(NR):
        t = [rng.randint(1, 6) for _ in range(3)]
        tot = sum(t)
        M2 = rng.choice(range(-tot, tot + 1, 2))
        s2 = rng.choice([rng.randint(1, 80), rng.randint(1, 600), rng.randint(1, 3000)])
        j5 = s2 - M2
        if j5 < 1:
            continue
        legs = t + [j5]; rng.shuffle(legs)
        a, b, c, d = legs
        tj = (s2, min(a, b), max(a, b), min(c, d), max(c, d))
        if sum(tj) % 2 or C.dim_inv(tj) < 2:
            continue
        tgt = max(Q0, Fr(4922951, 10 ** 6) * Fr(s2, 2) - 1370)
        jobs.append(('C6/max(q0,4.922951 s-1370)', tj, tgt))
    jobs.append(('S72-two/q0', (7, 35, 36, 37, 39), Q0)); jobs.append(('S72-two/q0', (7, 35, 37, 38, 39), Q0))
    t0 = time.time()
    stats = {}; fails = []
    with Pool(8) as pool:
        for cat, tj, qs, st, info in pool.imap_unordered(job, jobs, chunksize=4):
            d = stats.setdefault(cat, {})
            d[st] = d.get(st, 0) + 1
            if st not in ('certified', 'trivial'):
                fails.append((cat, tj, float(qs), st, info))
    for cat in sorted(stats):
        print(f"{cat:34s} {stats[cat]}")
    print(f"non-certified (excluding 'trivial' = no nonzero spectrum): {len(fails)}")
    for f in fails[:40]:
        print('   ', f)
    print(f"jobs {len(jobs)}, {time.time()-t0:.0f}s on 8 processes")

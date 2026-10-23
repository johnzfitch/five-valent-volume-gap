"""For each orbit representative eps and each leg h (the leg that will carry spin 1/2), list all
relabellings p (p[0] = h) whose sigma (leg p[4] eliminated) has chain form (alpha, beta, beta, 0).
Chain data: X-pair (legs p[1],p[2]) coupled to b, D-pair (legs p[3],p[4]), (|alpha|,|beta|).
Because the orbit is a single S5-orbit, a statement for one eps and every h covers every member.
Also: for every sorted-labelled sigma in each orbit, the chain forms available with leg 1 (the smallest
sorted spin) as the spin-1/2 leg, expressed as partitions of sorted legs {2,3,4,5}."""
import itertools, json, sys
sys.path.insert(0, __file__.rsplit("\\",1)[0].rsplit("/",1)[0])
from orbits import act, sigma
O = json.load(open(sys.argv[1]))
perms = list(itertools.permutations(range(5)))
for o in O["orbits"][1:]:
    rep = tuple(o["sigma"][0]) if False else None
    # representative eps: first eps in the orbit list of witnesses
    eps = tuple(int(x) for x in list(o["witness"].keys())[0].strip("()").split(","))
    print(f"orbit {o['orbit']}  representative eps = {eps}")
    for h in range(5):
        forms = set()
        for p in perms:
            if p[0] != h: continue
            s = sigma(act(eps, p))
            if s[3] == 0 and s[1] == s[2] and s[0] != 0:
                forms.add((abs(s[0]), abs(s[1]), tuple(sorted((p[1], p[2]))), tuple(sorted((p[3], p[4])))))
        print(f"   spin-1/2 on leg {h+1}: " + "; ".join(f"(|a|,|b|)=({a},{b}) Xpair={tuple(x+1 for x in X)} Dpair={tuple(x+1 for x in D)}" for a, b, X, D in sorted(forms)) if forms else f"   spin-1/2 on leg {h+1}: none")
print("\nSorted labelling, spin 1/2 on sorted leg 1: chain forms per sigma (orbit 3)")
o3 = O["orbits"][3]
allc = [tuple(int(x) for x in k.strip("()").split(",")) for k in O["witnesses"]]
by_sigma = {}
for e in allc:
    by_sigma.setdefault(sigma(e), []).append(e)
res = {}
for s in sorted({max(tuple(x), tuple(-y for y in x)) for x in map(tuple, o3["sigma"])}):
    forms = set()
    for e in by_sigma[s]:
        for p in perms:
            if p[0] != 0: continue
            t = sigma(act(e, p))
            if t[3] == 0 and t[1] == t[2] and t[0] != 0:
                forms.add((abs(t[0]), abs(t[1]), tuple(sorted((p[1]+1, p[2]+1))), tuple(sorted((p[3]+1, p[4]+1)))))
    res[s] = sorted(forms)
    print("  ", s, res[s])
json.dump({str(k): v for k, v in res.items()}, open(sys.argv[2], "w"), indent=1)

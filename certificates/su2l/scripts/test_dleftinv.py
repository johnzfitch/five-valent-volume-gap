import numpy as np, sys
from dleftinv import leftinv_bound
from t1core import qmin
from w0core import pairings
S = int(sys.argv[1])
for legs in [(39,40,40,40),(15,16,16,16),(9,10,10,10),(10,10,19,20),(17,17,33,34),(20,24,30,31),(12,30,30,41)]:
    if (S+sum(legs))%2: continue
    q = qmin(S,*legs)[0]
    res=[]
    for (p1,p2) in pairings(*legs):
        lg=(p1[0],p1[1],p2[0],p2[1])
        for dirn in ('up','down'):
            b=leftinv_bound(S,lg,dirn)
            res.append(f"{p1}{p2}{dirn}:{'-' if b is None else f'{b:.2f}'}")
    print(S, legs, f"q={q:.2f}", ' '.join(res))

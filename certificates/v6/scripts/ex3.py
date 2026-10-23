import sys, numpy as np, itertools
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))
from hv_core import Vertex, sigma_vec_BR
from ladder import model
sig6 = sigma_vec_BR((2,0,0,0,0,0,0,0,-2,0), 6)
for n in range(1,9):
    V=Vertex((1,1,n,n,n+1,n+1)); A=V.Qsigma(sig6); sf=np.sort(np.linalg.svd(A,compute_uv=False))
    res=[]
    for sY in itertools.product((1,-1),repeat=2):
        K,S=model(n,sY,ctx=np); sm=np.sort(np.linalg.svd(8*K,compute_uv=False))
        res.append((sY, float(np.max(np.abs(sm-sf))) if len(sm)==len(sf) else ('dim',len(sm),len(sf))))
    print(n, res)

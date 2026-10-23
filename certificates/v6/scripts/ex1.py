import sys, numpy as np, time
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))
from hv_core import Vertex, sigma_vec_BR
sig6 = sigma_vec_BR((2,0,0,0,0,0,0,0,-2,0), 6)
for n in range(1,16):
    t=time.time(); V=Vertex((1,1,n,n,n+1,n+1)); A=V.Qsigma(sig6)
    s=np.sort(np.linalg.svd(A,compute_uv=False))
    nz=s[s>1e-10*s.max()]
    print(n, V.dim, 'kernel',len(s)-len(nz), 'qmin %.6e'%nz[0], 'next %.4e %.4e'%(nz[2],nz[4]) if len(nz)>4 else '', '%.1fs'%(time.time()-t), flush=True)

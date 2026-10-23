import numpy as np, time, sys, json
from rs_core import *
M=int(sys.argv[1]) if len(sys.argv)>1 else 14
t0=time.time(); rows=[]
for J5 in range(1,M+1):
  for J4 in range(1,J5+1):
    for J3 in range(1,J4+1):
      for J2 in range(1,J3+1):
        for J1 in range(1,J2+1):
          if (J1+J2+J3+J4+J5)%2: continue
          Jd=(J1,J2,J3,J4,J5)
          V=Vertex(Jd)
          if V.dim==0: continue
          w=np.linalg.eigvalsh(Q_RS(V)); tol=1e-9*max(1,w.max())
          nz=w[w>tol]; k=int(np.sum(w<=tol))
          rows.append((Jd,V.dim,k,float(nz[0]) if len(nz) else None))
best={}
for Jd,d,k,m in rows:
    if m is None: continue
    jm=Jd[-1]
    if jm not in best or m<best[jm][0]: best[jm]=(m,Jd,d,k)
for jm in sorted(best): print('2jmax',jm,'min nonzero eig of Q_RS %.6f'%best[jm][0],best[jm][1:])
print('kernel dims seen',sorted(set(k for _,_,k,_ in rows)),'tuples',len(rows),'time %.0fs'%(time.time()-t0))
json.dump([(list(a),b,c,d) for a,b,c,d in rows],open(f'../outputs/rs_survey_M{M}.json','w'))

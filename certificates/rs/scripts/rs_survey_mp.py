"""Exhaustive valence-5 RS survey, all sorted tuples with 2j_max <= M, 8 CPU processes.
Records: dim, kernel dim of Q_RS, smallest nonzero eigenvalue of Q_RS, smallest nonzero eigenvalue of sum of range
projectors, and the pair bound for triples {1,3,5},{2,3,5} (sorted legs, 1-based)."""
import numpy as np, sys, json, time
from multiprocessing import Pool
from rs_core import *
def rangeproj(A):
    w,U=np.linalg.eigh(1j*A); tol=1e-9*max(1,np.abs(w).max()); Uk=U[:,np.abs(w)>tol]; return (Uk@Uk.conj().T).real
def work(Jd):
    V=Vertex(Jd)
    if V.dim==0: return None
    Q=qhats(V); A={T:absm(m) for T,m in Q.items()}
    full=sum(A.values()); w=np.linalg.eigvalsh(full); tol=1e-9*max(1,w.max())
    nz=w[w>tol]; k=int(np.sum(w<=tol))
    P=sum(rangeproj(m) for m in Q.values()); wp=np.linalg.eigvalsh(P); nzp=wp[wp>1e-9]
    pair=A[(0,2,4)]+A[(1,2,4)]; wq=np.linalg.eigvalsh(pair); kq=int(np.sum(wq<1e-9*max(1,wq.max()))); nzq=wq[wq>1e-9*max(1,wq.max())]
    return (list(Jd),V.dim,k,float(nz[0]) if len(nz) else None,float(nzp[0]) if len(nzp) else None,kq,float(nzq[0]) if len(nzq) else None)
if __name__=='__main__':
    M=int(sys.argv[1]); t0=time.time()
    tups=[(a,b,c,d,e) for e in range(1,M+1) for d in range(1,e+1) for c in range(1,d+1) for b in range(1,c+1) for a in range(1,b+1) if (a+b+c+d+e)%2==0]
    with Pool(8) as p: rows=[r for r in p.map(work,tups,chunksize=20) if r]
    json.dump(rows,open(f'../outputs/rs_survey_M{M}.json','w'))
    best={}
    for Jd,d,k,m,mp_,kq,mq in rows:
        if m is None: continue
        if Jd[-1] not in best or m<best[Jd[-1]][0]: best[Jd[-1]]=(m,Jd,d,k)
    lines=[f"2jmax={e}: min nonzero eig Q_RS = {best[e][0]:.6f} at {best[e][1]} (dim {best[e][2]}, ker {best[e][3]}); ratio/(jmax+1) = {best[e][0]/(e/2+1):.4f}" for e in sorted(best)]
    lines.append(f"kernel dims {sorted(set(r[2] for r in rows))}; tuples {len(rows)}")
    lines.append(f"min nonzero eig of sum of range projectors: {min(r[4] for r in rows if r[4] is not None):.6f}")
    pk=[r for r in rows if r[5]!=r[2]]
    lines.append(f"pair {{1,3,5}},{{2,3,5}}: kernel equal to ker Q_RS in all but {len(pk)} tuples; min nonzero eig/(jmax+1) = {min(r[6]/(r[0][-1]/2+1) for r in rows if r[6] is not None and r[5]==r[2]):.4f}")
    lines.append(f"wall {time.time()-t0:.0f}s")
    print("\n".join(lines)); open(f'../outputs/rs_survey_M{M}.txt','w').write("\n".join(lines)+"\n")

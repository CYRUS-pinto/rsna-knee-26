"""Label-free check of two arms: per-finding agreement, medial/lateral swap, and swap-by-laterality.
Usage: python cross_arm_check.py armA.csv armB.csv [laterality.csv]
armA/armB: StudyInstanceUID + 12 finding columns, same >=100 studies through BOTH arms.
laterality.csv (optional): StudyInstanceUID,Laterality  with values L or R (from DICOM ImageLaterality)."""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
a=pd.read_csv(sys.argv[1]); b=pd.read_csv(sys.argv[2])
m=a.merge(b,on='StudyInstanceUID',suffixes=('_a','_b'))
n=len(m); print(f"studies in both arms: {n}")
if n<100: print("WARNING: <100 studies; too noisy to act on.")
def rho(d,x,y): return spearmanr(d[x],d[y])[0]
rng=np.random.default_rng(0)
def boot(x,y,k=300):
    v=[spearmanr(m[x].values[i],m[y].values[i])[0] for i in (rng.integers(0,n,n) for _ in range(k))]
    return np.percentile(v,[2.5,97.5])
print(f"\n{'finding':<18}{'rho(A,B)':>9}  95% CI")
for c in L:
    lo,hi=boot(c+'_a',c+'_b'); print(f"{c:<18}{rho(m,c+'_a',c+'_b'):>9.3f}  [{lo:.2f}, {hi:.2f}]")
pairs=[('Medial Meniscus','Lateral Meniscus'),('Medial OA','Lateral OA')]
print("\nGLOBAL SWAP TEST (diag = same finding; swapped = medial<->lateral):")
for mm,ll in pairs:
    d=(rho(m,mm+'_a',mm+'_b')+rho(m,ll+'_a',ll+'_b'))/2
    s=(rho(m,mm+'_a',ll+'_b')+rho(m,ll+'_a',mm+'_b'))/2
    print(f"{mm}/{ll}: diag={d:.3f} swapped={s:.3f}"+("  <-- POSSIBLE GLOBAL SWAP" if s>d+0.10 else ""))
print("\nNOTE: if a swap happens for only SOME studies (e.g. one knee side), the global test can miss it.")
print("Signature then: these 4 columns agree much less than the other 8, with diag AND swapped both low.")
if len(sys.argv)>3:
    lat=pd.read_csv(sys.argv[3]); mm_=m.merge(lat,on='StudyInstanceUID')
    print(f"\nBY LATERALITY (n={len(mm_)}): diag rho per side, medial and lateral columns")
    for side,g in mm_.groupby('Laterality'):
        if len(g)<30: print(f"{side}: n={len(g)} too small"); continue
        out=[f"{c}={rho(g,c+'_a',c+'_b'):.2f}" for p in pairs for c in p]
        sw=[f"{p[0]}~{p[1]}={rho(g,p[0]+'_a',p[1]+'_b'):.2f}" for p in pairs]
        print(f"side {side} (n={len(g)}): "+"  ".join(out)+"  | cross: "+"  ".join(sw))
    print("A side whose diag is low/negative while the other side's is high => laterality-dependent swap.")

"""E4 redone with multiple seeds per alpha.

Single-sample curves are dominated by sampling noise (measured diversity 0.105),
so requiring 5 points to be monotone is close to a coin flip. Average over seeds.
"""
import json, os, sys, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import dist_util
from run_experiments import load, sample, denorm

DATA="./dataset/crisp"; SEEDS=6
ALPHAS=[0.0,0.25,0.5,0.75,1.0,1.25,1.5]

def run(tag, ckpt, res):
    model,diff,ds,a=load(ckpt)
    idx=json.load(open(f"{DATA}/index.json"))
    z=np.load(f"{DATA}/text/clip_vit_l14_pooled.npz",allow_pickle=True)
    allk=list(z["sample_keys"]); emb=z["scenario_emb"]; row={k:i for i,k in enumerate(allk)}
    by=collections.defaultdict(dict)
    for k,v in idx.items():
        if v["speed"]=="slow_long": by[v["behavior"]][v["style"]]=k
    behs=[b for b,d in by.items() if {"subtle","exaggerated"}<=set(d)][:12]
    print(f"\n===== {tag} (lambda_vel={a.lambda_vel}) E4 with {SEEDS} seeds/alpha =====",flush=True)
    curves={}
    for b in behs:
        e_s=emb[row[by[b]["subtle"]]]; e_e=emb[row[by[b]["exaggerated"]]]
        E=torch.from_numpy(np.stack([(1-al)*e_s+al*e_e for al in ALPHAS])).float()
        acc=[]
        for s in range(SEEDS):
            gs=sample(model,diff,E,[90]*len(ALPHAS),seed=s)
            acc.append([float(np.abs((m:=denorm(g,ds))-m[0]).max()) for g in gs])
        curves[b]=dict(mean=np.mean(acc,0).tolist(), std=np.std(acc,0).tolist())
        print("  %-26s "%b + " ".join("%.2f"%v for v in curves[b]["mean"]),flush=True)
    M=np.array([c["mean"] for c in curves.values()])
    i1=ALPHAS.index(1.0)
    mono=sum(all(m[i]<=m[i+1]+0.02 for i in range(i1)) for m in M)
    print("  mean curve over %d behaviours: "%len(M) + " ".join("%.2f"%v for v in M.mean(0)),flush=True)
    print("  alphas:                       " + " ".join("%.2f"%a_ for a_ in ALPHAS),flush=True)
    print("  monotone over [0,1] (seed-averaged): %d/%d"%(mono,len(M)),flush=True)
    print("  exaggerated > subtle: %d/%d   ratio %.2f   (data ratio ~1.70)"%(
        int((M[:,i1]>M[:,0]).sum()),len(M),M[:,i1].mean()/max(M[:,0].mean(),1e-6)),flush=True)
    print("  extrapolation a=1.5: %.2f vs a=1.0 %.2f"%(M[:,-1].mean(),M[:,i1].mean()),flush=True)
    res[tag]=dict(alphas=ALPHAS,curves=curves,monotone=int(mono),n=len(M),seeds=SEEDS,
                  mean_curve=M.mean(0).tolist())
    del model,diff; torch.cuda.empty_cache()

if __name__=="__main__":
    dist_util.setup_dist(0); res={}
    for tag,ck in (("v1","save/crisp_v1/model000060012.pt"),("v2","save/crisp_v2/model000060012.pt")):
        run(tag,ck,res); json.dump(res,open("experiments/e4_seeds.json","w"),indent=1)
    print("\nDONE -> experiments/e4_seeds.json",flush=True)

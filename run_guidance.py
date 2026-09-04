"""Is the amplitude under-production a CFG artifact or the model?

Generated peaks run well below ground truth (0.60 vs 1.65 on a matched caption).
Sweep guidance_param on held-out cells and compare peak against ground truth.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders.crisp.dataset import Crisp
from utils import dist_util
from run_experiments import load, sample, denorm

DATA="./dataset/crisp"
if __name__=="__main__":
    dist_util.setup_dist(0)
    idx=json.load(open(f"{DATA}/index.json"))
    z=np.load(f"{DATA}/text/clip_vit_l14_pooled.npz",allow_pickle=True)
    allk=list(z["sample_keys"]); emb=z["scenario_emb"]; row={k:i for i,k in enumerate(allk)}
    res={}
    for tag,ck in (("v1","save/crisp_v1/model000060012.pt"),("v2","save/crisp_v2/model000060012.pt")):
        model,diff,ds,a=load(ck)
        d=Crisp(split="test_comp",num_frames=201,datapath=DATA); keys=d.keys
        E=torch.from_numpy(np.stack([emb[row[k]] for k in keys])).float()
        L=[idx[k]["n_frames"] for k in keys]
        gtp=np.array([float(np.abs((g:=np.load(f"{DATA}/motions/{k}.npy"))-g[0]).max()) for k in keys])
        print(f"\n=== {tag} (lambda_vel={a.lambda_vel}) test_comp n={len(keys)}  GT mean peak {gtp.mean():.3f} ===",flush=True)
        res[tag]={"gt_mean_peak":float(gtp.mean())}
        for g_ in (1.0,1.5,2.5,4.0,6.0):
            gen=[denorm(m,ds) for m in sample(model,diff,E,L,guidance=g_)]
            gp=np.array([float(np.abs(m-m[0]).max()) for m in gen])
            res[tag][str(g_)]=dict(mean_peak=float(gp.mean()),
                                   peak_err=float(np.abs(gp-gtp).mean()),
                                   ratio=float(gp.mean()/gtp.mean()))
            print(f"  guidance {g_:4.1f}  mean peak {gp.mean():.3f}  ratio to GT {gp.mean()/gtp.mean():.2f}"
                  f"  peak_err {np.abs(gp-gtp).mean():.3f}",flush=True)
        del model,diff; torch.cuda.empty_cache()
    json.dump(res,open("experiments/guidance.json","w"),indent=1)
    print("\nDONE -> experiments/guidance.json",flush=True)

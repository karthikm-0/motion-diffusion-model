"""E2 redone with a SHAPE metric, not just amplitude.

peak_err rewards any motion whose amplitude happens to match, so retrieval -- which
returns a real training motion -- is favoured by construction even when it returns
the wrong gesture. The question is whether the generated motion resembles the
ground-truth *trajectory*, so compare time-normalized shape, and which joints move.
"""
import json, os, sys, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders.crisp.dataset import Crisp
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from utils import dist_util
from run_experiments import load, sample, denorm, norm64, reldiff, stats

DATA="./dataset/crisp"; OUT="./experiments"; os.makedirs(OUT,exist_ok=True)

def moved(m, thr=0.05):
    return {i for i in range(m.shape[1]) if (np.abs(m[:,i]-m[0,i]).max() > thr)}

def jacc(a,b):
    return 1.0 if not (a|b) else len(a&b)/len(a|b)

def cos_shape(A,B):
    """cosine between time-normalized excursion trajectories -- 1.0 = same shape."""
    a=(norm64(A)-norm64(A)[0]).ravel(); b=(norm64(B)-norm64(B)[0]).ravel()
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return float(a@b/(na*nb)) if na>1e-9 and nb>1e-9 else 0.0

def run(tag, ckpt, res):
    model,diff,ds,a = load(ckpt)
    idx=json.load(open(f"{DATA}/index.json"))
    z=np.load(f"{DATA}/text/clip_vit_l14_pooled.npz",allow_pickle=True)
    allk=list(z["sample_keys"]); emb=z["scenario_emb"]; row={k:i for i,k in enumerate(allk)}
    keys=[]; split_of={}
    for sp in ("val","test_comp","test_behav"):
        d=Crisp(split=sp,num_frames=201,datapath=DATA)
        for k in d.keys: split_of[k]=sp
        keys+=d.keys
    E=torch.from_numpy(np.stack([emb[row[k]] for k in keys])).float()
    L=[idx[k]["n_frames"] for k in keys]
    gen=[denorm(m,ds) for m in sample(model,diff,E,L)]
    gt =[np.load(f"{DATA}/motions/{k}.npy") for k in keys]
    trk=ds.keys; TE=np.stack([emb[row[k]] for k in trk]); TEn=TE/np.linalg.norm(TE,axis=1,keepdims=True)
    ret=[]
    for k in keys:
        q=emb[row[k]]; q=q/np.linalg.norm(q)
        ret.append(np.load(f"{DATA}/motions/{trk[int((TEn@q).argmax())]}.npy"))

    def sc(pred, sf=None):
        rd,cs,jc,pk=[],[],[],[]
        for k,p,g in zip(keys,pred,gt):
            if sf and split_of[k]!=sf: continue
            rd.append(reldiff(p,g)); cs.append(cos_shape(p,g))
            jc.append(jacc(moved(p),moved(g))); pk.append(abs(stats(p)["peak"]-stats(g)["peak"]))
        return dict(n=len(rd), shape_cos=float(np.mean(cs)), reldiff=float(np.median(rd)),
                    moved_jaccard=float(np.mean(jc)), peak_err=float(np.mean(pk)))
    out={}
    print(f"\n===== {tag} (lambda_vel={a.lambda_vel}) — SHAPE metrics, higher shape_cos/jaccard is better =====",flush=True)
    print(f"  {'split':12s} {'n':>4s} | {'MDM shape_cos':>13s} {'reldiff':>8s} {'jacc':>6s} | {'RET shape_cos':>13s} {'reldiff':>8s} {'jacc':>6s} | winner",flush=True)
    for sf in (None,"test_comp","test_behav","val"):
        nm=sf or "all_heldout"; m=sc(gen,sf); r=sc(ret,sf); out[nm]={"mdm":m,"retrieval":r}
        win="MDM" if m["shape_cos"]>r["shape_cos"] else "retrieval"
        print(f"  {nm:12s} {m['n']:4d} | {m['shape_cos']:13.3f} {m['reldiff']:8.2f} {m['moved_jaccard']:6.2f}"
              f" | {r['shape_cos']:13.3f} {r['reldiff']:8.2f} {r['moved_jaccard']:6.2f} | {win}",flush=True)
    res[tag]=dict(checkpoint=ckpt,lambda_vel=a.lambda_vel,**out)
    del model,diff; torch.cuda.empty_cache()

if __name__=="__main__":
    dist_util.setup_dist(0); res={}
    for tag,ck in (("v1","save/crisp_v1/model000060012.pt"),("v2","save/crisp_v2/model000060012.pt")):
        run(tag,ck,res); json.dump(res,open(f"{OUT}/e2_shape.json","w"),indent=1)
    print("\nDONE -> experiments/e2_shape.json",flush=True)

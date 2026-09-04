"""Run E0/E1/E2/E4/E5/E7 and write results to experiments/. Metrics only, no rendering."""
import json, os, sys, itertools, collections
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders.crisp.dataset import Crisp
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from utils import dist_util

DATA = "./dataset/crisp"
OUT  = "./experiments"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"

def log(*a):
    print(*a, flush=True)

def stats(m):
    return dict(frames=int(len(m)), peak=float(np.abs(m-m[0]).max()),
                path=float(np.abs(np.diff(m,axis=0)).sum()))

def norm64(M, n=64):
    t = np.linspace(0, len(M)-1, n)
    return np.stack([np.interp(t, np.arange(len(M)), M[:,c]) for c in range(M.shape[1])], 1)

def reldiff(A, B):
    A, B = norm64(A), norm64(B)
    eA, eB = np.abs(A-A[0]).mean(), np.abs(B-B[0]).mean()
    return float(np.abs(A-B).mean() / max(1e-6, (eA+eB)/2))

class _A:
    def __init__(s, d): s.__dict__.update(d)

def load(ckpt):
    a = _A(json.load(open(os.path.join(os.path.dirname(ckpt), "args.json"))))
    ds = Crisp(split="train", num_frames=a.num_frames, datapath=DATA)
    class D: dataset = ds
    model, diff = create_model_and_diffusion(a, D())
    load_saved_model(model, ckpt)
    model.to(DEV); model.eval()
    return model, diff, ds, a

CANVAS = 201

def sample(model, diff, embeds, lengths, guidance=2.5, seed=0, batch=32):
    """embeds: (N, dim) float tensor. returns list of (T_i, J) raw-radian arrays."""
    torch.manual_seed(seed); np.random.seed(seed)
    wrapped = ClassifierFreeSampleModel(model) if guidance != 1.0 else model
    J = model.njoints
    out = []
    for lo in range(0, len(embeds), batch):
        hi = min(lo+batch, len(embeds)); nb = hi-lo
        # Always generate on the 201-frame canvas the model was trained with; the
        # mask carries the real length. A shorter canvas is off-distribution and
        # makes output amplitude swing erratically (0.03 -> 1.13 for one caption).
        L = [min(x, CANVAS) for x in lengths[lo:hi]]; T = CANVAS
        cond = {"y": {
            "text_embed": embeds[lo:hi].to(DEV).unsqueeze(0),
            "lengths": torch.tensor(L, device=DEV),
            "mask": (torch.arange(T, device=DEV)[None,:].expand(nb,T)
                     < torch.tensor(L, device=DEV)[:,None])[:,None,None,:],
        }}
        if guidance != 1.0:
            cond["y"]["scale"] = torch.full((nb,), guidance, device=DEV)
        with torch.no_grad():
            s = diff.p_sample_loop(wrapped, (nb, J, 1, T), clip_denoised=False,
                                   model_kwargs=cond, progress=False)
        arr = s.squeeze(2).permute(0,2,1).cpu().numpy()
        for i in range(nb):
            out.append(arr[i, :L[i]])
    return out

def denorm(m, ds): return m * ds.std + ds.mean

# ---------------------------------------------------------------- experiments
RESULTS = {}

def held_out_keys(ds_train):
    keys, splits = [], {}
    for sp in ("val", "test_comp", "test_behav"):
        d = Crisp(split=sp, num_frames=201, datapath=DATA)
        for k in d.keys: splits[k] = sp
        keys += d.keys
    return keys, splits

def E0_E1_E2(tag, ckpt):
    log(f"\n===== {tag}: E0 (smoothness) + E1 (held-out cells) + E2 (retrieval) =====")
    model, diff, ds_tr, a = load(ckpt)
    idx = json.load(open(os.path.join(DATA, "index.json")))
    cap = json.load(open(os.path.join(DATA, "captions.json")))
    z = np.load(os.path.join(DATA, "text", "clip_vit_l14_pooled.npz"), allow_pickle=True)
    allkeys = list(z["sample_keys"]); emb = z["scenario_emb"]
    row = {k:i for i,k in enumerate(allkeys)}

    keys, split_of = held_out_keys(ds_tr)
    E = torch.from_numpy(np.stack([emb[row[k]] for k in keys])).float()
    L = [idx[k]["n_frames"] for k in keys]
    gen = [denorm(m, ds_tr) for m in sample(model, diff, E, L)]
    gt  = [np.load(os.path.join(DATA, "motions", k+".npy")) for k in keys]

    # ---- E2 retrieval baseline: nearest TRAIN caption, return its motion
    tr_keys = ds_tr.keys
    TE = np.stack([emb[row[k]] for k in tr_keys])
    TEn = TE/np.linalg.norm(TE,axis=1,keepdims=True)
    ret = []
    for k in keys:
        q = emb[row[k]]; q = q/np.linalg.norm(q)
        ret.append(np.load(os.path.join(DATA, "motions", tr_keys[int((TEn@q).argmax())]+".npy")))

    def score(pred, split_filter=None):
        pe, pr, du = [], [], []
        for k, p, g in zip(keys, pred, gt):
            if split_filter and split_of[k] != split_filter: continue
            sp_, sg = stats(p), stats(g)
            pe.append(abs(sp_["peak"]-sg["peak"]))
            pr.append(sp_["path"]/max(sg["path"],1e-6))
            du.append(abs(sp_["frames"]-sg["frames"]))
        return dict(n=len(pe), peak_err=float(np.mean(pe)),
                    path_ratio_median=float(np.median(pr)), dur_err=float(np.mean(du)))

    out = {"checkpoint": ckpt, "lambda_vel": a.lambda_vel}
    for sf in (None, "test_comp", "test_behav", "val"):
        name = sf or "all_heldout"
        out[name] = {"mdm": score(gen, sf), "retrieval": score(ret, sf)}
        m, r = out[name]["mdm"], out[name]["retrieval"]
        log(f"  {name:12s} n={m['n']:3d} | MDM peak_err {m['peak_err']:.3f} path_ratio {m['path_ratio_median']:.2f}"
            f" | RETRIEVAL peak_err {r['peak_err']:.3f} path_ratio {r['path_ratio_median']:.2f}"
            f" | MDM better: {m['peak_err']<r['peak_err']}")
    RESULTS[tag] = out
    return model, diff, ds_tr, emb, row, idx, cap

def E4_style(tag, model, diff, ds, emb, row, idx, cap):
    log(f"\n===== {tag}: E4 style interpolation =====")
    by = collections.defaultdict(dict)
    for k,v in idx.items():
        if v["speed"]=="slow_long": by[v["behavior"]][v["style"]] = cap[k]["scenario"]
    behs = [b for b,d in by.items() if {"subtle","exaggerated"} <= set(d)][:12]
    alphas = [0.0,0.25,0.5,0.75,1.0,1.25,1.5]
    curves = {}
    for b in behs:
        e_s = emb[row[[k for k,v in idx.items() if v["behavior"]==b and v["style"]=="subtle" and v["speed"]=="slow_long"][0]]]
        e_e = emb[row[[k for k,v in idx.items() if v["behavior"]==b and v["style"]=="exaggerated" and v["speed"]=="slow_long"][0]]]
        E = torch.from_numpy(np.stack([(1-al)*e_s + al*e_e for al in alphas])).float()
        gs = sample(model, diff, E, [90]*len(alphas))   # ~4.5s, typical slow_long
        curves[b] = [float(np.abs((m:=denorm(g,ds))-m[0]).max()) for g in gs]
        log(f"  {b:28s} " + " ".join(f"{a:.2f}:{p:.2f}" for a,p in zip(alphas,curves[b])))
    mono = sum(all(c[i]<=c[i+1]+0.02 for i in range(4)) for c in curves.values())
    log(f"  monotone over alpha in [0,1]: {mono}/{len(curves)}")
    RESULTS.setdefault(tag,{})["E4"] = dict(alphas=alphas, curves=curves, monotone=mono, n=len(curves))

def E5_blend(tag, model, diff, ds, emb, row, idx, cap):
    log(f"\n===== {tag}: E5 behavior blending =====")
    pick = lambda b: [k for k,v in idx.items() if v["behavior"]==b and v["style"]=="neutral" and v["speed"]=="slow_long"]
    pairs = [("listening_nod","small_right_wave"), ("point_left","celebration_arm_raise"),
             ("polite_bow","beckon_right"), ("stop_palm","three_nods_enthusiasm")]
    res = {}
    for a_,b_ in pairs:
        ka, kb = pick(a_), pick(b_)
        if not ka or not kb: continue
        ea, eb = emb[row[ka[0]]], emb[row[kb[0]]]
        E = torch.from_numpy(np.stack([ea,(ea+eb)/2,eb])).float()
        gs = [denorm(g,ds) for g in sample(model, diff, E, [90]*3)]
        # which endpoint does the blend resemble?
        d_a, d_b = reldiff(gs[1],gs[0]), reldiff(gs[1],gs[2])
        res[f"{a_}+{b_}"] = dict(peak=[float(np.abs(g-g[0]).max()) for g in gs],
                                 reldiff_to_a=d_a, reldiff_to_b=d_b)
        log(f"  {a_:24s}+{b_:24s} peaks {[round(x,2) for x in res[f'{a_}+{b_}']['peak']]}"
            f"  blend->A {d_a:.2f}  blend->B {d_b:.2f}")
    RESULTS.setdefault(tag,{})["E5"] = res

def E7_diversity(tag, model, diff, ds, emb, row, idx):
    log(f"\n===== {tag}: E7 sample diversity (data reference: median 0.60) =====")
    cells = collections.defaultdict(list)
    for k,v in idx.items():
        if v["split"]=="test_comp": cells[v["cell"]].append(k)
    out = {}
    for cell, ks in list(cells.items())[:10]:
        e = emb[row[ks[0]]]
        E = torch.from_numpy(np.stack([e]*8)).float()
        gs = [denorm(g,ds) for g in sample(model, diff, E, [idx[ks[0]]["n_frames"]]*8, seed=1)]
        ds_ = [reldiff(gs[i],gs[j]) for i,j in itertools.combinations(range(8),2)]
        out[cell] = float(np.median(ds_))
    med = float(np.median(list(out.values())))
    log(f"  median pairwise relative difference across {len(out)} cells: {med:.3f}  (data: 0.60)")
    RESULTS.setdefault(tag,{})["E7"] = dict(per_cell=out, median=med, data_reference=0.60)

if __name__ == "__main__":
    dist_util.setup_dist(0)
    for tag, ckpt in (("v1","save/crisp_v1/model000060012.pt"),
                      ("v2","save/crisp_v2/model000060012.pt")):
        model, diff, ds, emb, row, idx, cap = E0_E1_E2(tag, ckpt)
        E4_style(tag, model, diff, ds, emb, row, idx, cap)
        E5_blend(tag, model, diff, ds, emb, row, idx, cap)
        E7_diversity(tag, model, diff, ds, emb, row, idx)
        json.dump(RESULTS, open(os.path.join(OUT,"results.json"),"w"), indent=1)
        del model, diff; torch.cuda.empty_cache()
    json.dump(RESULTS, open(os.path.join(OUT,"results.json"),"w"), indent=1)
    log("\nDONE -> experiments/results.json")

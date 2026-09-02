"""Sample motions from a crisp-trained MDM checkpoint.

MDM's stock sample/generate.py is SMPL-only: it calls rot2xyz and the HumanML3D
stick-figure plotter. This writes plain (T, 10) joint-angle arrays instead, which
crisp_vla/src/crisp_vla/playback_crisp_motion.py renders.

  # prompts straight from a split (paired with their ground truth for comparison)
  ./crisp sample --model save/crisp_v1/model000010000.pt --split test_comp

  # your own prompt
  ./crisp sample --model save/crisp_v1/model000010000.pt \
      --text "The robot waves. The robot performs the response with exaggerated expressiveness and quickly with a short motion."

Outputs <out>/<name>.npy (raw radians, denormalized) plus results.json holding the
prompt, length and source key for each sample.
"""
import argparse, json, os, sys

import numpy as np
import torch

_MDM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _MDM)
DEFAULT_DATA = os.path.join(_MDM, "dataset", "crisp")

from data_loaders.crisp.dataset import Crisp
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import get_cond_mode
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel


def embed_texts(texts, data):
    """Embed novel prompts with whatever encoder built the cached .npz."""
    import glob
    caches = sorted(glob.glob(os.path.join(data, "text", "*.npz")))
    if not caches:
        sys.exit("no text cache in %s/text -- run prepare_crisp_text.py" % data)
    enc = str(np.load(caches[0], allow_pickle=True)["encoder"])
    if enc == "clip_vit_l14_pooled":
        sys.path.insert(0, os.path.join(os.path.dirname(_MDM), "crisp_vla", "src", "crisp_vla"))
        from prepare_crisp_text import encode_clip_local
        return encode_clip_local(list(texts))[0]
    from sentence_transformers import SentenceTransformer
    return np.asarray(SentenceTransformer(enc.replace("_", "/", 1)).encode(list(texts)), dtype=np.float32)


class _Args:
    """Rebuild the training args from the checkpoint's args.json."""
    def __init__(self, d):
        self.__dict__.update(d)


def load_train_args(model_path):
    d = os.path.dirname(os.path.abspath(model_path))
    p = os.path.join(d, "args.json")
    if not os.path.exists(p):
        sys.exit("no args.json beside %s -- cannot reconstruct the model config" % model_path)
    return _Args(json.load(open(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to model<step>.pt")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="", help="draw prompts from this split")
    ap.add_argument("--text", default="", help="a single prompt (overrides --split)")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--num_repetitions", type=int, default=1,
                    help="samples per prompt -- text->motion is one-to-many, so >1 shows the spread")
    ap.add_argument("--length", type=int, default=0, help="frames; 0 = use the GT length per prompt")
    ap.add_argument("--guidance_param", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    targs = load_train_args(args.model)
    targs.device = args.device
    dist_util.setup_dist(args.device if args.device != "cpu" else -1)

    ds = Crisp(split=(args.split or "train"), num_frames=targs.num_frames, datapath=args.data)
    mean, std = ds.mean, ds.std

    # prompts: either a novel string (embedded now, with the same encoder the cache
    # used) or the captions of the chosen split, paired with their ground truth
    if args.text:
        keys = ["custom"]
        prompts = [args.text]
        embeds = torch.from_numpy(embed_texts(prompts, args.data)).float()
        lengths = [args.length or 80]
    else:
        keys = ds.keys[:args.num_samples]
        prompts = [ds.captions[k]["scenario"] for k in keys]
        embeds = torch.from_numpy(ds.text_emb[:args.num_samples]).float()
        lengths = [args.length or ds.index[k]["n_frames"] for k in keys]

    class _D:  # create_model_and_diffusion only needs .dataset for text_dim/num_actions
        dataset = ds
    model, diffusion = create_model_and_diffusion(targs, _D())
    load_saved_model(model, args.model, use_avg=True)
    if args.guidance_param != 1.0:
        model = ClassifierFreeSampleModel(model)
    model.to(args.device).eval()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.model)),
        "samples_%s" % os.path.basename(args.model).replace(".pt", ""))
    os.makedirs(out, exist_ok=True)

    n = len(keys)
    results = []
    for rep in range(args.num_repetitions):
        maxlen = max(lengths)
        cond = {"y": {
            "text_embed": embeds.to(args.device).unsqueeze(0),
            "lengths": torch.tensor(lengths, device=args.device),
            "mask": torch.arange(maxlen, device=args.device)[None, :].expand(n, maxlen)
                    < torch.tensor(lengths, device=args.device)[:, None],
        }}
        cond["y"]["mask"] = cond["y"]["mask"][:, None, None, :]
        if args.guidance_param != 1.0:
            cond["y"]["scale"] = torch.full((n,), args.guidance_param, device=args.device)

        shape = (n, model.njoints if not hasattr(model, "model") else model.model.njoints, 1, maxlen)
        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model, shape, clip_denoised=False, model_kwargs=cond,
                skip_timesteps=0, init_image=None, progress=True,
                dump_steps=None, noise=None, const_noise=False)

        arr = sample.squeeze(2).permute(0, 2, 1).cpu().numpy()   # [n, T, J]
        for i, k in enumerate(keys):
            m = arr[i, :lengths[i]] * std + mean                 # denormalize -> radians
            name = "%s__rep%d" % (k, rep) if args.num_repetitions > 1 else k
            np.save(os.path.join(out, name + ".npy"), m.astype(np.float32))
            results.append(dict(name=name, source_key=k, rep=rep, prompt=prompts[i],
                                n_frames=int(lengths[i]),
                                peak_excursion=float(np.abs(m - m[0]).max()),
                                path_len=float(np.abs(np.diff(m, axis=0)).sum())))

    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
    print("\nwrote %d samples -> %s" % (len(results), os.path.abspath(out)))
    for r in results[:6]:
        print("  %-52s %3df  peak %.2f  path %5.2f" %
              (r["name"][:52], r["n_frames"], r["peak_excursion"], r["path_len"]))


if __name__ == "__main__":
    main()

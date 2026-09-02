"""Loader for the crisp_vla Booster T1 expressive-behavior dataset.

The cache is produced by crisp_vla/src/crisp_vla/prepare_crisp_dataset.py and
prepare_crisp_text.py. Layout:

    <datapath>/motions/<key>.npy        (T, 10) float32, RAW radians
    <datapath>/{Mean,Std}.npy           (10,)   train-split statistics
    <datapath>/captions.json            {key: {"scenario": str, "plan": str}}
    <datapath>/index.json               per-sample cell/behavior/style/speed/split
    <datapath>/meta.json                joint layout + the 13 constant joints
    <datapath>/text/<tag>.npz           precomputed caption embeddings
    <datapath>/{train,val,test_comp,test_behav}.txt

Representation: 10 upper-body joint angles at 20 fps. The other 13 joints of the
23-DoF robot are constant across the whole dataset (waist + both legs), so they
are dropped here and re-inserted at decode time from meta.json -- keeping them
would put zero-variance channels through the normalization.

Captions are the `scenario` field only. `plan` is cached alongside for the
intermediate-plan ablation and is deliberately NOT mixed into training text:
it is a different register (it names effectors and joints) and would not be
available at inference.
"""
import json
import os

import numpy as np
import torch
from torch.utils import data

from data_loaders.tensors import collate as base_collate


class Crisp(data.Dataset):
    def __init__(self, split="train", num_frames=None, datapath="./dataset/crisp",
                 text_tag="clip_vit_l14_pooled", caption_field="scenario",
                 fixed_len=0, device=None, **kwargs):
        self.datapath = datapath
        self.split = split
        self.dataname = "crisp"
        self.caption_field = caption_field
        self.fixed_len = int(fixed_len or 0)

        if not os.path.isdir(datapath):
            raise FileNotFoundError(
                "%s not found. Build it first:\n"
                "  python3 ../crisp_vla/src/crisp_vla/prepare_crisp_dataset.py\n"
                "  python3 ../crisp_vla/src/crisp_vla/prepare_crisp_text.py" % datapath)

        self.meta = json.load(open(os.path.join(datapath, "meta.json")))
        self.captions = json.load(open(os.path.join(datapath, "captions.json")))
        self.index = json.load(open(os.path.join(datapath, "index.json")))
        self.mean = np.load(os.path.join(datapath, "Mean.npy")).astype(np.float32)
        self.std = np.load(os.path.join(datapath, "Std.npy")).astype(np.float32)

        split_file = os.path.join(datapath, "%s.txt" % split)
        if not os.path.exists(split_file):
            raise FileNotFoundError("no split file %s" % split_file)
        self.keys = [l.strip() for l in open(split_file) if l.strip()]

        # precomputed caption embeddings -- the text encoder never runs in the loop
        emb_path = os.path.join(datapath, "text", "%s.npz" % text_tag)
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                "%s not found. Run prepare_crisp_text.py --encoder ..." % emb_path)
        z = np.load(emb_path, allow_pickle=True)
        row = {k: i for i, k in enumerate(list(z["sample_keys"]))}
        field = "%s_emb" % caption_field
        if field not in z:
            raise KeyError("%s has no %r (available: %s)" % (emb_path, field, list(z.files)))
        self.text_dim = int(z[field].shape[1])
        self.text_emb = np.stack([z[field][row[k]] for k in self.keys]).astype(np.float32)

        self.max_motion_length = int(num_frames) if num_frames else max(
            self.index[k]["n_frames"] for k in self.keys)

        print("[crisp] split=%-10s %4d samples | %3d cells | %2d behaviors | "
              "njoints=%d | text=%s(%d)"
              % (split, len(self.keys),
                 len({self.index[k]["cell"] for k in self.keys}),
                 len({self.index[k]["behavior"] for k in self.keys}),
                 self.meta["njoints"], text_tag, self.text_dim))

        self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None] if device else None
        self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None] if device else None

    def __len__(self):
        return len(self.keys)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, item):
        key = self.keys[item]
        motion = np.load(os.path.join(self.datapath, "motions", "%s.npy" % key))
        motion = (motion - self.mean) / self.std

        # a fixed_len run (prefix/autoregressive) needs every clip the same length:
        # random-crop the long ones, zero-pad the short ones
        target = self.fixed_len or self.max_motion_length
        n = len(motion)
        if n > target:
            start = np.random.randint(0, n - target + 1)
            motion = motion[start:start + target]
            n = target
        elif n < target:
            motion = np.concatenate(
                [motion, np.zeros((target - n, motion.shape[1]), dtype=motion.dtype)], axis=0)

        return dict(
            inp=torch.from_numpy(motion.T).float().unsqueeze(1),   # [J, 1, seqlen]
            text=self.captions[key][self.caption_field],
            text_embed=torch.from_numpy(self.text_emb[item]).float(),
            lengths=int(n),
            key=key,
        )


def crisp_collate(batch):
    """base collate + the precomputed text embedding.

    mdm.MDM.forward consumes cond['y']['text_embed'] directly (its existing
    caching hook), which must be [1, bs, dim] to broadcast against the timestep
    embedding.
    """
    notnone = [b for b in batch if b is not None]
    motion, cond = base_collate(notnone)
    cond["y"]["text_embed"] = torch.stack([b["text_embed"] for b in notnone], dim=0).unsqueeze(0)
    return motion, cond

# CRISP — MDM on the Booster T1 expressive-behavior dataset

This fork teaches [MDM](https://github.com/GuyTevet/motion-diffusion-model) to
generate expressive robot behaviors from text. Instead of human motion capture, it
trains on a dataset of LLM-generated behaviors for the Booster T1 humanoid.

That dataset comes from a separate pipeline, `crisp_vla`: a scenario prompt is
turned into a behavior plan, compiled to joint angles, rendered in MuJoCo, and
scored by a VLM critic, retrying until the critic accepts it. This repo trains a
diffusion model on the accepted output.

Branch `crisp`, based on upstream `ef8edce`.

---

## Where things stand

Read this before running anything.

**The main result is negative.** On held-out behavior/style/length combinations, a
nearest-neighbour lookup — return the training motion whose caption embedding is
closest — beats the trained model on amplitude, trajectory shape, and which joints
move, in **7 of 8** comparisons.

**One capability survives, and retrieval cannot match it:** continuous style
control. Interpolating the caption embedding between `subtle` and `exaggerated`
gives a monotone amplitude curve whose range (1.82×) matches the dataset's own
(1.70×). A lookup table has nothing to return halfway between two captions.

**The main weakness is mode collapse.** Samples of the same caption differ by a
median of 0.105–0.172, against 0.60 between real variants of the same cell.

| document | what it holds |
|---|---|
| `CRISP.md` | this file — setup, the dataset, and every change to upstream |
| `FINDINGS.md` | measured results for v1 and v2, and what to fix next |
| `EXPERIMENTS.md` | the experiment plan (E0–E10) and what each result would license |

### Training runs

All runs: 60k steps, batch 64, lr 1e-4, 201-frame canvas, precomputed CLIP text
embeddings. Checkpoints are in `save/crisp_vN/`.

| run | model (latent × layers) | λ_vel | data | question | results |
|---|---|---|---|---|---|
| v1 | 256 × 6 | 0 | plain | baseline | `FINDINGS.md` |
| v2 | 256 × 6 | 1.0 | plain | does velocity loss help? | `FINDINGS.md` |
| v3 | 256 × 6 | 0 | mirrored | does left/right augmentation help? | not recorded |
| v4 | 128 × 4 | 0 | plain | does a smaller model collapse less? | not recorded |
| v5 | 128 × 4 | 0 | mirrored | smaller model + augmentation | not recorded |
| v6 | 96 × 3 | 0 | mirrored | size sweep | not recorded |
| v7 | 64 × 2 | 0 | mirrored | size sweep | not recorded |

v3–v7 follow `FINDINGS.md`'s recommendations for attacking mode collapse. They are
trained, but no evaluation of them is committed yet.

---

## Setting up from a fresh clone

### 1. Get both repos and the data, side by side

This repo is one of two, and the code assumes they sit next to each other under a
shared folder along with the source data:

```
expressive_motion_gen/                   any name works
├── motion-diffusion-model/              this repo           branch crisp
├── crisp_vla/                           the data pipeline   branch mdm
├── final-ds/                            the source dataset (~1.8 GB)
├── renders/                             created on first render
└── crisp                                optional dispatcher script, see below
```

```bash
mkdir expressive_motion_gen && cd expressive_motion_gen
git clone -b crisp git@github.com:karthikm-0/motion-diffusion-model.git
git clone -b mdm   git@github.com:simran-ra/crisp_vla.git
```

`final-ds/` is not in either repo. Get `final-ds.zip` (1.6 GB) from whoever handed
this off and unzip it into the shared folder. Every script finds `final-ds/`,
`crisp_vla/` and `motion-diffusion-model/dataset/` relative to that folder, so the
layout above is what makes the default paths work without flags.

`crisp_vla` is a separate GitHub repo; you need read access to it.

### 2. Build the environment

```bash
cd motion-diffusion-model
uv sync
```

This one environment runs everything — training here, and the dataset and
rendering scripts in `crisp_vla`.

### 3. The `./crisp` dispatcher

`crisp` is a short shell script that pairs the `crisp_vla` scripts with this repo's
environment and sets `MUJOCO_GL`. **It is not tracked in either repo** — it lives
in the shared folder. Ask for a copy, or skip it and call the scripts directly:

| `./crisp …` | runs |
|---|---|
| `dataset` | `crisp_vla/src/crisp_vla/prepare_crisp_dataset.py` |
| `text` | `crisp_vla/src/crisp_vla/prepare_crisp_text.py` |
| `train` | `python -m train.train_mdm --dataset crisp --text_encoder_type precomputed`, from `motion-diffusion-model/` |
| `sample` | `motion-diffusion-model/sample/generate_crisp.py` |
| `play` | `crisp_vla/src/crisp_vla/playback_crisp_motion.py` |
| `compare` | `crisp_vla/src/crisp_vla/compare_crisp_motion.py` |
| `gallery` | `crisp_vla/src/crisp_vla/build_gallery.py` |
| `serve` | `python -m http.server` in `renders/`, on the first free port from 8000 |

Without the dispatcher, use the environment's Python from the shared folder, and
set `MUJOCO_GL=osmesa` for anything that renders:

```bash
export MUJOCO_GL=osmesa
motion-diffusion-model/.venv/bin/python crisp_vla/src/crisp_vla/prepare_crisp_dataset.py
```

### 4. Build the dataset and train

```bash
./crisp dataset                      # final-ds/ -> motion-diffusion-model/dataset/crisp/
./crisp text --encoder clip_local    # caption embeddings -> dataset/crisp/text/

./crisp train --save_dir save/my_run --num_frames 201 \
              --latent_dim 256 --layers 6 --batch_size 64 \
              --num_steps 60000 --save_interval 10000
```

`dataset/crisp/` is gitignored, so this step is required on every fresh clone.
Held-out behaviors are chosen automatically; the script prints a nearest-neighbour
audit of the holdout. Read it before quoting any `test_behav` number.

### 5. Sample and look at the results

```bash
./crisp sample  --model save/my_run/model000060000.pt --split test_comp
./crisp compare --samples save/my_run/samples_model000060000
./crisp serve                        # browse renders/ in a browser
```

---

## The dataset

| | |
|---|---|
| source | `final-ds/` — 1080 variants, of which the 1019 the critic accepted are used |
| structure | 60 behaviors × 3 styles (subtle/neutral/exaggerated) × 2 lengths (fast_short/slow_long) = 360 cells, 357 populated, ~3 variants each |
| representation | **10 joint angles** in radians, 20 fps, 17–201 frames |
| conditioning | the `scenario` caption — 357 unique strings |

The T1 has 23 joints, but **13 never move anywhere in the dataset** (waist and both
legs — the generator never actuates them). They are dropped before training, since
zero-variance channels break normalization, and re-inserted at playback from
`meta.json["constant_joints"]`.

Source trajectories store **keyframes**, not frames. The converter fills in the
frames with the same min-jerk / smoothstep / linear interpolation that
`crisp_vla/physics_playback.py` uses, so training data matches what the critic saw.
Verified exact (0.00e+00 error) at keyframes landing on the 20 fps grid.

### Splits

Splits are by **cell**, never by variant. A cell's ~3 variants share an identical
caption, so splitting them would put the exact test prompt in the training set.

```
train        822 variants  289 cells
val           35 variants   12 cells
test_comp     59 variants   20 cells   behavior × style × length never seen together
test_behav   103 variants   36 cells   6 behaviors held out entirely
```

`Mean.npy` / `Std.npy` are computed on **train only**.

### Mirrored variant: `dataset/crisp_mirror/`

```bash
./crisp dataset --mirror --out motion-diffusion-model/dataset/crisp_mirror
./crisp text    --data motion-diffusion-model/dataset/crisp_mirror --encoder clip_local
```

Adds a left/right-mirrored copy of every motion, doubling the dataset to 2038
samples (train 1644, val 70, test_comp 118, test_behav 206). Mirroring swaps
`Left_`/`Right_` joints and negates roll and yaw joints while pitch joints keep
their sign — checked against the joint limits in the robot's URDF. Captions are
mirrored too: "left" and "right" swap, so the text still matches the motion.

It exists because the data is badly imbalanced by side: only 6 of 60 behaviors
move the left arm. Mirrored copies stay in their source's split, so no test motion
leaks into training.

**It is training-time augmentation, not new data.** Evaluate models on
`dataset/crisp/` regardless of which dataset they trained on, or mirrored copies
double-count every held-out sample.

Select a dataset at training time with `--data_dir`:

```bash
./crisp train --data_dir ./dataset/crisp_mirror --save_dir save/my_run ...
```

It defaults to `./dataset/crisp`.

---

## Repository map

What this fork adds on top of upstream MDM:

```
data_loaders/crisp/dataset.py     Crisp dataset + crisp_collate
sample/generate_crisp.py          sampling (MDM's generate.py is SMPL-only)
pyproject.toml, uv.lock           uv environment; environment.yml is an old conda export

run_experiments.py                E0/E1/E2/E4/E5/E7 metrics  -> experiments/results.json
run_e2_shape.py                   retrieval baseline, shape metric -> experiments/e2_shape.json
run_e4_seeds.py                   style interpolation, multi-seed  -> experiments/e4_seeds.json
run_guidance.py                   classifier-free guidance sweep   -> experiments/guidance.json

CRISP.md, FINDINGS.md, EXPERIMENTS.md
chain.sh, chain3.sh               one-off job chaining; wait on hardcoded PIDs, not reusable
```

The experiment runners are hardcoded to v1 and v2 checkpoints. They produce
metrics only, no renders.

Dataset construction and rendering live in `crisp_vla/src/crisp_vla/`:
`prepare_crisp_dataset.py`, `prepare_crisp_text.py`, `playback_crisp_motion.py`,
`compare_crisp_motion.py`, `build_gallery.py`.

---

## Changes to upstream files

Eight upstream files change, each by a few lines. The transformer, the input and
output layers, and the diffusion schedule are untouched.

### Registering the dataset

Additive — no existing behavior changes.

**`utils/parser_util.py`** — `'crisp'` added to `--dataset` choices, `'precomputed'`
to `--text_encoder_type` choices, and `'crisp'` to the text list in `get_cond_mode()`
so it conditions on text rather than falling through to `'action'`.

**`data_loaders/get_data.py`** — a `'crisp'` branch in `get_dataset_class`, in
`get_collate_fn` (returning `crisp_collate`, placed before the `hml_mode == 'gt'`
check since there is no HumanML3D eval mode here), and in `get_dataset`. The
`Crisp` constructor takes `fixed_len` and `device`, not `mode`, `abs_path` or
`autoregressive`. `get_dataset` and `get_dataset_loader` also take `data_dir`,
which selects a prepared dataset directory for `crisp`.

**`utils/model_util.py`** — a `crisp` branch setting `njoints=10, nfeats=1,
data_rep='hml_vec'`, plus one cross-cutting line:

```python
clip_dim = getattr(getattr(data, 'dataset', None), 'text_dim', 512)
```

Upstream never passed `clip_dim`, so it was pinned to MDM's 512 default. Reading
it from the dataset — defaulting to 512, so existing datasets are unaffected — is
what makes the text encoder swappable without editing the model.

`data_rep='hml_vec'` only selects the plain-`Linear` path in
`InputProcess`/`OutputProcess`. It implies no HumanML3D geometry.

### The velocity loss

**`diffusion/gaussian_diffusion.py`** — adds a `vel_loss_drop_last` flag, set in
`utils/model_util.py` to `False` for `crisp` and `True` for everything else.

Upstream's velocity loss always discards the last channel, because in SMPL's
rotation format that channel holds root translation rather than a joint. For this
robot the last channel is `Right_Elbow_Yaw`, a real joint, so upstream's loss would
silently ignore it. With the flag, `--lambda_vel` is safe to use on `crisp`; v2
trains at 1.0. Behavior for SMPL datasets is unchanged.

### Making SMPL optional

`MDM.__init__` builds `Rotation2xyz` unconditionally, which loads
`body_models/smpl/SMPL_NEUTRAL.pkl`. A robot has no SMPL body, so this crashed
before the first training step.

**`model/mdm.py`** — construction wrapped in `try/except`, falling back to
`self.rot2xyz = None` with a printed reason, and the two uses in `_apply` and `train`
guarded. Also adds the `'precomputed'` text encoder: no encoder is built, and
`encode_text` raises a clear error if called. Conditioning itself needed **no**
patch — `forward` already accepted `cond['y']['text_embed']`.

**`train/train_mdm.py`** — guards `model.rot2xyz.smpl_model.eval()`, and passes
`data_dir` through to the data loader.

Behavior is identical when SMPL assets are present.

### Upstream bugs (worth PRing back)

**`train/training_loop.py`** imported the HumanML3D evaluator, SMPL and the moviepy
plotting stack at module level. That made `train_mdm.py` impossible to import for
*any* dataset without a full HumanML3D install. Four imports now load inside the
functions that use them; all were already gated on `dataset in ['kit','humanml']` or
`eval_during_training`, so upstream behavior is unchanged.

**`train/train_mdm.py`** built the training platform *before* checking whether
`save_dir` exists. `TensorboardPlatform` creates that directory as it starts, so
`--train_platform_type TensorboardPlatform` raised `FileExistsError` on every fresh
run. The check now comes first.

### Sampling on the training canvas

**`sample/generate_crisp.py`** (a new file, but the fix matters) always samples on
the full 201-frame canvas and trims each motion to its requested length afterward.

Every training sequence is padded to 201 frames, so that is the only canvas the
model has seen. Sampling on a shorter canvas is off-distribution and makes
amplitude swing wildly — one caption, one checkpoint, gave peaks of 0.03, 0.85,
0.05 and 1.13 at canvases of 80, 120, 170 and 201 frames. The fix landed in commit
`2e039da`.

**Anything sampled with `generate_crisp.py` before `2e039da` is unreliable**,
including the galleries under `renders/gallery_model*`. The committed experiment
results were measured with their own 201-frame harness and are not affected.

---

## Text conditioning

Captions are embedded **offline** into `dataset/crisp/text/<encoder>.npz` and passed
in through `cond['y']['text_embed']`, so there is no text encoder in the training
graph — a trained checkpoint contains no `clip_model` weights. Swapping encoders
means re-running one script:

```bash
./crisp text --encoder clip_local                              # CLIP ViT-L/14, 768-d, offline
./crisp text --encoder st --model google/embeddinggemma-300m   # any sentence-transformers model
```

A linear probe, grouped by behavior, confirms the embeddings still encode style and
speed: **0.986 / 0.970** accuracy against chance of 0.333 / 0.500.

---

## Model

v1 and v2 use 5.07M trainable parameters, against original MDM's 17.88M for
HumanML3D — but on 822 training samples instead of ~23,000, so **parameters per
sample are about 8× higher**. Treat overfitting as the default hypothesis; v4–v7
test smaller models for that reason.

| | original MDM (humanml) | v1 / v2 |
|---|---|---|
| trainable | 17,880,327 | 5,072,394 |
| `latent_dim` / `layers` | 512 / 8 | 256 / 6 |
| `njoints` | 263 | 10 |
| `num_frames` | 60 | 201 |

`arch`, `num_heads`, `ff_size`, `diffusion_steps`, `noise_schedule`, `lr`,
`batch_size` and `cond_mask_prob` are all upstream defaults.

---

## Known limitations

**Retrieval beats the model on held-out combinations.** See *Where things stand*
and `FINDINGS.md`. Any compositional-generalization claim has to clear that bar.

**Mode collapse.** Sample diversity is 0.105 (v1) and 0.172 (v2) against 0.60 in
the data. Training loss never reveals it — it has to be measured.

**No validation loss.** MDM's loop logs training loss only, which on 822 samples
keeps falling through memorization. The `val` split exists but nothing uses it;
adding a val-loss hook in `training_loop.py` would make runs self-diagnosing.

**No learned evaluator.** There is no text-to-motion evaluator for this robot, so
`run_experiments.py` measures peak excursion, path length, trajectory shape and
moved-joint overlap directly. Re-scoring generated motion with the same VLM critic
that accepted the dataset (E3 in `EXPERIMENTS.md`) is planned but not built.

**`test_behav` is weak evidence.** The 60 behavior labels span only ~14 distinct
motion families. `listening_nod` and `small_acknowledgement_nod` sit at a caption
cosine distance of 0.000; `apology_bow` and `polite_bow` at 0.056; the median
between any two behaviors is 0.844. The best possible 6-behavior holdout still
leaves a training neighbour at 0.054. That is a ceiling in the data, not in the
split algorithm.

**`test_comp` is the defensible split.** Style and length really are separated in
the data: peak excursion rises 0.58 → 0.74 → 0.99 across subtle, neutral and
exaggerated, holding within 90 of 120 behavior/length pairs, and the two lengths
barely overlap (≈2.05 s vs ≈4.01 s, 12% overlap).

**Data coverage is lopsided.** 29 of 60 behaviors move only the head, 6 move the
left arm, and 4 move both arms. `welcoming_open_arms` (ground-truth peak 2.20)
generates at 0.07 — there is effectively one example of that motion pattern.
Mirrored augmentation addresses the left/right side; nothing yet addresses the
shortage of two-armed behaviors.

**Every training sequence is padded to 201 frames.** The median clip is 61 frames,
so about 70% of each sequence is masked padding, and it ties the model to a single
canvas size — the root cause of the sampling bug above.

**`--text` in `generate_crisp.py` is untested.** Only `--split` has been exercised.
The style-interpolation experiments feed embeddings in directly rather than through
`--text`.

---

## Environment notes

- **uv, not conda.** Upstream `environment.yml` is a Python 3.7 / CUDA 11.0 conda
  export that no longer resolves. `uv sync` gives torch 2.10.0+cu128 on Python 3.10.
- **SMPL assets are absent and not needed.** The model prints
  `Rotation2xyz unavailable ...; rot2xyz disabled.` at startup. That is expected.
- **`MUJOCO_GL=osmesa`.** On the original host EGL rendering fails (identically
  across mujoco 3.3.3 and 3.12.0, so it is a driver issue, not a package one) and
  OSMesa renders correctly on CPU. `./crisp` sets this automatically;
  `crisp_vla/run_scenario.sh` defaults to `egl` and needs the override. EGL may work
  on other machines.
- `spacy`, `chumpy` and the HumanML3D evaluator stack are **not** installed. They
  sit in an optional `humanml` dependency group.

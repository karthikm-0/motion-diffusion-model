# CRISP — MDM on the Booster T1 expressive-behavior dataset

This fork adds a dataset (`crisp`) to [MDM](https://github.com/GuyTevet/motion-diffusion-model)
so it trains on LLM-generated expressive robot behaviors instead of human motion
capture. Branch `crisp`, based on upstream `ef8edce`.

The data comes from the `crisp_vla` pipeline: a scenario prompt is turned into a
behavior plan, compiled to a joint program, rendered in MuJoCo, and scored by a
visual critic, retrying until it passes. This trains a diffusion model on the
accepted output.

**Upstream diff is small — 68 insertions, 15 deletions across 6 files.** Nothing in
`diffusion/`, nothing in the transformer, no loss changes. MDM's architecture is
representation-agnostic, so a 10-DoF robot flows through the stock `hml_vec` path
unmodified; most of the work was tooling around it.

---

## The dataset

| | |
|---|---|
| source | `final-ds/` — 1080 variants, 1019 with `status == "accepted"` |
| structure | 60 behaviors × 3 styles (subtle/neutral/exaggerated) × 2 lengths (fast_short/slow_long) = 360 cells, 357 populated, ~3 variants each |
| representation | **10 joint angles** in radians, 20 fps, 17–201 frames |
| conditioning | the `scenario` caption — 357 unique strings |

The T1 has 23 joints, but **13 are constant across the entire dataset** (waist +
both legs — the generator never moves them). They are dropped before training,
since zero-variance channels break normalization, and re-inserted at playback
from `meta.json["constant_joints"]`.

Source trajectories store **keyframes**, not frames. The converter densifies them
with the same min-jerk / smoothstep / linear weights used by
`crisp_vla/physics_playback.py`, so training data matches what the critic saw.
Verified exact (0.00e+00 error) at keyframes landing on the 20 fps grid.

### Splits

Splits are by **cell**, never by variant: a cell's ~3 variants share a
byte-identical caption, so splitting them would leak the exact prompt.

```
train        822 variants  289 cells
val           35 variants   12 cells
test_comp     59 variants   20 cells   behavior × style × length never seen together
test_behav   103 variants   36 cells   6 behaviors held out entirely
```

`Mean.npy` / `Std.npy` are computed on **train only**.

---

## Quickstart

Everything runs through the `./crisp` dispatcher at the repo root, which pairs
these scripts with the uv env and sets `MUJOCO_GL`.

```bash
uv sync                                   # in this directory

./crisp dataset --held_behaviors apology_bow,beckon_left,listening_nod
./crisp text    --encoder clip_local      # precompute caption embeddings

./crisp train   --save_dir save/crisp_v1 --num_frames 201 \
                --latent_dim 256 --layers 6 --batch_size 64 \
                --num_steps 60000 --save_interval 10000

./crisp sample  --model save/crisp_v1/model000060000.pt --split test_comp
./crisp compare --samples save/crisp_v1/samples_model000060000
./crisp serve                             # browse renders on :8000
```

`--lambda_vel` **must stay 0.** MDM's velocity loss drops the last channel as
"the root location" ([`gaussian_diffusion.py:1337`](diffusion/gaussian_diffusion.py));
here that channel is `Right_Elbow_Yaw`, a real joint. The other λ default to 0 too.

---

## Changes to upstream files

### Registering the dataset (additive — no existing behavior changes)

**`utils/parser_util.py`** — `'crisp'` added to `--dataset` choices, `'precomputed'`
to `--text_encoder_type` choices, and `'crisp'` to the text list in `get_cond_mode()`
so it conditions on text rather than falling through to `'action'`.

**`data_loaders/get_data.py`** — a `'crisp'` branch in `get_dataset_class`, in
`get_collate_fn` (returning `crisp_collate`, placed before the `hml_mode == 'gt'`
check since there is no HumanML3D eval mode here), and in `get_dataset` (the
`Crisp` constructor takes `fixed_len`/`device` but not `mode`/`abs_path`/`autoregressive`).

**`utils/model_util.py`** — a `crisp` branch setting `njoints=10, nfeats=1,
data_rep='hml_vec'`, plus one cross-cutting line:

```python
clip_dim = getattr(getattr(data, 'dataset', None), 'text_dim', 512)
```

Upstream never passed `clip_dim`, so it was pinned to MDM's 512 default. Reading
it from the dataset (defaulting to 512, so existing datasets are unaffected) is
what makes the text encoder swappable — a different embedding dim needs no model
edit.

`data_rep='hml_vec'` selects only the plain-`Linear` path in
`InputProcess`/`OutputProcess`. It implies no HumanML3D geometry.

### Making SMPL optional

`MDM.__init__` builds `Rotation2xyz` unconditionally, which loads
`body_models/smpl/SMPL_NEUTRAL.pkl`. A robot dataset has no SMPL body, so this
crashed before step 1.

**`model/mdm.py`** — construction wrapped in `try/except` → `self.rot2xyz = None`
with a printed reason, and the two dereferences in `_apply` and `train` guarded.
Also adds the `'precomputed'` text encoder branch: no encoder is built, and
`encode_text` is bound to a method that raises a clear error if called.
Conditioning itself needed **no** patch — `forward` already supported
`cond['y']['text_embed']` as a caching hook.

**`train/train_mdm.py`** — guards `model.rot2xyz.smpl_model.eval()`.

Behavior is identical when SMPL assets are present.

### Upstream bugs (worth PRing back)

**`train/training_loop.py`** imported the HumanML3D evaluator, SMPL and the
moviepy plotting stack at module scope, making `train_mdm.py` unimportable for
*any* dataset without a full HumanML3D install (fails on spacy → smplx →
moviepy in sequence). Four imports moved inside their call sites; all were
already gated on `dataset in ['kit','humanml']` or `eval_during_training`, so
upstream behavior is unchanged.

**`train/train_mdm.py`** constructed the train platform *before* the
`os.path.exists(save_dir)` check. `TensorboardPlatform.__init__` creates a
`SummaryWriter(log_dir=save_dir)`, which creates the directory — so
`--train_platform_type TensorboardPlatform` raised `FileExistsError` on every
fresh run. The validation now precedes platform construction.

### Not changed

`diffusion/` (untouched), the transformer, `InputProcess`/`OutputProcess`, and
every loss. Training uses `L_simple` only — the masked MSE on x̂₀ — which is also
what upstream does for text-to-motion. The geometric losses are used only for the
SMPL action-to-motion datasets (README line 464).

---

## New files

```
data_loaders/crisp/dataset.py      Crisp dataset + crisp_collate
sample/generate_crisp.py           sampling (MDM's generate.py is SMPL-only)
pyproject.toml / uv.lock           uv env; environment.yml is a py3.7/CUDA-11 export
```

Dataset construction and rendering live in `crisp_vla/src/crisp_vla/`:
`prepare_crisp_dataset.py`, `prepare_crisp_text.py`, `playback_crisp_motion.py`,
`compare_crisp_motion.py`.

`dataset/crisp/` is gitignored — regenerate it with the two prepare scripts.

### Text conditioning

Captions are embedded **offline** into `dataset/crisp/text/<encoder>.npz` and
delivered via `cond['y']['text_embed']`, so no text encoder is in the training
graph (a trained checkpoint contains zero `clip_model` tensors). Swapping
encoders is re-running one script:

```bash
./crisp text --encoder clip_local                              # CLIP ViT-L/14, 768-d, offline
./crisp text --encoder st --model google/embeddinggemma-300m   # any sentence-transformers model
```

A linear probe (grouped by behavior) confirms style and speed survive caching:
**0.986 / 0.970** against chance of 0.333 / 0.500.

---

## Model

5.07M trainable params vs original MDM's 17.88M for HumanML3D — but on 822
samples vs ~23,000, so **params per sample are ~8× higher**. Treat overfitting as
the default hypothesis.

| | original MDM (humanml) | this fork |
|---|---|---|
| trainable | 17,880,327 | 5,072,394 |
| `latent_dim` / `layers` | 512 / 8 | 256 / 6 |
| `njoints` | 263 | 10 |
| `num_frames` | 60 | 201 |

`arch`, `num_heads`, `ff_size`, `diffusion_steps`, `noise_schedule`, `lr`,
`batch_size` and `cond_mask_prob` are all stock.

---

## Known limitations

**`test_behav` is weak evidence.** The 60 behavior labels span only ~14 distinct
motion families — `listening_nod` and `small_acknowledgement_nod` differ by a
cosine distance of 0.000; `apology_bow` and `polite_bow` by 0.056; the median
between any two behaviors is 0.844. The best achievable 6-behavior holdout still
leaves a training neighbour at 0.054. This is a ceiling in the data, not the
split algorithm. `prepare_crisp_dataset.py` prints a nearest-train audit for
whatever holdout you choose — read it before quoting a result.

**`test_comp` is the defensible claim.** The style and length axes are genuinely
separated: peak excursion rises monotonically 0.58 → 0.74 → 0.99 across
subtle/neutral/exaggerated, holding per behavior in 90/120 cases, and durations
are near-disjoint (2.05 s vs 4.01 s, 12% overlap).

**No validation loss.** MDM's loop logs train loss only, which on 822 samples
declines through memorization. The `val` split exists but is unused — adding a
val-loss hook would make runs self-diagnosing.

**No evaluation harness yet.** There is no learned T2M evaluator for this robot,
so checkpoint selection is currently by eye. Planned: held-out-cell amplitude and
duration matching, a CLIP-retrieval baseline, and re-scoring samples through the
same visual critic that gated the dataset.

**Generated motion is jittery.** At 10k steps, samples reach roughly the right
amplitude (mean |peak difference| 0.061 rad) but travel **~2.1× the ground-truth
path length** — the right poses via a noisy route, where the originals are smooth
min-jerk splines. A corrected `L_vel` (without the root-dropping assumption) is
the obvious candidate.

**`--text` in `generate_crisp.py` is untested.** Only the `--split` path has been
exercised.

---

## Environment notes

- **uv, not conda.** Upstream `environment.yml` is a Python 3.7 / CUDA 11.0 conda
  export that no longer resolves. `uv sync` gives torch 2.10.0+cu128 on Python 3.10.
- **SMPL assets are absent and not needed.** The model prints
  `Rotation2xyz unavailable ...; rot2xyz disabled.` at startup — expected.
- **`MUJOCO_GL=osmesa`.** EGL is broken on this host (fails identically across
  mujoco 3.3.3 and 3.12.0, so it is a driver/headless issue, not a package one).
  OSMesa renders correctly on CPU. `./crisp` exports this automatically;
  `crisp_vla/run_scenario.sh` defaults to `egl` and will need the override.
- `spacy`, `chumpy` and the HumanML3D evaluator stack are **not** installed —
  they sit in an optional `humanml` dependency group.

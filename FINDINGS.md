# Findings — crisp_v1 vs crisp_v2

Two runs, identical except `lambda_vel` (v1 = 0.0, v2 = 1.0). Both 60k steps,
`latent_dim 256`, `layers 6`, 5.07M trainable params.

All numbers use a **201-frame sampling canvas** — read the harness bug first,
because everything measured before that fix is void.

---

## 0. A harness bug that invalidated the first pass

`Crisp.__getitem__` pads **every** training sequence to 201 frames, so 201 is the
only canvas the model ever saw. Both `sample/generate_crisp.py` and the first
experiment harness generated on a canvas of `max(batch lengths)`; E4/E5/E7 used 80.
Off-distribution canvases make output amplitude swing erratically — same caption,
same checkpoint:

| behaviour | GT peak | T=80 | T=120 | T=170 | T=201 |
|---|---|---|---|---|---|
| acknowledge_applause | 1.00 | 0.03 | 0.85 | 0.05 | 1.13 |
| after_you | 1.70 | 0.25 | 1.25 | 1.39 | 1.40 |
| celebration_arm_raise | 2.00 | 0.07 | 0.07 | 0.11 | 1.98 |

Pinning the canvas to 201 moved `all_heldout` peak error **0.480 → 0.287**.

**Fixed** in `run_experiments.py` (`CANVAS = 201`). **Still to fix:**
`sample/generate_crisp.py` has the same bug — anything sampled with it, including
the gallery renders, is suspect.

`mask_frames` is off, so the length mask does nothing at sampling time; it only
shapes the training loss. Generate at 201 and truncate afterwards.

---

## 1. Results

### Retrieval baseline (E2) — the headline, and it is negative

Nearest training caption by cosine similarity, returning that motion verbatim.
Two metrics: `peak_err` (amplitude only) and `shape_cos` (cosine between
time-normalized excursion trajectories — did it produce the *right motion*).

**Amplitude:**

| model | split | MDM | retrieval | winner |
|---|---|---|---|---|
| v1 | test_comp | 0.255 | **0.196** | retrieval |
| v1 | val | **0.171** | 0.270 | MDM |
| v2 | test_comp | 0.304 | **0.196** | retrieval |
| v2 | val | **0.148** | 0.270 | MDM |

**Shape (higher is better) — the fair test:**

| model | split | MDM shape_cos | retrieval | jacc MDM/ret | winner |
|---|---|---|---|---|---|
| v1 | all_heldout | 0.392 | **0.479** | 0.67 / 0.64 | retrieval |
| v1 | test_comp | 0.507 | **0.661** | 0.71 / 0.79 | retrieval |
| v1 | test_behav | 0.314 | **0.324** | 0.57 / 0.51 | retrieval |
| v1 | val | 0.425 | **0.629** | 0.86 / 0.79 | retrieval |
| v2 | all_heldout | 0.429 | **0.479** | 0.66 / 0.64 | retrieval |
| v2 | test_comp | 0.583 | **0.661** | 0.69 / 0.79 | retrieval |
| v2 | test_behav | **0.339** | 0.324 | 0.55 / 0.51 | MDM |
| v2 | val | 0.431 | **0.629** | 0.90 / 0.79 | retrieval |

**Retrieval wins 7 of 8.** I expected `peak_err` to be unfairly favouring retrieval,
since it returns a real training motion whose amplitude is in-distribution by
construction. It was biased — but in the *other* direction: the shape metric makes
MDM look **worse**, and retrieval also picks the correct moving joints more often on
`test_comp` (Jaccard 0.79 vs 0.69–0.71).

The single MDM win (v2 / `test_behav`, 0.339 vs 0.324) is marginal and both scores
are poor — neither approach handles unseen behaviors.

**Read:** on held-out cells, returning the nearest training motion by caption
similarity beats the trained diffusion model on essentially every metric. With 357
conditions over ~14 motion families, a generative model is not currently earning its
place on this task.

### Style control (E4) — the one result retrieval cannot match

Interpolating caption embeddings between `subtle` and `exaggerated`, **6 seeds per
α**, averaged over 12 behaviours:

```
α:      0.00   0.25   0.50   0.75   1.00  |  1.25   1.50    ratio
v1:     0.71   0.80   1.03   1.20   1.29  |  1.21   1.34    1.82
v2:     0.76   0.82   0.86   0.92   1.01  |  1.15   1.08    1.33
                                          ^ end of training range
dataset subtle->exaggerated ratio: 1.70
```

Both mean curves are **monotone across the trained interval**. v1's range (1.82)
closely matches the data's own 1.70; v2's is compressed to 1.33. Direction correct
in 11/12 (v1) and 10/12 (v2) behaviours.

This is a continuous control axis learned from three discrete labels, and
**retrieval structurally cannot produce a point at α=0.5**. It is the only
demonstrated capability a nearest-neighbour lookup does not have.

Caveats: per-behaviour curves remain noisy (6/12 monotone) because sample diversity
is 0.105 and 6 seeds does not fully average it out — only the aggregate is clean.
And **extrapolation does not hold**: past α=1 the curve wobbles (1.29 → 1.21 →
1.34), so the claim is interpolation within range, not extrapolation.

### Sample diversity (E7)

| | median pairwise relative difference |
|---|---|
| dataset variants | **0.60** |
| v1 | 0.105 |
| v2 | 0.172 |

Both far tighter than the data. **Partial mode collapse is real** — the model
learned a much narrower conditional distribution than the data exhibits, and
training loss never reveals it.

### Behavior blending (E5) — inconclusive

v1 collapsed to one endpoint in all three pairs; v2 looked more intermediate. These
are **single samples** and the differences sit inside the noise floor. Needs the
multi-seed treatment before any claim.

### λ_vel (E0) — a genuine trade, no clear winner

| | v1 (λ=0) | v2 (λ=1) |
|---|---|---|
| amplitude accuracy (test_comp peak_err) | **0.255** | 0.304 |
| trajectory shape (test_comp shape_cos) | 0.507 | **0.583** |
| style range (ratio vs data's 1.70) | **1.82** | 1.33 |
| sample diversity (data 0.60) | 0.105 | **0.172** |
| smoothness (path ratio) | 1.16 | **1.12** |

λ_vel improves **shape fidelity, diversity and smoothness**; it costs **amplitude
accuracy and style range**. Which checkpoint is better depends on the claim:
**v1 for the style-control figure, v2 for motion quality.** The jitter λ_vel
targeted (path ratio 2.12 at 10k) had largely resolved on its own by 60k.

A middle setting (0.1–0.3) is the obvious next run — the two effects trade
smoothly enough that an intermediate value may keep the shape gain without
compressing style range.

### Guidance sweep (test_comp, peak_err)

```
        1.0     1.5     2.5     4.0     6.0
v1     0.337   0.325   0.266   0.257   0.271
v2     0.327   0.301   0.293   0.261   0.287
```

Shallow bowl, minimum at 4.0 for both; default 2.5 near-optimal. **CFG was never
the cause of the low amplitudes** — that was the canvas bug.

---

## 2. What to fix

### Data / generation — highest leverage

The dataset is severely imbalanced **by motion pattern**, not just behavior label:

```
head_only  29/60      right arm  21/60
left arm    6/60      bilateral   4/60
```

Only four behaviors move both arms, two of which are bows (head-dominant), so
bilateral arm-raising has essentially **one** exemplar. Consequence:
`welcoming_open_arms` (GT peak 2.20, rank 7/60 by amplitude) generates at **0.07** —
the model cannot produce it at all.

- Generate more left-arm and bilateral behaviors. Generator prompt/scenario work,
  not a model change, and the single clearest fix.
- Left-side results rest on 6/60 behaviors; treat them as anecdotal.

### Training pipeline

- **Add a validation-loss hook.** The 35-sample `val` split is unused. Train loss
  falls indefinitely by memorization on 822 samples and told us nothing about mode
  collapse or the amplitude failures. ~15 lines in `training_loop.py`; makes every
  future run self-diagnosing.
- **Log sample diversity during training.** E7 is cheap and is the only thing that
  would have caught the collapse early.
- **Reconsider fixed-201 padding.** Median clip is 61 frames, so ~70% of every
  training sequence is masked padding, and it hard-couples the model to one canvas
  size. Length bucketing, or training at several canvas sizes, removes the sampling
  fragility entirely.

### Model / architecture

- **Mode collapse is the core weakness.** 5.07M params on 822 samples with plain
  MSE gives a near-deterministic conditional. Cheapest first: mirrored left/right
  augmentation (also fixes the 6/60 left-arm imbalance); shrink to `--layers 4
  --latent_dim 128` (params-per-sample is ~8× original MDM's); then longer training.
- **λ_vel 0.1–0.3**, not 1.0.
- **Do not chase CFG** — already at its optimum.

---

## 3. Honest status

**The core claim did not survive.** On held-out compositional cells, a
nearest-neighbour lookup on caption embeddings beats the trained model on
amplitude, trajectory shape, and which joints move — 7 of 8 comparisons. The
compositional-generalization result the work was aiming for is not there.

**One capability survives and is retrieval-proof:** continuous style control.
Interpolating between `subtle` and `exaggerated` gives a monotone amplitude curve
across the trained range with a magnitude matching the dataset's own 1.70 ratio.
No lookup table produces α=0.5. That is a real but much narrower claim than
"expressive behavior generation".

**Three things are broken or missing**, in priority order:

1. **Mode collapse** — sample diversity 0.105–0.172 against 0.60 in the data. The
   model is close to a deterministic conditional, which is also the most likely
   reason it loses to retrieval: it returns a blurred average where retrieval
   returns a real, sharp motion.
2. **Data coverage** — 4/60 behaviors bilateral, 6/60 left-arm.
   `welcoming_open_arms` (GT peak 2.20) generates at 0.07.
3. **No validation signal** — nothing in training would have surfaced 1 or 2.

**What I would do next, in order:**

- Fix the canvas bug in `sample/generate_crisp.py` (everything sampled with it,
  including the galleries, is currently suspect).
- Add val-loss and diversity logging, then re-run. Without these you cannot tell a
  good run from a bad one.
- Attack mode collapse before anything else — mirrored augmentation, a smaller
  model, λ_vel ≈ 0.2. If diversity does not approach the data's 0.60, the model
  will keep losing to retrieval.
- Only then revisit the retrieval comparison. If it still loses, the honest paper
  is about the *generation pipeline* and the style-control result, not about
  beating retrieval at compositional generalization.

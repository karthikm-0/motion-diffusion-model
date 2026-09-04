# Experiments

What to run on `crisp_v1` (λ_vel=0) and `crisp_v2` (λ_vel=1), and what each result
would actually license you to claim.

Ordered so that the cheap credibility checks come before the interesting ones —
if E2 fails, most of the rest stops being worth reporting.

Available without any learned evaluator: peak excursion, joint path length,
duration, moved-joint sets, and **the same VLM critic that gated the dataset**.
That last one is the most under-used asset here.

---

## Tier 0 — did λ_vel do anything

### E0. v1 vs v2 on smoothness

**Claim:** the velocity term reduces jitter without costing accuracy.

**Measure:** median path-length ratio (generated ÷ ground truth) over all 197
held-out samples, plus mean |peak difference|. Already computed by
`./crisp gallery`. Baselines measured on v1: ratio **2.12 at 10k**, **1.32 at 30k**.

```bash
./crisp gallery --model save/crisp_v1/model000060012.pt --jobs 24
./crisp gallery --model save/crisp_v2/model000060012.pt --jobs 24
```

**Read it:** ratio → 1.0 is the target. If v1 already reaches ~1.0 by 60k, λ_vel
bought nothing and you should say so. If v2 is closer but peak error is worse,
that is a real trade-off worth reporting, not a failure.

**Effort:** ~20 min (mostly rendering). No new code.

---

## Tier 1 — credibility

### E1. Held-out cell reconstruction

**Claim:** the model produces the right motion for combinations it never saw.

**Measure:** for each of the 59 `test_comp` samples, does generated peak excursion
and duration fall inside the band of the held-out cell's ground-truth variants?
Report hit rate, and the per-axis error.

**Stronger version — ranking.** For a behavior with 5 cells in train and 1 held
out, generate all 6 and check the amplitude ordering is
`subtle < neutral < exaggerated`. Ground truth holds this in **90/120**
(behavior, speed) pairs, so that is the number to beat, not 100%.

**Effort:** half a day. Needs a small script over `results.json` + `index.json`.

### E2. Retrieval baseline — **run this before anything else**

**Claim:** the model does more than look up the nearest training example.

**Method:** for each held-out caption, embed it, find the nearest *training*
caption by cosine similarity in the cached embedding space, and return that
training motion verbatim. Score it with the same metrics as E1.

**Read it:** if MDM does not beat retrieval on `test_comp`, the honest conclusion
is that 357 conditions over ~14 motion families does not need a generative model,
and the paper needs a different claim. If it does beat it — especially on cells
where the nearest neighbour is a different style — that is the core result.

**Why it matters:** it is the first objection a reviewer will raise, and with this
dataset's structure it is a fair one.

**Effort:** a few hours. Embeddings are already cached in
`dataset/crisp/text/*.npz`; it is a nearest-neighbour lookup plus the E1 metrics.

### E3. Critic-in-the-loop

**Claim:** generated motions meet the same bar as the training data.

**Method:** render held-out samples and run them through the same VLM critic
(`score >= 7 and scenario_match`, `gemini-3.5-flash-lite`, 8 fps video + 8 stills)
that accepted the dataset.

**Read it:** *"X% of generated motions pass the same critic that accepted the
training data"* is a clean headline, and it closes the loop — the pipeline's own
judge evaluates the model trained on its output. Compare against the dataset's own
acceptance rate as the ceiling.

**Watch for:** the critic scored 787/1019 at first attempt, so its bar is not
especially high; report the score *distribution*, not just pass rate.

**Effort:** a day, mostly wiring `generate_dataset.py`'s critic call to accept
external videos.

---

## Tier 2 — the interesting claims

These are the ones worth showing the community. E4 and E5 are the strongest
because **neither is possible for a retrieval system**, which makes them the
direct answer to E2's objection.

### E4. Continuous style control ★

**Claim:** the model learned style as a continuous axis, not three labels.

**Method:** style exists only as 3 discrete levels in the data, but conditioning is
a continuous embedding. For a fixed behavior, interpolate

```
e(α) = (1−α)·e_subtle + α·e_exaggerated ,  α ∈ [0, 1]
```

sample at α = 0, 0.25, 0.5, 0.75, 1, and plot peak excursion against α.
Then **extrapolate**: α = 1.25, 1.5 — does amplitude keep growing?

**Read it:** a monotone, smooth curve means you have a continuous intensity knob
the training data never contained. A lookup table categorically cannot produce
this. Extrapolation working is a bonus; failing gracefully is fine, failing
catastrophically is worth reporting honestly.

**Why it is the showpiece:** it is visual (a strip of renders at increasing α),
quantitative (one curve), and unavailable to retrieval.

**Effort:** half a day. `generate_crisp.py --text` already embeds novel strings;
this is a loop over α feeding embeddings directly.

### E5. Behavior blending ★

**Claim:** the conditioning space composes behaviors, not just modifiers.

**Method:** average two behavior caption embeddings — `nod` + `wave`, or
`point_left` + `celebration_arm_raise` — and sample. Each training caption
describes exactly one behavior, so any blend is outside the data.

**Read it:** does it produce a nod *and* a wave, one of them, or an incoherent
mixture? All three outcomes are reportable. Coherent blending would be a genuinely
striking demo; "collapses to the dominant behavior" is an honest and interesting
negative result about the conditioning geometry.

**Effort:** half a day, shares all its tooling with E4.

### E6. Arbitrary scenarios

**Claim:** it responds sensibly to prompts outside the generated set.

**Method:** write 15–20 scenarios in the dataset's register but describing
situations never generated — *"someone drops their keys nearby"*, *"a child waves
shyly from across the room"*. Embed, sample, render, and have humans (or the
critic) rate whether the response is socially appropriate.

**Read it:** this is the demo that reads as "expressive behavior generation"
rather than "trajectory interpolation". Expect it to map novel prompts onto
nearby known behaviors — which is fine, and worth measuring: report *which*
training behavior each novel prompt lands closest to.

**Caveat:** `--text` in `generate_crisp.py` is **implemented but untested**. Verify
it before building on it.

**Effort:** a day including writing prompts and rating.

### E7. Sample diversity matches data diversity

**Claim:** it learned a distribution over motions, not the conditional mean.

**Method:** sample the same caption N=10 times with different seeds; measure
pairwise difference the same way the dataset's variants were measured
(time-normalized, relative to each motion's own excursion).

**Reference numbers from the data:** variants of the same cell differ by a median
of **0.60** relative difference; only 2.6% of pairs are near-duplicates, and 34.5%
use a *different set of joints*.

**Read it:** if generated samples cluster far tighter than 0.60, the model is
mode-collapsing to the mean — a real and reportable weakness for a *generative*
model, and one the training loss will never reveal.

**Effort:** a few hours; `--num_repetitions` already exists.

---

## Tier 3 — stretch

### E8. Duration control

Generate a known behavior at lengths absent from the data (the grid has only
`fast_short` ≈2.05 s and `slow_long` ≈4.01 s, 12% overlap). Ask for 3 s. Does it
interpolate timing, or snap to one of the two modes?

### E9. In-betweening / editing

MDM supports inpainting: fix the first and last N frames from a ground-truth
motion and let the model fill the middle. Shows the model as an editing tool, not
just a sampler. Needs the inpainting path wired for `crisp`.

### E10. Style transfer

Take a `subtle` ground-truth motion, condition on its `exaggerated` caption, and
inpaint — does it amplify the same choreography, or produce a different motion?

---

## What this dataset cannot support

State these rather than let a reviewer find them:

- **Held-out *behavior* generalization is weak evidence.** The 6 held-out names all
  have near-identical twins in train (`listening_nod` ↔ `small_acknowledgement_nod`
  at cosine distance **0.000**; best achievable 6-behavior holdout still leaves a
  neighbour at 0.054, against a median of 0.844). Report `test_behav` with the
  nearest-train audit attached, or not at all.
- **Open-vocabulary text generalization.** Captions are templated —
  `<base sentence> + <one of 3 style clauses> + <one of 2 speed clauses>` — so E6
  probes robustness, not true open-vocabulary understanding.
- **Dynamic feasibility.** Nothing here validates that a robot can execute these.
  Upper-body only, stationary base.

---

## Suggested order

1. **E2 retrieval baseline** — decides whether the rest is worth writing up
2. **E0** — cheap, already tooled, settles the λ_vel question
3. **E4 + E5** — the showpieces, shared tooling, ~1 day together
4. **E1** — the quantitative backbone for the composition claim
5. **E3** — the framing win (the pipeline's own critic judges the model)
6. E6, E7 as time allows

E2 first is deliberate. Everything downstream is more interesting if you can say
"and it beats retrieval", and considerably less interesting if you cannot.

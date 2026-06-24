# Activation Steering Investigation — Findings & Reproduction Guide

Running log of the causal-intervention/steering experiments on LeWM's PushT
encoder. Each section has: what we found, where the numbers/artifacts live,
and the exact command to reproduce or extend it.

All code lives in `interp_utils/steering/`. All job scripts in `jobs/`.
Real probe weights: `/scratch-shared/orinxAI/stable-wm-data/probes/{block_angle,block_position,agent_position}/`.

## TL;DR

- **Probe-inversion steering (`steering_math.steering_vector`) doesn't point in the right direction.** Properly controlled rendering-based validation: cosine_sim ≈ **-0.025** (block_angle) and **0.107** (block_position) against the true empirical effect — even though block_position's probe is far more accurate (R=0.996 vs 0.933). Better probe accuracy doesn't fix it.
- **Direct delta regression (fit `Δfeature → Δh` from real rendered pairs) is better but still capped.** Held-out cosine_sim ≈ **0.30** (block_position, n=5000) and **0.19** (block_angle, n=5000). Confirmed with 33x more data (150→5000 examples) that this is a real ceiling, not sampling noise — in-sample and held-out scores are statistically indistinguishable.
- **Why it's capped**: per-example "true" steering direction varies enormously frame-to-frame — typical deviation from the population-average direction is **3-4x larger** than the average direction's own magnitude. No small set of PCA components explains it either (need ~10 PCs for 70% of variance). The embedding's response to a perturbation is dominated by frame-specific context, not a clean shared direction.
- **But discrete/coarse direction classification recovers far more than continuous regression suggests.** Binary sign classification: **99.7%** (block_position), **78.0%** (block_angle) — vs. chance 50%, with a shuffled-label negative control correctly collapsing to ~51% (confirms the pipeline isn't leaking signal). Magnitude-only (sign-agnostic) classification is much weaker (~59-65%) — **the embedding encodes *which way* far more reliably than *how far*.**
- **A teammate's higher numbers came from a different metric, not a better method.** Their `compare_embeddings()` computes cosine similarity on full embeddings (`steered` vs `synthetic`), not on delta vectors. Since the shared base embedding (`‖h‖≈8`) dominates the small deltas (`‖v‖≈1.3`, `‖δh‖≈3.9`), full-embedding cosine_sim looks high (0.86) even when delta-vs-delta cosine_sim is only 0.29 **on the same data**. Verified both analytically and empirically.
- **A classifier's decision direction is a worse steering vector than direct regression, despite being a good classifier.** For `block_angle`: 78% held-out classification accuracy (left/right), but its decision-boundary direction only achieves cosine_sim=**0.043** against the true delta — worse than delta-regression's 0.192. Discriminating direction and generating/steering in that direction are different objectives; logistic regression optimizes the former, not the latter.

**Direction-derivation methods ranked (block_angle, cosine_sim vs. true rendered delta):** probe-inversion -0.025 < classifier decision boundary 0.043 < direct delta regression **0.192** (best of the three).
- **The global direction's mediocre 0.192 was hiding a much bigger problem: it's anti-aligned (negative cosine_sim) across ~70% of the angle range.** Binning by the block's *current* angle and fitting a separate direction per bin (`conditional_direction.py`) flips both negative bins positive and improves every bin — the right steering direction genuinely depends on the current state, not just the requested delta (`v(Δ, h_base)`, not `v(Δ)`).
- **Discrimination and steering can point in opposite directions entirely.** The block's orientation (which quadrant it's pointing in) is *perfectly* classifiable for opposite quadrants (100% accuracy, 180° apart) — yet a delta-regression direction fit specifically on ~180° rotations gets cosine_sim **≈0** (worse than small rotations' 0.192). Easy separability comes from clusters sitting in different directions depending on context, not a uniform shift, so the easiest distinctions to classify can be the hardest to actually steer with a single vector.
- **On real task data (not synthetic perturbations), rotation-direction-to-goal is even more strongly encoded**: 95.6% held-out accuracy classifying which way the block needs to rotate to reach its actual 25-step-ahead task goal, from `h(goal) - h(current)` — beating every synthetic-perturbation result, though this real `delta_h` entangles other simultaneously-changing state too (position, velocity), so it's a less causally clean claim than the rendered single-variable tests.
- **Concatenating all 12 layers beats the best single layer for every target** (block_angle 0.933→0.984, block_position 0.996→0.9997, agent_position 0.984→0.997) — cross-layer information is complementary, not redundant. But PCA on the concatenation reveals `block_angle`/`agent_position`'s extra signal lives partly in low-variance directions: even 1200 of 2304 PCA components don't fully recover the raw-feature result, while `block_position` matches it with just 300.

## Background

Original review concern: `steering_vector` derives a perturbation by inverting
a linear probe (`D=192 → K=1or2`). The probe's null space is huge
(`D-K` dimensions) and completely unconstrained — the minimum-norm choice is
a modeling assumption, not a fact about the model. Real activations might
differ from `current + steering_vector` by a lot in directions the probe
never modeled. This doc is the empirical investigation of whether that's
actually a problem here.

---

## 1. Infrastructure

| File | Purpose |
|---|---|
| `interp_utils/steering/probe_io.py` | Load `linear_probe_weights.npz`/`split.json`/`metrics.csv` from `probing.py` output |
| `interp_utils/steering/steering_math.py` | `steering_vector()` — probe-inversion direction (independent-per-dimension, closed form) |
| `interp_utils/steering/episode_pairs.py` | Mine real episode pairs by initial-state delta (`find_episode_pairs`, `load_initial_states`) |
| `interp_utils/steering/metrics.py` | MSE/MAE/cosine-sim battery + CSV/JSON writer (`write_metrics_table`) |
| `interp_utils/steering/steered_model.py` | `SteeredLeWM` — forward-hook mechanism for live model steering (Stage 2/3 closed-loop use) |
| `interp_utils/steering/open_loop.py`, `closed_loop.py`, `aggregate.py` | Dataset-only / real-env / sweep steering experiments (built before the validation work below revealed the direction itself is the problem) |
| `interp_utils/steering/validate_direction.py` | **Superseded** — CPU-only check against episode-pair deltas; confounded by cross-episode differences (agent position, etc. also differ). Kept for reference. |
| `interp_utils/steering/validate_direction_rendered.py` | **The real validation tool.** Same scene, only target dim changed, re-rendered. Needs model+GPU+renderer. |
| `interp_utils/steering/delta_regression.py` | Direct `Δfeature → Δh` regression, fit from rendered pairs. Generates the `.npz` datasets everything below reuses. |
| `interp_utils/steering/direction_classification.py` | Sweep: bin deltas into N classes (sign × magnitude quantile), classification accuracy vs. N |
| `interp_utils/steering/binary_separability.py` | Battery of varied binary tests (sign, magnitude, restricted ranges, shuffled control) |
| `interp_utils/steering/classifier_direction.py` | Left/right classifier's decision direction as a candidate steering vector |
| `interp_utils/steering/conditional_direction.py` | Per-bin (conditioned on true base state) vs. global steering direction comparison |
| `interp_utils/steering/orientation_classification.py` | Quadrant (orientation) classification from a single frame's cached embedding, chunk-aligned block sampling |
| `interp_utils/steering/goal_relative_orientation.py` | Real (not synthetic) goal-relative rotation-direction classification, from `delta` or `current`-only features |
| `interp_utils/steering/concat_layer_probe.py` | Lightweight sklearn regression on concatenated all-layer CLS features, with optional PCA(n_components) sweep |
| `jobs/steer_open_loop.sh`, `steer_closed_loop.sh`, `steer_aggregate.sh`, `steer_delta_regression.sh`, `steer_conditional_direction.sh`, `steer_orientation_classification.sh`, `steer_goal_relative_orientation.sh`, `steer_concat_layer_probe.sh` | SLURM job wrappers (`gpu_a100` for rendering/encoding; `rome` CPU-only for pure numpy/sklearn analyses) |
| `tests/test_steering_math.py` etc. | Unit tests, pure numpy, no model needed |
| `poster.tex` (`Steering` block) | LaTeX writeup of the steering-vector derivation + delta-regression formula |

Teammate-provided code (read for comparison, not modified):
`interp_utils/_steering.py` (joint-pseudoinverse hook), `interp_utils/compare_steering_embeddings.py`
(full-embedding comparison), `render_utils.py` (PushT renderer — reused throughout).

---

## 2. Probe-inversion validation (confounded — episode pairs)

**Script**: `validate_direction.py`. Compares `steering_vector`'s direction
against the empirical delta between two *different* episodes mined to have
similar initial-state deltas. **Confounded**: those episodes differ in
agent position, etc. too, not just the target dim — explains the very low
cosine_sim numbers seen here before we built the rendering-based version.
Superseded by Section 3; kept only as the CPU-only/login-node-quick option.

```bash
python -m interp_utils.steering.validate_direction \
  --probe-dir /scratch-shared/orinxAI/stable-wm-data/probes/block_angle \
  --delta 0.3 --tolerance 0.02 --num-pairs 10
```

---

## 3. Probe-inversion validation (controlled — rendering-based)

**Script**: `validate_direction_rendered.py`. For one real frame, builds a
synthetic state identical except for the target dim, re-renders with
`render_pusht_state_vector`, encodes, and compares the *delta* vectors. No
cross-episode confound. Needs model+GPU+renderer (works headlessly, no
display needed — confirmed).

**Sanity check (always run first)**: `rerender_vs_real_cosine_sim ≈ 0.999`,
`norm ≈ 0.4-0.5` for delta=0 (re-rendering the unperturbed state reproduces
the original frame's embedding almost exactly) — rules out the rendering
pipeline itself being a confound.

### Results

| Target | Probe layer | Delta(s) tested | cosine_sim |
|---|---|---|---|
| `block_angle` | layer_09 (R=0.933) | 0.1, 0.3, 0.5 rad | **-0.025, -0.017, +0.004** |
| `block_position` (x) | layer_09 (R=0.996) | 10, 30, 50 px | **0.055, 0.094, 0.104** |

Flat/near-zero across a 5x range of magnitudes for both targets — not a
delta-size artifact. `probe_rendered_delta_error` is large and noisy at
every magnitude (one block_angle example even predicted the **wrong sign**
of the true change).

Outputs: `outputs/steering/validate_rendered/{block_angle,block_position}/delta_sweep.{csv,json}`

```bash
python -m interp_utils.steering.validate_direction_rendered \
  --dataset /scratch-shared/orinxAI/stable-wm-data/datasets/pusht_expert_train.h5 \
  --probe-dir /scratch-shared/orinxAI/stable-wm-data/probes/block_angle \
  --target block_angle --target-dim-index 0 \
  --deltas 0.1 0.3 0.5 --num-frames 2 --device cpu \
  --output-dir outputs/steering/validate_rendered/block_angle

python -m interp_utils.steering.validate_direction_rendered \
  --dataset /scratch-shared/orinxAI/stable-wm-data/datasets/pusht_expert_train.h5 \
  --probe-dir /scratch-shared/orinxAI/stable-wm-data/probes/block_position \
  --target block_position --target-dim-index 0 \
  --deltas 10 30 50 --num-frames 2 --device cpu \
  --output-dir outputs/steering/validate_rendered/block_position
```

---

## 4. Direct delta regression

**Script**: `delta_regression.py`. Instead of inverting the probe, fit
`Δh = Δ · B` directly (no intercept) from `N` rendered `(base_frame, delta)`
pairs. Reuses the same rendering methodology as Section 3.

**Timing** (checked via `sacct`, A100 GPU): ~0.3 sec/example. 150 examples →
84s total job. 5000 examples → ~11-17 min total job. Login-node estimates
earlier in the investigation were 100x too pessimistic — that was
filesystem contention (`uptime` load avg 60-130 during parts of this
session), not real per-example cost.

### Results

| Target | n | in-sample cosine_sim | held-out cosine_sim (n_test) |
|---|---|---|---|
| `block_position` (x) | 150 | 0.306 ± 0.158 | 0.288 ± 0.160 (n=30) |
| `block_position` (x) | **5000** | 0.294 ± 0.166 | **0.300 ± 0.164** (n=1000) |
| `block_angle` | **5000** | 0.193 ± 0.347 | **0.192 ± 0.335** (n=1000) |

In-sample ≈ held-out at n=5000 for both targets → not overfitting, this is
the real, well-estimated ceiling for a single global linear direction.
`block_position` direction is recovered noticeably better than
`block_angle`'s, consistent with its higher probe R².

Outputs: `outputs/steering/delta_dataset_{block_angle,block_position}.npz`
(contains `deltas`, `delta_hs`, `frame_indices`, `coefficients`, `layer_name`).
Job logs: `logs/steer-delta-regression-{24189566,24190717,24192587}.log`.

```bash
# block_angle, 5000 examples (defaults in the job script)
sbatch jobs/steer_delta_regression.sh

# block_position, 5000 examples, x-axis, 10-50px delta range
sbatch --export=ALL,TARGET=block_position,TARGET_DIM_INDEX=0,DELTA_MIN=10,DELTA_MAX=50 \
  jobs/steer_delta_regression.sh
```

---

## 5. PCA / implied-direction diagnostic

Pure numpy, no new compute — reuses the `.npz` from Section 4. For each
example, `implied_direction_i = delta_h_i / delta_i` (what direction would
have perfectly steered *that one* example). Then: (a) uncentered SVD across
all examples — is there a dominant shared direction? (b) compare the fitted
regression coefficients to the simple mean of all `implied_direction_i`.

### Results

| Target | Top PC variance | Cumulative @10 PCs | cos(fit, top PC) | cos(fit, mean direction) | mean dir. norm | typical deviation |
|---|---|---|---|---|---|---|
| `block_position` | 19.8% | 73.2% | 0.74 | **0.996** | 0.045 | 0.137 (**3.0x**) |
| `block_angle` | 26.5% | 70.0% | 0.94 | **0.995** | 3.42 | 12.86 (**3.8x**) |

Same story for both targets: no low-rank subspace explains the variation
(need ~10 PCs for 70%); the regression direction is essentially exactly the
population mean direction (cos≈0.995-0.996, this is what no-intercept OLS
mathematically converges to); but any *individual* frame's true response
deviates from that mean by 3-4x the mean's own size. The ceiling is
structural (frame-specific context dominates), not a fitting problem.

```bash
python -c "
import numpy as np
data = np.load('outputs/steering/delta_dataset_block_angle.npz')  # or _block_position
deltas, delta_hs, coefficients = data['deltas'], data['delta_hs'], data['coefficients']
implied = delta_hs / deltas[:, None]
_, S, Vt = np.linalg.svd(implied, full_matrices=False)
evr = (S**2) / np.sum(S**2)
print('top-10 PC variance ratio:', evr[:10], 'cumulative:', np.cumsum(evr[:10]))
mean_dir = implied.mean(axis=0)
coef_unit, mean_unit = coefficients/np.linalg.norm(coefficients), mean_dir/np.linalg.norm(mean_dir)
print('cos(fit, mean):', abs(coef_unit @ mean_unit))
print('mean norm:', np.linalg.norm(mean_dir), 'typical deviation:', np.linalg.norm(implied-mean_dir, axis=1).mean())
"
```

---

## 6. Why the teammate's numbers looked much better (metric, not method)

Teammate's `compare_steering_embeddings.py` → `compare_embeddings()` computes
`cosine_similarity(steered, synthetic)` on **full embeddings**
(`steered = h_real + v`, `synthetic = h_real + δh`), not on the delta
vectors. Since `‖h_real‖ ≈ 8` dominates `‖v‖ ≈ 1.3` and `‖δh‖ ≈ 3.9`, the
shared base term swamps the metric regardless of whether `v` and `δh` are
aligned.

**First-order approximation** (assuming `v, δh` uncorrelated with `h`'s
direction): with `H=‖h‖`, `ε₁=‖v‖`, `ε₂=‖δh‖`, `ρ=cos(v,δh)`,
```
cos(h+v, h+δh) ≈ 1 - (ε₁²+ε₂²)/(2H²) + (ε₁ε₂/H²)·ρ
```
`ρ` only enters as a *second-order* correction. Verified on our own real
data (200 examples from the block_position dataset):

| Metric | Value |
|---|---|
| delta-vs-delta cosine_sim (`ρ`, ours) | 0.293 |
| full-embedding cosine_sim (theirs) | **0.863** |
| approximation predicts | 0.888 (close, expected residual from higher-order terms) |

Recommend asking the teammate to also report delta-vs-delta cosine_sim on
their setup — almost certainly lands in the same ~0.1-0.3 range.

```bash
python -c "
import h5py, numpy as np
data = np.load('outputs/steering/delta_dataset_block_position.npz')
deltas, delta_hs, frame_indices, coefficients = data['deltas'], data['delta_hs'], data['frame_indices'], data['coefficients']
layer_index = int(str(data['layer_name']).split('_')[1])
idx = np.arange(200)
rows = frame_indices[idx]; order = np.argsort(rows)
with h5py.File('/scratch-shared/orinxAI/embeddings/pusht_encoder_cls_fp32.h5', 'r') as f:
    h_reals = f['encoder_cls_layers'][rows[order], layer_index, :][np.argsort(order)]
for i in idx[:1]:
    pass  # see chat log for the full loop; full_cos ~0.86 vs delta_cos ~0.29
"
```

---

## 7. Direction-granularity classification

**Script**: `direction_classification.py`. Bin deltas into N classes (sign ×
magnitude quantile, N even), fit multinomial logistic regression on
`delta_h → bin`, sweep N.

### Results

| N bins | block_position acc. (chance, ×chance) | block_angle acc. (chance, ×chance) |
|---|---|---|
| 2 | 99.7% (50%, **1.99x**) | 78.0% (50%, **1.56x**) |
| 4 | 89.5% (25%, 3.58x) | 56.3% (25%, 2.25x) |
| 6 | 77.7% (16.7%, 4.66x) | 40.9% (16.7%, 2.45x) |
| 8 | 64.1% (12.5%, 5.13x) | 30.9% (12.5%, 2.47x) |
| 10 | 57.6% (10%, **5.76x**) | 27.5% (10%, **2.75x**) |

Both targets: absolute accuracy drops with finer bins (expected, harder
task) but accuracy-over-chance climbs monotonically — real information
keeps showing up well past binary granularity. `block_position` is
substantially easier than `block_angle` throughout, consistent with every
earlier result.

Outputs: `outputs/steering/direction_classification/{block_angle,block_position}/direction_classification.{csv,json,png}`

```bash
python -m interp_utils.steering.direction_classification \
  --dataset outputs/steering/delta_dataset_block_position.npz \
  --num-bins-list 2 4 6 8 10 \
  --output-dir outputs/steering/direction_classification/block_position
```

---

## 8. Binary separability battery

**Script**: `binary_separability.py`. Varied binary tests on the same data:
sign (full range), magnitude-only (sign-agnostic), sign restricted to
smallest/largest magnitude tercile, and a **shuffled-label negative
control**.

### Results

| Experiment | block_position acc. (×chance) | block_angle acc. (×chance) |
|---|---|---|
| `sign_all` | 99.7% (1.99x) | 78.0% (1.56x) |
| `magnitude_above_below_median` (sign-agnostic) | **58.9%** (1.18x) | 64.8% (1.30x) |
| `sign_small_deltas_only` (bottom third) | **100.0%** (2.00x) | 74.7% (1.49x) |
| `sign_large_deltas_only` (top third) | 100.0% (2.00x) | 81.5% (1.63x) |
| `sign_shuffled_control` (negative control) | 51.6% (1.03x) | 51.5% (1.03x) |

Key takeaways:
- **Shuffled control ≈ chance for both targets** → methodology is sound,
  high accuracies elsewhere aren't an artifact.
- **Direction (sign) >> magnitude (sign-agnostic)** for `block_position`
  (99.7% vs 58.9%) — the embedding encodes *which way* far more reliably
  than *how far*. Less pronounced for `block_angle` (78.0% vs 64.8%).
- **`block_position` direction survives even at the smallest tested
  magnitudes** (100% even on the bottom third) — the earlier SNR/noise-floor
  worry about small deltas doesn't hold for *direction* specifically.
- **`block_angle` is uniformly harder than `block_position`** across every
  experiment in this battery, mirroring every other result in this
  document.

Outputs: `outputs/steering/binary_separability/{block_angle,block_position}/binary_separability.{csv,json}`

```bash
python -m interp_utils.steering.binary_separability \
  --dataset outputs/steering/delta_dataset_block_position.npz \
  --output-dir outputs/steering/binary_separability/block_position
```

---

## 8b. Classifier decision-direction as a steering vector (block_angle only)

**Script**: `classifier_direction.py`. A third way to derive a direction,
tried specifically for rotation: fit a left/right logistic-regression
classifier on `delta_h`, take its decision-boundary normal vector as a
candidate steering direction, scale by `alpha`, and check agreement with
the true empirical delta on held-out data.

**Math note**: cosine similarity is invariant to any `alpha > 0` (scaling
never changes direction) — only the L2-norm-of-difference (magnitude
agreement) depends on `alpha`. Reported cosine_sim once; swept `alpha` only
for the L2 curve, including the closed-form optimum
(`alpha* = mean[sign(delta_i) * (w_unit . delta_h_i)]`, derived by zeroing
the derivative of the mean squared L2 error).

### Result

| | value |
|---|---|
| Classifier held-out sign accuracy | 78.0% (matches `binary_separability.py`'s `sign_all`) |
| Direction agreement (cosine_sim, alpha-invariant) | **0.043 ± 0.052** |
| Optimal alpha | 0.177 |
| L2 error across alpha sweep | 3.317-3.329 (nearly flat — expected when cosine_sim≈0) |

**Surprising negative result**: the classifier discriminates left/right well
(78%, far above chance) but its decision direction is a *worse* steering
vector than direct delta regression (0.043 vs 0.192) — even slightly worse
than probe-inversion's -0.025 in absolute terms (though wrong-signed there).
Logistic regression optimizes for class separability given the data's
covariance structure, which is a different objective from "point toward
the true average delta" — good discrimination does not imply good
generation/steering here.

Outputs: `outputs/steering/classifier_direction/block_angle/classifier_direction_{alpha_sweep,summary}.{csv,json}`,
`.png`.

```bash
python -m interp_utils.steering.classifier_direction \
  --dataset outputs/steering/delta_dataset_block_angle.npz \
  --output-dir outputs/steering/classifier_direction/block_angle
```

---

## 8c. Conditioning the direction on the true base state (block_angle only)

**Script**: `conditional_direction.py`. Tests the `v(Δ, h_base)` hypothesis
directly and concretely: bin examples by their TRUE base state (the actual
`block_angle` value at the start frame, read from the dataset, not the
embedding), fit a SEPARATE delta-regression direction per bin using only
that bin's data, and compare against the single GLOBAL direction —
evaluated on the *same* held-out examples for both, so the comparison is
apples-to-apples (one global 80/20 split, bins are just subsets of it).

Job script: `jobs/steer_conditional_direction.sh` (CPU-only, `rome`
partition — this is pure numpy/sklearn, no GPU needed; submitted as a
SLURM job rather than run on the login node because the full state-column
read was badly stalled by login-node filesystem contention, see job
24193509's notes below).

### Result (4 bins, quantile-split on base block_angle, n=5000)

| Bin | Base angle range (rad) | Global cosine_sim | Local cosine_sim | Improvement |
|---|---|---|---|---|
| 0 | [0.00, 0.75) | 0.280 | 0.426 | +0.146 |
| 1 | [0.75, 1.09) | 0.523 | 0.544 | +0.020 |
| 2 | [1.09, 3.54) | **-0.040** | 0.185 | **+0.225** |
| 3 | [3.57, 6.28) | **-0.065** | 0.182 | **+0.247** |

**The pooled global cosine_sim (0.192/0.167 depending on split) was masking
real structure**: the global direction is actually anti-aligned with the
true effect across roughly 70% of the angle range (bins 2-3 combined),
while doing reasonably well in the rest (bins 0-1). Conditioning on the
base angle helps in every bin and helps *most* exactly where the global
model was failing — both negative bins flip positive under the local fit.
Confirms the PCA-diagnostic's large per-example deviation (Section 5) is at
least partly explained by base-state dependence, not just unstructured
noise. Local fits still only reach ~0.18-0.54 (better, not great) — finer
binning or conditioning on more state dimensions is the natural next step,
not yet done.

Outputs: `outputs/steering/conditional_direction/block_angle/conditional_direction.{csv,json}`.
Job log: `logs/steer-conditional-direction-24193509.log`.

```bash
sbatch jobs/steer_conditional_direction.sh
# or override: sbatch --export=ALL,NUM_BINS=6 jobs/steer_conditional_direction.sh
```

---

## 8d. Orientation (quadrant) classification and opposite-quadrant delta regression

A simpler reframing of the probing question, suggested as: forget exact
angle regression, forget deltas — is the block's *current orientation*
(which way the T points; split `[0, 2*pi)` into 4 quadrants) linearly
separable from a single frame's cached embedding? No rendering or
perturbation needed, just the existing recorded frames.

**I/O note**: `encoder_cls_layers` is chunked `(1024, 12, 192)` — a few
thousand SCATTERED rows would force decompressing nearly the entire ~21GB
array (same chunk-touching problem as `state`, much bigger). Fixed by
sampling a handful of CONTIGUOUS, chunk-aligned blocks spread across the
file instead (`orientation_classification.py`): each PushT episode is only
~125 frames on average, so one 1024-row block already spans ~8 episodes
and a wide range of angles, while only touching 1-2 chunks. 12 blocks
(~113MB) finished in 1 minute vs. the earlier full-column reads taking
7-15+ minutes.

### Result: quadrant 0 vs quadrant 2 (180° apart, "opposite orientation")

**100.0% held-out accuracy** (n_test=1892, chance=50%). Note quadrants are
unevenly populated in the real data (`[8290, 955, 1170, 1873]` for Q0-Q3 —
Q0 is likely close to the block's spawn orientation), so the majority-class
baseline is ~87.6%, not 50% — but 100% still genuinely beats that,
correctly identifying every minority-class example too.

```bash
sbatch jobs/steer_orientation_classification.sh   # defaults to quadrant 0 vs 2
```

### Follow-up: does an opposite-quadrant flip have a clean additive direction?

Generated a fresh `delta_regression.py` dataset with delta sampled near
**π ± 0.25 rad** (instead of the usual 0.05-0.5 rad small-rotation range)
to test whether *this specific, perfectly-classifiable* transformation has
a correspondingly clean linear steering direction.

| Delta range | in-sample cosine_sim | held-out cosine_sim (n=1000) |
|---|---|---|
| Small rotations (0.05-0.5 rad) | 0.193 | 0.192 |
| **Opposite-quadrant (~π)** | **0.013 ± 0.375** | **0.005 ± 0.375** |

**Essentially zero — worse than the small-rotation case, not better.**
Despite quadrant membership being perfectly classifiable, there is no
single additive direction that performs the 180° flip: the actual
embedding-space displacement from "this specific starting angle" to "its
180°-opposite" depends heavily on the starting angle itself (consistent
with the `v(Δ, h_base)` finding in Section 8c), and across the full range
of starting angles those individual displacements don't share a common
direction — large std (0.375) shows individual examples scatter both
positive and negative, averaging to noise.

**This sharpens the discrimination-vs-steering gap first seen in Section
8b** (a classifier good at telling left/right apart was still a poor
steering vector): discriminability and additive-steerability are nearly
opposite concerns. Easy separability comes from two clusters sitting in
*different* directions depending on context, not from a uniform shift — so
the *easier* a distinction is to classify, the *less* likely a single
fixed vector can actually produce that transformation.

Outputs: `outputs/steering/delta_dataset_block_angle_opposite.npz`. Job
log: `logs/steer-delta-regression-24195236.log`.

```bash
sbatch --export=ALL,TARGET=block_angle,DELTA_MIN=2.9,DELTA_MAX=3.4,NUM_EXAMPLES=5000,\
OUTPUT="$HOME/Interpretability-for-LeWorldModel/outputs/steering/delta_dataset_block_angle_opposite.npz" \
  jobs/steer_delta_regression.sh
```

---

## 8e. Goal-relative orientation (real task data, not synthetic perturbations)

Every other experiment in this doc uses SYNTHETIC, single-variable
perturbations (render a frame differing only in the target dim). This one
instead uses REAL task-relevant pairs: `eval.py` confirms the env's goal is
always the same episode's frame `goal_offset_steps` ahead
(`goal_offset_steps: 25` in every `config/eval/*.yaml`, exactly what
`world.evaluate(..., goal_offset=...)` uses for the actual PushT task).
Question: does `delta_h_real = h(goal) - h(current)` encode which way the
block still needs to rotate to reach its real task goal?

Reuses the same chunk-aligned block-sampling trick as Section 8d's
orientation classification (`goal_relative_orientation.py`) -- since
`goal_offset=25` is tiny relative to the 1024-row chunk size, `(s, s+25)`
pairs land in the same chunk; episode boundaries within a block are
filtered out via `episode_idx`.

### Result

| | Value |
|---|---|
| Real (current, goal) pairs sampled | 9140 |
| Typical goal-relative rotation | median\|Δangle\|=0.226 rad (~13°), mean\|Δangle\|=0.373 rad (~21°) |
| Label balance (CW vs CCW needed) | 3485 / 5655 (61.9% majority class) |
| **Held-out accuracy** | **95.6%** (n_test=1828, chance=50%, majority-baseline=61.9%) |

Beats every synthetic-perturbation sign-classification result for
`block_angle` (78.0% in Section 8, on rendered single-axis perturbations of
comparable magnitude). **Caveat**: unlike the synthetic experiments, this
`delta_h` isn't isolating orientation alone -- over a real 25-step horizon,
agent position, block position, and velocity are all changing
simultaneously too, so part of the 95.6% could reflect correlated
real-world structure (e.g. a pushing strategy systematically associated
with which way the block ends up needing to rotate) rather than purely
orientation-direction information in isolation. Still a genuinely positive,
task-grounded signal: whatever the single-direction steering limitations
are (Section 8d), the model's real-task embeddings clearly carry strong
information about which way things still need to rotate to reach goal.

Outputs: job log `logs/steer-goal-relative-orientation-24195408.log`.

```bash
sbatch jobs/steer_goal_relative_orientation.sh
```

### Follow-up: can the CURRENT frame alone predict it (no goal frame at all)?

Same labels (which way to rotate to reach the real 25-step-ahead goal), but
features = `h(current)` only, via `--feature current`
(`goal_relative_orientation.py` supports both `delta` and `current` feature
modes).

| Feature | Held-out accuracy |
|---|---|
| `delta` = `h(goal) - h(current)` | 95.6% |
| **`current` = `h(current)` alone** | **92.2%** |

Almost all of the predictive power is already in the current frame by
itself -- the goal frame adds only ~3.4 points on top of 92.2%. Strong
evidence that PushT uses a fixed (or tightly constrained) target pose, so
"which way to rotate to reach goal" is close to a deterministic function of
the *current* orientation relative to that fixed target, and the encoder
represents that relationship well with no lookahead needed.

```bash
sbatch --export=ALL,FEATURE=current jobs/steer_goal_relative_orientation.sh
```

### Follow-up: does the concatenated all-layer representation (Section 8f) help here too?

`goal_relative_orientation.py` now also supports `--concat-layers` (all 12
layers, 2304-dim, same flattening as `concat_layer_probe.py`) instead of a
single `--layer-index`.

| Feature | layer_09 only | concat_layers (2304-dim) |
|---|---|---|
| `current` alone | 92.2% | **97.4%** |
| `delta` (current+goal) | 95.6% | **98.5%** |

Concatenation improves both modes meaningfully, consistent with Section
8f's continuous-regression result. The gap between current-only and
delta-based also *narrows* with the richer representation (1.1 points
apart vs. 3.4 at layer_09 alone) -- with more information available, the
current frame alone captures even more of what's needed, leaving the goal
frame even less unique signal to add. Further strengthens the fixed/
near-fixed target-pose interpretation.

```bash
sbatch --export=ALL,FEATURE=current,CONCAT_LAYERS=1 jobs/steer_goal_relative_orientation.sh
sbatch --export=ALL,FEATURE=delta,CONCAT_LAYERS=1 jobs/steer_goal_relative_orientation.sh
```

### Follow-up: does the classifier's decision direction actually align with the true delta_h?

Mirrors Section 8b's diagnostic (classifier accuracy vs. steering-direction
quality) but on this REAL goal-relative data instead of synthetic
perturbations. `goal_relative_orientation.py --feature delta` now also
reports `cosine_sim(sign(delta_angle) * w_unit, delta_h)` for the already-
fitted classifier's decision direction `w_unit` (alpha-invariant, same
diagnostic as `classifier_direction.py`).

| | Held-out accuracy | Classifier-direction cosine_sim vs. true delta_h |
|---|---|---|
| Synthetic small rotations (Section 8b, layer_09) | 78.0% | 0.043 |
| Real goal-relative, layer_09 | 95.6% | **0.051** |
| Real goal-relative, concat_layers | **98.5%** | **0.032** |

**The discrimination-vs-steering gap holds on real data too, and gets
worse with the richer representation, not better.** Even at near-perfect
classification accuracy, the decision direction stays essentially
orthogonal to the true embedding delta (cosine_sim ~0.03-0.05 throughout --
barely different from the synthetic case's 0.043). Going from layer_09 to
concat_layers *increases* accuracy (95.6% -> 98.5%) while *decreasing*
direction agreement (0.051 -> 0.032): more dimensions give the classifier
more freedom to find an easily-separating direction that is even less
aligned with the actual average displacement. Classification ease and
steering-direction quality aren't just unrelated here -- they can trade off
against each other.

```bash
sbatch --export=ALL,FEATURE=delta jobs/steer_goal_relative_orientation.sh
sbatch --export=ALL,FEATURE=delta,CONCAT_LAYERS=1 jobs/steer_goal_relative_orientation.sh
```

## 8f. Concatenated-layer probing (does combining layers beat the best single layer?)

A side investigation, not steering per se: every probe in this doc picks
ONE layer (`layer_09`, the best-probing-R² layer for `block_angle`/
`block_position`; `layer_10` for `agent_position`). Does concatenating all
12 layers' CLS tokens into one 2304-dim vector and probing that beat the
best single layer?

**Deliberately NOT an extension of the teammate's `probing.py` pipeline**
(torch closed-form OLS, episode-held-out splits, writes to
`probes/<target>/`) — built instead as a lightweight sklearn script
(`concat_layer_probe.py`) matching this doc's other "is there something
here" scripts, reusing the same chunk-aligned block-sampling trick. Caveat:
uses a simple random train/test split on ~30k block-sampled examples, not
the teammate pipeline's strict episode-held-out split on the full 1.6M+
frames — some optimism is possible if test frames share an episode with
train frames.

### Result: raw 2304-dim features + Ridge vs. best single layer

| Target | Best single layer R | concat_layers mean_pearson_r |
|---|---|---|
| `block_angle` | 0.933 (layer_09) | **0.984** |
| `block_position` | 0.996 (layer_09) | **0.9997** |
| `agent_position` | 0.984 (layer_10; layer_09 alone is only 0.801) | **0.997** |

Concatenation beats the best single layer for every target, most
noticeably for `block_angle` (0.933 -> 0.984) -- cross-layer information is
genuinely complementary, not redundant.

### Follow-up: PCA(n_components) + LinearRegression sweep

Same train/test split, PCA fit on train only, sweeping `n_components` in
[10, 25, 50, 100, 300, 600, 1200] out of 2304:

| n_components | explained_var | block_angle r | block_position r | agent_position r |
|---|---|---|---|---|
| 10 | 66.2% | 0.578 | 0.898 | 0.654 |
| 25 | 85.7% | 0.840 | 0.973 | 0.904 |
| 50 | 93.6% | 0.920 | 0.992 | 0.944 |
| 100 | 97.5% | 0.941 | 0.996 | 0.968 |
| 300 | 99.7% | 0.965 | 0.998 | 0.987 |
| 600 | 99.95% | 0.975 | 0.999 | 0.993 |
| 1200 | 100.0% | 0.978 | 0.9993 | 0.994 |
| raw 2304-dim Ridge | -- | **0.984** | **0.9997** | **0.997** |

`block_position` compresses cleanly: 300 components (13% of the original
dims) already matches the raw-feature result (r=0.998 vs 0.9997). Most of
its signal lives in the high-variance directions PCA naturally keeps
first.

`block_angle` and `agent_position` do NOT fully compress, even at 1200
components (52% of dims) -- both plateau measurably below the raw-feature
baseline (0.978 vs 0.984; 0.994 vs 0.997). A meaningful share of their
target-relevant signal lives in lower-variance directions that PCA
(unsupervised, variance-ordered) deprioritizes -- the same "the directions
that matter aren't the dominant directions" theme as Sections 5 and 8.
`block_angle` is again the hardest/least-compressible of the three,
consistent with every other result in this doc.

Job logs: `logs/steer-concat-layer-probe-{24195802,24195803,24195804}.log`.

```bash
sbatch --export=ALL,TARGET=block_angle jobs/steer_concat_layer_probe.sh
# PCA_COMPONENTS and NUM_BLOCKS are also overridable via --export
```

---

## 9. Conclusion: what worked, what didn't, is anything promising?

**Every single fixed, global, additive steering vector tried is bad** —
not a borderline call:

| Method | cosine_sim vs. true delta |
|---|---|
| Probe-inversion (`steering_vector`) | -0.025 (angle), 0.107 (position) |
| Classifier decision boundary | 0.032-0.043 |
| Global delta-regression (best of the four) | 0.19 (angle), 0.30 (position) |
| Delta-regression on opposite-quadrant (~180°) rotations | 0.005-0.013 |

None clear a bar you'd trust for closed-loop control. The opposite-quadrant
result is the sharpest illustration: the exact transformation that's
*easiest* to classify (100% accuracy) is the one where a single vector
fails *hardest* (Section 8d) -- and this pattern repeats on real task data
too (Section 8e follow-up), getting *worse*, not better, with a richer
representation.

**Binary/coarse classification works great (78-100% across every
variant tried) but is a different problem, not a softer version of the
same one.** It proves the information exists and is linearly decodable; it
says nothing about whether an injectable vector exists that produces that
information's effect. We confirmed these are genuinely different objectives
twice (synthetic data, Section 8b; real data, Section 8e follow-up) --
classification ease and steering-direction quality can trade off against
each other, not just fail to correlate.

**Root cause, confirmed not hypothesized**: (1) per-example true response
deviates from the population-mean direction by 3-4x the mean's own size
(Section 5) -- there is no shared direction to find; (2) the direction is
state-dependent, `v(Δ, h_base)` not `v(Δ)` -- the global block_angle
direction is actively anti-aligned across ~70% of the angle range, not
uniformly weak (Section 8c).

**Is anything promising?** One thing: state-conditioning
(`conditional_direction.py`, Section 8c) is the only result in this whole
investigation that *fixed* a demonstrated failure rather than just being
weak everywhere -- splitting into 4 crude quantile bins and fitting a
direction per bin flipped both anti-aligned bins positive and improved
every bin. Not solved (local fits cap at 0.18-0.54), but a real, positive,
reproducible response to a sensible change. Every other approach tried
(pseudoinverse, independent-sum, classifier-direction, opposite-quadrant,
raw concatenation) is a dead end for actual steering, even though several
are valuable for *understanding* what the model represents. If continuing
this line of work, a real `v(Δ, h_base)` model -- kernel-weighted local
regression, or a small learned function of base state, rather than 4 hard
bins -- is the one lead worth pushing to its conclusion.

---

## 10. Open items / not yet done

- `agent_position` has a probe (`/scratch-shared/orinxAI/stable-wm-data/probes/agent_position/`) but hasn't been run through any of Sections 3-8 yet.
- `block_position` y-axis (`--target-dim-index 1`) hasn't been tested — only x so far.
- Joint (Δx,Δy) position perturbations (true "quadrant" classification, as opposed to single-axis sign) would need a new `delta_regression.py` run generating 2D deltas — not built yet.
- A layer sweep (we've only ever used `layer_09`, the best-probing-R² layer) for any of the above — untested whether a different layer has a cleaner direction even with worse raw probe R².
- `conditional_direction.py` only conditions on the target's own base value (block_angle) with coarse (4-bin) quantile splits. Finer bins, conditioning on additional state dims (e.g. block position, agent position) jointly, or a smooth/kernel-weighted local model instead of hard bins are all natural follow-ons — local fits still only reach ~0.18-0.54, so there's more structure left to capture.
- The `Steering` block in `poster.tex` has the method formulas; the empirical findings above (Sections 3-8) aren't reflected in the poster yet.

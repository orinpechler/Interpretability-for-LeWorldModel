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
| `jobs/steer_open_loop.sh`, `steer_closed_loop.sh`, `steer_aggregate.sh`, `steer_delta_regression.sh` | SLURM job wrappers, GPU (`gpu_a100`) |
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

## 9. Open items / not yet done

- `agent_position` has a probe (`/scratch-shared/orinxAI/stable-wm-data/probes/agent_position/`) but hasn't been run through any of Sections 3-8 yet.
- `block_position` y-axis (`--target-dim-index 1`) hasn't been tested — only x so far.
- Joint (Δx,Δy) position perturbations (true "quadrant" classification, as opposed to single-axis sign) would need a new `delta_regression.py` run generating 2D deltas — not built yet.
- A layer sweep (we've only ever used `layer_09`, the best-probing-R² layer) for any of the above — untested whether a different layer has a cleaner direction even with worse raw probe R².
- `conditional_direction.py` only conditions on the target's own base value (block_angle) with coarse (4-bin) quantile splits. Finer bins, conditioning on additional state dims (e.g. block position, agent position) jointly, or a smooth/kernel-weighted local model instead of hard bins are all natural follow-ons — local fits still only reach ~0.18-0.54, so there's more structure left to capture.
- The `Steering` block in `poster.tex` has the method formulas; the empirical findings above (Sections 3-8) aren't reflected in the poster yet.

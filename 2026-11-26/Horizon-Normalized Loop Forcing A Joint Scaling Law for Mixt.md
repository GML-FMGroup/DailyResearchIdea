# Horizon-Normalized Loop Forcing: A Joint Scaling Law for Mixture-of-Loops Attention

## Motivation

Efficient-transformer mechanisms (MoLA in Mixture-of-Loops Transformer, token merging in 3D point-cloud transformers, GPTAQ quantization, TSSA linear attention, Radar decoding) each report an efficiency-quality win at a single scale, but none establishes a scale-invariant relation between its efficiency knob and downstream quality, so no design rule transfers across parameter counts. The Mixture-of-Loops Transformer is the concrete instance: its loop-forcing horizon is fixed a priori and its U-shaped memory-construction/readout separation is validated only at 700M–3B (400B-token pretraining), leaving the structural relation between loop depth L, KV-reuse ratio r, and quality Q unestablished — structurally because a fixed horizon couples the training-time loop budget to the model width/depth, so the same L means different effective reuse as N grows.

## Key Insight

The loop-forcing horizon is not a free hyperparameter but a width/depth-dependent quantity: normalizing it to the model's effective residual depth makes the KV-reuse ratio r a scale-invariant coordinate, so the efficiency-quality frontier of MoLA becomes a single surface Q(L, r, N) that can be fit and extrapolated rather than re-tuned per scale.

## Method

**What it is.** HNLF (Horizon-Normalized Loop Forcing) is a training-and-fitting procedure for Mixture-of-Loops Transformer that (i) replaces the fixed loop-forcing horizon with a horizon schedule H(N) = ceil(c · d_eff(N)), where d_eff is the number of distinct residual-stream depths the model can construct and c is a dimensionless constant, and (ii) fits a joint scaling surface Q(L, r, N) over loop depth L, KV-reuse ratio r, and parameter count N. Inputs: a MoLT-KV/U-MoLT configuration (L, layer count, d_model, KV budget) and a target N; outputs: a horizon-normalized training run and a fitted quality predictor with extrapolation to ≥13B.

**Load-bearing assumption (stated explicitly).** The procedure assumes that d_eff(N) = num_layers(N) · L_max(N) is a faithful proxy for the traversable residual-depth manifold, so that holding c fixed across N makes r a scale-free coordinate and a single low-order surface Q(L, r, N) fitted on 0.7B–7B extrapolates to 13B/30B. This assumption is not taken on faith: it is calibrated and verified in Phase 0 below, and the closed-form schedule is retained only if the measured effective depth tracks the structural proxy across the N grid.

**How it works.**
```pseudocode
# Phase 0: calibration of effective residual depth (new, inside existing structure)
for each model size N in grid {0.7B, 1.5B, 3B, 7B, 13B}:
    train a short MoLT-KV probe (10B tokens, same data order as main run)
    measure per-loop gradient norm ||g_l|| for l=1..L_max on a 512-example calibration set
    measure residual-stream covariance rank: effective rank of hidden states at each loop exit
    d_eff_meas(N) = number of loops l with ||g_l|| > 0.1 * ||g_1||   # traversable depth
    record correlation between d_eff_meas(N) and num_layers(N)*L_max(N)
    if correlation < 0.8: flag and use d_eff_meas(N) in H(N) below

# Phase 1: horizon normalization
for each model size N in grid {0.7B, 1.5B, 3B, 7B, 13B}:
    d_eff(N) = d_eff_meas(N) if Phase 0 correlation >= 0.8 else num_layers(N)*L_max(N)
    H(N)     = ceil(c * d_eff(N))                 # c tuned once on 700M, held fixed
    train MoLT-KV with loop forcing horizon H(N)  # parallel training, single-traversal decode

# Phase 2: efficiency knob sweep
for each (L, r) in grid:
    L in {1,2,4,8};  r = KV_memory_reused / KV_total in {0.25,0.5,0.75,1.0}
    train U-MoLT at horizon H(N); measure Q = downstream avg over held-out tasks

# Phase 3: joint law fit
fit  log Q = a + b·log N + f(L) + g(r) + h(L,r)   # f,g,h low-order poly in log-space
report residuals; extrapolate to 13B and 30B; validate 13B run against prediction
```

**Matched-compute control (sub-block D).** To ensure the extrapolation residual isolates horizon normalization rather than confounds, every fixed-horizon vs. HNLF comparison at a given N uses identical token count (400B), identical data order (fixed seed), identical optimizer (AdamW, β1=0.9, β2=0.95, weight decay 0.1), identical learning-rate schedule (cosine, 2k warmup, peak 3e-4 for 0.7B scaled by 1/sqrt(N/0.7B)), and identical effective compute budget (FLOPs matched to within 1% by adjusting batch size or gradient accumulation, not by changing tokens). Any residual gap is then attributable to H.

**Why this design.** We chose a closed-form horizon schedule H(N)=ceil(c·d_eff(N)) over a per-scale tuned horizon because the loop-forcing horizon's job is to cover the residual-depth manifold the model can traverse, and that manifold grows with depth, not with parameter count alone — accepting the cost that c must be calibrated once and may be suboptimal for very wide/shallow models. We chose MoLT-KV with U-shaped memory-construction/readout separation over MoLT-R because U-MoLT skips unused decoder traversals without changing predictions, so the L axis is a genuine compute knob rather than a quality knob, accepting that MoLT-R's residual-stream retention is not exercised. We chose a low-order polynomial in log-space over a neural scaling-law predictor because the grid is small (L×r×N ≈ 4×4×5) and a neural fit would be unidentifiable, accepting that the fitted surface cannot capture sharp phase transitions. A domain expert would not describe this as 'MoLT + scaling laws' because the load-bearing object is the horizon normalization that makes r scale-invariant; without it the fitted surface is per-scale and does not extrapolate, which is exactly the gap the Mixture-of-Loops Transformer leaves open. The Phase 0 calibration is what turns d_eff from a chosen formula into a measured quantity, and the matched-compute control is what makes the extrapolation residual a clean test of H rather than of compute.

**Why it measures what we claim.** `d_eff(N) = num_layers·L_max` measures *effective residual depth* because we assume each loop adds a distinct residual-stream depth up to L_max; this assumption fails when loops collapse onto the same residual subspace (observed as near-zero gradient on loop>1), in which case d_eff overstates the traversable manifold and H(N) over-allocates horizon — Phase 0's per-loop gradient-norm threshold (||g_l|| > 0.1·||g_1||) and residual-covariance effective rank are the concrete checks for this failure, and if the correlation between d_eff_meas and num_layers·L_max falls below 0.8 we switch to the measured value. `r = KV_memory_reused / KV_total` measures *KV-memory reuse ratio* because we assume KV entries reused across loops are the dominant memory cost; this fails when attention sparsity makes non-reused KV dominate, in which case r understates the true reuse — we log the fraction of attention mass on reused vs. non-reused KV per layer to detect this regime. `Q` measured as downstream average measures *efficiency-quality frontier position* because we assume the held-out task mix is representative of the pretraining distribution; this fails under distribution shift, in which case the fitted surface reflects in-distribution quality only.

## Contribution

['A horizon-normalized loop-forcing schedule H(N)=ceil(c·d_eff(N)) that makes the KV-reuse ratio a scale-invariant coordinate of MoLA training, replacing the fixed loop-forcing horizon inherited from Mixture-of-Loops Transformer.', 'A fitted joint scaling surface Q(L, r, N) over loop depth, KV-reuse ratio, and parameter count, with an extrapolation test at ≥13B that probes whether the U-shaped memory-construction/readout separation preserves its frontier beyond the 3B validation ceiling.', 'An open grid of (L, r, N) training runs and fitted coefficients as a reusable design rule for allocating loop depth and KV-reuse budget at a target scale.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 400B-token pretraining corpus, held-out task suite (same mix as MoLT paper) | Matches MoLT pretraining scale and task mix |
| Primary metric | Downstream average quality Q at fixed compute (FLOPs matched to within 1%) | Efficiency-quality frontier position is the claim |
| Baseline | OLMo3 at 700M/1.5B dense, 3B MoE | Establishes the frontier MoLT must beat |
| Baseline | Fixed-horizon MoLT-KV (no H(N) schedule, H held at 700M value) | Tests whether horizon normalization drives extrapolation |
| Baseline | Per-scale tuned horizon MoLT-KV (grid search over H at each N) | Tests closed-form schedule against oracle tuning |
| Ablation-of-ours | HNLF with r held at 1.0 (no KV reuse sweep) | Isolates the r scale-invariance contribution |
| Calibration check | Phase 0 probe: 10B-token short runs at each N, per-loop gradient norm and residual covariance rank | Verifies d_eff proxy tracks measured traversable depth |

### Why this setup validates the claim

The central claim is that horizon normalization is what makes the (L, r, N) quality surface extrapolable across scale, so the design must separate "MoLT is good" from "horizon normalization is what makes it scale." OLMo3 tests the first sub-claim: if HNLF does not beat OLMo3 at matched token budget, the whole frontier argument collapses and nothing downstream matters. The fixed-horizon baseline is the load-bearing comparison — it is MoLT-KV with the identical architecture and sweep but H held constant, so any gain on the extrapolation check (predicting 13B from the 0.7B–7B fit) is attributable to the schedule rather than to MoLT itself. The per-scale-tuned baseline tests whether the closed-form H(N)=ceil(c·d_eff(N)) is actually competitive with a horizon chosen by search at each scale; if per-scale tuning wins by a wide margin, the closed-form schedule is a convenience, not a mechanism. The r=1.0 ablation isolates the scale-invariance claim about r: if the fitted surface extrapolates just as well without sweeping r, then r is not the scale-invariant knob we claim and the joint-law framing is wrong. The Phase 0 calibration check is the direct test of the load-bearing assumption: if the measured effective depth does not track num_layers·L_max (correlation < 0.8), the closed-form schedule is replaced by the measured value, and the experiment reports which proxy was used. Downstream average Q at fixed compute is the right metric because the claim is about frontier position, not raw quality — a model that is more accurate per FLOP but worse per token has not moved the frontier. The extrapolation residual (predicted vs. measured 13B) is the single number that most directly tests the thesis, and the matched-compute control (identical tokens, data order, optimizer, and FLOPs) ensures that residual reflects H rather than confounds.

### Expected outcome and causal chain

**vs. OLMo3** — On a 3B MoE configuration trained at 400B tokens, OLMo3 spends its compute budget on fixed-depth layers with no loop-reuse structure, so at a given FLOP budget it cannot trade depth for KV reuse and its quality is pinned to the layer count. Our method instead trains U-MoLT at horizon H(N) with the (L, r) knob swept, so the same budget can be reallocated toward deeper effective residual traversal where the task mix rewards it, and we expect a modest but consistent Q gap over OLMo3 at 3B that is largest on tasks requiring long-range composition and near parity on shallow local tasks.

**vs. Fixed-horizon MoLT-KV** — On the 7B-to-13B extrapolation check, the fixed-horizon baseline is trained with the same H at every scale, so at 13B its horizon under-covers the residual-depth manifold that the larger layer count and L_max can traverse; the fit is then per-scale and the predicted 13B Q drifts from measured. Our method instead sets H(N)=ceil(c·d_eff(N)), so the horizon grows with the traversable manifold and the fitted surface is anchored in a normalized coordinate, and we expect the fixed-horizon variant to show a growing residual with scale while HNLF's residual stays roughly flat, with the gap concentrated at the largest scale. The matched-compute control ensures this gap is not a compute artifact.

**vs. Per-scale tuned horizon MoLT-KV** — On the 0.7B and 1.5B scales where per-scale search is cheap, the tuned baseline can match or slightly beat the closed-form H(N) because it optimizes the horizon directly for that scale, at the cost of a separate search per model size. Our method instead fixes c once on 700M and holds it, so it gives up a small amount of per-scale optimality in exchange for a schedule that transfers; we expect near-parity at 700M/1.5B and a crossover at 7B/13B where the tuned baseline's search cost becomes prohibitive and its horizon is effectively frozen, leaving HNLF ahead on the extrapolation residual.

### What would falsify this idea

If the extrapolation residual for HNLF is no smaller than the fixed-horizon baseline's, or if the r=1.0 ablation extrapolates just as well, then horizon normalization and the r-sweep are not the mechanisms carrying the scale transfer and the central claim is wrong. A uniform gain across all scales and subsets, rather than a gain concentrated at the largest scale and on long-range tasks, would also indicate the effect is generic MoLT quality rather than the predicted horizon-normalization mechanism. Additionally, if Phase 0 shows the measured effective depth does not track num_layers·L_max (correlation < 0.8) and the measured-value schedule still fails to extrapolate, the structural proxy is falsified.

## References

1. Mixture-of-Loops Transformer

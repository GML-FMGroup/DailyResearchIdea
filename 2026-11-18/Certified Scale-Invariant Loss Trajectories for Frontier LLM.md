# Certified Scale-Invariant Loss Trajectories for Frontier LLM Pretraining

## Motivation

Across the current training-efficiency literature, each branch optimizes a different lever — batch-size schedules (Critical Batch Size Revisited, E1/E8), checkpoint merging (Model Merging in Pre-training, E3/E7), memory-efficient adapters (LoRAM, E2/E11), patch-level tokenization (E4/E9), and pre-training distillation (MiniPLM, E5/E10) — yet none establishes that its intervention preserves the loss trajectory when scale or substrate changes, so efficiency gains are reported per-run and cannot be composed or certified. The frontier report (Xing 4, E6/E12) inherits this gap directly: it states architectural and systems design principles for a 285B Ascend run but reports no loss-trajectory invariant, meaning a practitioner cannot tell whether merging, batch warmup, and adapter training interact benignly at frontier scale or silently trade off against each other. The root cause is structural: efficiency is measured as a point estimate of final loss rather than as a property of the trajectory curve, so there is no object that could be invariant in the first place.

## Key Insight

A training-efficiency intervention is scale-transferable exactly when the ordering it induces on loss trajectories is preserved under a monotone reparameterization of compute, so certification reduces to testing whether the intervention's loss-versus-compute curve is a monotone rescaling of the reference curve rather than a curve crossing it.

## Method

**What it is.** We introduce **CTI (Certified Trajectory Invariant)**, a protocol that consumes a set of loss trajectories from matched training runs at two or more scales and one or more substrates and outputs a per-intervention certificate: *certified* if the intervention's loss curve dominates the reference curve at every matched compute checkpoint under a monotone compute rescaling, *conditional* if dominance holds only beyond a fitted crossover point, and *refuted* otherwise. Inputs are checkpointed loss traces; outputs are certificates plus the fitted rescaling exponent.

**Load-bearing assumption (stated explicitly).** CTI assumes that an efficiency intervention's effect on the loss-vs-compute curve is a single monotone reparameterization of the compute axis (one scalar alpha_c), so that after rescaling the intervention curve and the reference curve are pointwise comparable and dominance can be certified by a sign test. This assumption fails when an intervention changes the loss-curve curvature (e.g., merging creates discontinuities at merge checkpoints, or batch-size warmup changes the effective gradient noise scale over training), in which case a single alpha fitted by least squares on log t cannot represent a curve that is worse early and better late except as an average, which is exactly the curve-crossing case the certificate is supposed to detect. To make this assumption testable, we replace the single global scalar alpha_c with a checkpoint-local rescaling: we fit alpha on sliding windows of 6 consecutive checkpoints (step 3), and we fit a monotone spline warp alpha(t) with 5 knots (placed at the 0th, 25th, 50th, 75th, and 100th percentiles of log t) constrained to be non-decreasing. The certificate becomes a statement about the sign of the derivative of the gap d/dt[L_c(alpha(t)*t) - L_ref(t)] rather than a single global dominance test, so a curve crossing is detected as a sign change in the warp's derivative rather than being averaged away by one alpha. We report both the global alpha_c (for backward comparability) and the local alpha(t) profile.

**How it works.**

```pseudocode
# Phase 1: trajectory extraction
for each (scale s in {small, large}, substrate h in {Ascend, GPU}, config c in {ref, +warmup, +merge, +adapter, +all}):
    record loss L[s,h,c](t) at compute checkpoints t in T = {geometric grid, 24 points, 1e20..full budget}
    normalize t by total tokens processed (not wall-clock), so t is substrate-invariant

# Phase 2: monotone-rescaling fit (per intervention c, per (s,h) pair)
# Global fit (for reporting alpha_c):
fit alpha_c via least squares on log t:  minimize sum_t || L_ref(t) - L_c(alpha_c * t) ||^2
    constrained alpha_c > 0; robust to checkpoint noise via Huber loss (delta=0.01 nats)
# Local fit (for curvature diagnostic):
fit monotone spline warp alpha(t) with 5 knots (percentiles of log t), non-decreasing constraint
fit window-local alpha_w on sliding windows of 6 consecutive checkpoints (step 3)

# Phase 3: dominance test (the certificate)
for each checkpoint t in T:
    delta_c(t) = L_c(alpha(t) * t) - L_ref(t)          # signed gap, nats
    block-bootstrap 1000 resamples over contiguous checkpoint windows of length 4 -> CI on delta_c(t)
    # (block length 4 chosen to respect autocorrelation; i.i.d. resampling replaced)
compute derivative sign: d/dt[delta_c(t)] via finite differences on the spline warp
certify c as CERTIFIED iff upper CI bound of delta_c(t) <= 0 for all t in T AND derivative sign is non-positive everywhere
    certify as CONDITIONAL iff delta_c(t) <= 0 for all t >= t* (t* = fitted crossover) OR derivative sign changes from positive to negative at t*
    else REFUTED

# Phase 4: scale-transfer check
invariant_c = (sign of delta_c(t) agrees across s=small and s=large at every matched t)
certificate_c = CERTIFIED and invariant_c

# Phase 5: joint ablation on Xing 4 285B (future work / industry collaboration)
run configs {ref, +warmup, +merge, +adapter, +all} at 285B on 8192 Ascend 910C
    warmup: batch size 0.5x -> 4x critical batch size (CBS from E1 protocol) over first 15% of tokens
    merge: checkpoint averaging every 10% of tokens with constant LR (E7 recipe)
    adapter: LoRAM-style pruned-model low-rank training (E2), rank 64, 4-bit base
report certificate for each config and for the joint config
```

**Why this design.** We chose a monotone-rescaling fit over a direct per-checkpoint comparison because interventions like batch-size warmup legitimately shift the compute axis (fewer steps to the same loss, E8), and a pointwise test would falsely refute them; accepting the cost that alpha absorbs genuine speedups and cannot distinguish them from curve-shape changes. We chose a nonparametric bootstrap dominance test over a parametric loss-law fit (e.g., Chinchilla-style power law) because power-law fits are known to be brittle at frontier scale and would impose a functional form the certificate should not assume, accepting wider confidence intervals and the need for more checkpoints. We chose to require agreement across scales as a separate gate rather than folding scale into one pooled fit because pooling would let a large-scale improvement mask a small-scale regression, accepting a stricter certificate that many interventions will fail. This differs from the standard practice in Critical Batch Size Revisited (E1), Model Merging in Pre-training (E7), and LoRAM (E2), each of which reports a single-run efficiency delta and would not be described by an expert as a variant of CTI, because none of them defines a cross-scale ordering object at all.

**Why this measures what we claim.** The quantity `delta_c(t)` (signed loss gap at matched compute) measures *trajectory dominance* because the assumption that total tokens processed is a substrate-invariant compute coordinate makes the two curves comparable pointwise; this assumption fails when Ascend and GPU runs differ in numerical precision or optimizer sharding such that the same token count yields different effective update noise, in which case delta reflects substrate mismatch rather than the intervention. The quantity `alpha_c` (fitted rescaling exponent) measures *efficiency transfer* because the assumption that an efficiency gain acts as a monotone time reparameterization makes a single scalar sufficient; this assumption fails when an intervention changes the loss-curve curvature (e.g., merging creates a discontinuity at merge checkpoints), in which case alpha is an average that hides a curve crossing. The quantity `invariant_c` (cross-scale sign agreement) measures *scale-transferability* because the assumption that the small-scale run is in the same training regime as the large-scale run makes sign agreement meaningful; this assumption fails when the small model is below the emergence threshold for the relevant capability, in which case agreement reflects regime coincidence rather than transfer. The quantity `d/dt[delta_c(t)]` (derivative of the gap under the spline warp) measures *curvature invariance* because the assumption that the intervention does not change the loss-curve curvature makes the derivative sign constant; this assumption fails when merging discontinuities or warmup-induced noise-scale changes produce a sign change, in which case the derivative sign change flags a curve crossing that a single alpha would average away.

## Contribution

(1) CTI, a certification protocol that defines a scale-transferable monotone loss-trajectory invariant and issues per-intervention certificates (certified/conditional/refuted) from checkpointed loss traces at matched token counts. (2) A joint-ablation design that isolates batch-size warmup, checkpoint merging, and memory-efficient adapter training as individual and combined interventions on a single frontier run, yielding per-mechanism certificates rather than aggregate efficiency claims. (3) A reusable certificate format and fitting procedure that lets any future training-efficiency intervention be tested for scale transfer without re-running the full frontier budget.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Matched OLMo 1B/7B loss traces on GPU (24 checkpoints, geometric grid 1e20..full budget) + a single 7B Ascend run for substrate floor | Two-scale, two-substrate traces needed for cross-scale gate; 285B moved to future work |
| Primary metric | Certificate verdict per intervention (CERTIFIED/CONDITIONAL/REFUTED) | Discrete ordering object is the claim itself |
| Baseline | Single-run efficiency delta (E1/E7/E2 style) | Tests whether pointwise deltas over-certify |
| Baseline | Parametric power-law loss fit | Tests whether functional-form assumption mis-orders curves |
| Baseline | Pooled cross-scale fit (no separate gate) | Tests whether pooling masks small-scale regressions |
| Ablation-of-ours | CTI without monotone-rescaling (pointwise dominance only) | Isolates the rescaling-fit contribution to verdicts |
| Ablation-of-ours | CTI with global alpha only (no spline warp, no derivative test) | Isolates the curvature diagnostic contribution |
| Substrate control | Identical reference config on Ascend and GPU at 7B, report delta_ref(t) as noise floor | Operationalizes substrate-invariance assumption as a measured quantity |

### Why this setup validates the claim

The central claim is that a cross-scale, cross-substrate ordering object over loss trajectories yields verdicts that single-run efficiency deltas cannot, and that the monotone-rescaling fit is necessary to avoid false refutations of axis-shifting interventions like batch-size warmup. The matched OLMo 1B/7B traces let us test the scale-transfer gate directly, since the same intervention appears at both scales and the certificate must agree. The single 7B Ascend run supplies the substrate-invariance test on Ascend, where numerical precision and sharding differ from GPU runs, so a substrate mismatch would surface as a spurious delta; we measure delta_ref(t) on the identical reference config and require intervention deltas to exceed this floor before certification. The three baselines each probe a distinct sub-claim: the single-run delta baseline tests whether per-run improvements are sufficient to certify, the power-law baseline tests whether imposing functional form changes the ordering, and the pooled-fit baseline tests whether the separate scale gate is load-bearing. The ablation without rescaling strips the rescaling fit so that any verdict change is attributable to that mechanism alone; the ablation with global alpha only strips the spline warp and derivative test so that any verdict change is attributable to the curvature diagnostic. Certificate verdict is the right metric because the claim is about an ordering object, not a scalar loss; a scalar metric would not detect the predicted pattern of conditional-but-not-certified verdicts.

### Expected outcome and causal chain

**vs. Single-run efficiency delta** — On a case where batch-size warmup reaches slightly better final loss with 43% fewer steps, the baseline produces an unqualified "better" verdict because it compares endpoints and ignores curve shape at matched compute. Our method instead fits a monotone rescaling and tests dominance at every checkpoint, so a warmup run that is worse early but better late yields CONDITIONAL rather than CERTIFIED. We expect the baseline to over-certify warmup and merging, while CTI returns CONDITIONAL for both and reserves CERTIFIED for interventions whose dominance holds from the first checkpoint.

**vs. Parametric power-law loss fit** — On a case where merging creates a discontinuity at merge checkpoints, the baseline produces a smooth power-law fit that averages across the jump and reports a misleading efficiency gain, because it imposes a functional form the trajectory violates. Our method instead uses a nonparametric block-bootstrap over contiguous checkpoint windows, so the jump widens the confidence interval at the affected checkpoints rather than being absorbed. We expect the power-law baseline to rank merging above adapter training, while CTI flags merging as CONDITIONAL with a wide interval near merge points.

**vs. Pooled cross-scale fit** — On a case where the 7B run improves but the 1B run regresses under the same intervention, the baseline produces a CERTIFIED verdict because pooling lets the large-scale gain dominate the fit. Our method instead requires sign agreement across scales as a separate gate, so the same intervention is REFUTED. We expect a noticeable divergence on the joint config, where pooling certifies but CTI refutes, and parity on warmup, where both scales agree.

**vs. CTI with global alpha only (no spline warp, no derivative test)** — On a case where merging creates a discontinuity, the global-alpha variant fits a single scalar that averages the jump and returns CERTIFIED, because it cannot represent a curve crossing. Our method instead fits a monotone spline warp and tests the derivative sign of the gap, so the discontinuity produces a sign change and the verdict becomes CONDITIONAL. We expect the global-alpha variant to over-certify merging and warmup, while full CTI returns CONDITIONAL for both, confirming that the curvature diagnostic is load-bearing.

### What would falsify this idea

If CTI's verdicts coincide with the single-run efficiency-delta baseline on every intervention, including warmup and merging, then the cross-scale ordering object adds nothing and the central claim is wrong. A second falsifier: if the ablation without rescaling produces identical verdicts to full CTI, the rescaling fit is not load-bearing and the mechanism story fails.

## References

1. TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks

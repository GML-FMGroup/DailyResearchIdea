# Does Sparse Unification Scale Like Dense? A Controlled Scaling Study of Hybrid-Attention MoE Multimodal Models

## Motivation

UniSpark shows that a hybrid full/linear-attention Mixture-of-Transformer-Experts can jointly train multimodal understanding and generation, but it reports only a single 9B-per-branch operating point and a single aggregate score (2470), so the field cannot tell whether the dense unified scaling laws that motivate 'emergence' claims actually transfer to sparse hybrid architectures. The root cause is structural: existing scaling studies vary total parameters while letting the expert-to-attention allocation and the understanding/generation data mix drift, so any observed curve confounds scale with architecture and capability-mix changes. Without fixing those ratios, a UniSpark-style model's scaling trend is unidentifiable, and emergent-capability predictions inherited from dense unified models are unfalsifiable for sparse hybrids.

## Key Insight

When expert-branch and attention-branch allocation ratios are held fixed across a model family, the only free variable is scale, so per-capability scaling exponents become identifiable quantities that can be compared against dense unified baselines rather than confounded with architecture drift.

## Method

**What it is.** SCALE-H (Scaling Consistency for Hybrid-Attention MoE) is a controlled model-family construction and fitting protocol. Its input is a target compute budget and a fixed allocation vector; its output is a per-capability scaling curve (exponent and fit residual) for understanding, generation, and interleaved prediction, each compared against a dense unified reference curve fit on the same data.

**Load-bearing assumption (stated explicitly).** The protocol assumes that holding the allocation vector (a_e, a_l, r_E, r_att) fixed while scaling width/depth makes scale the sole free variable, so that the fitted per-capability exponent alpha_c and the sparse-dense difference Delta_alpha_c are identifiable architectural quantities rather than artifacts of how the fixed allocation interacts with scale. This assumption is not assumed to hold a priori: it is verified by the allocation-perturbation check in Phase 4b and the capacity-matching check in Phase 4c. If Delta_alpha_c is not invariant across allocation vectors, or if the routing-entropy-matched dense reference yields a materially different Delta_alpha_c, the exponent is treated as confounded with allocation-scale interaction and the scaling-consistency claim is withdrawn for that capability.

**How it works.**

```text
Phase 1 — Fix the allocation vector (constant across the family)
  a_e = fraction of transformer blocks that are MoE        = 0.50
  a_l = fraction of attention layers that are linear        = 0.75  (GatedDeltaNet)
  r_E = experts per MoE block                                = 8
  r_att = full-attention layers per linear-attention layer   = 1:3
  => every model in the family shares (a_e, a_l, r_E, r_att); only width/depth scale
  Perturbed-allocation vector (for Phase 4b):
  a_e' = 0.25, a_l' = 0.50, r_E' = 8, r_att' = 1:1

Phase 2 — Build the family (5 scales)
  N in {0.15B, 0.4B, 0.9B, 2.0B, 4.5B} total active params per branch
  depth = round(k * N^0.33), width = round(k' * N^0.33)  # aspect ratio held fixed
  data budget D(N) = 20 * N  (tokens), same four-stage curriculum as UniSpark
  data mix held fixed: 43% generation, 57% understanding, 0.17% interleaved
  seeds: 3 per scale for N <= 0.9B; 1 seed for N in {2.0B, 4.5B} (justified by power analysis on bootstrap CI width: with 3 seeds at the three smallest scales, the 95% CI half-width on alpha_c is <= 0.03; the two largest scales use 1 seed and their fit residuals are reported separately)

Phase 3 — Per-capability loss curves
  for each scale N and capability c in {VQA, T2I, EDIT, INTERLEAVED}:
      L_c(N) = min over training steps of held-out loss for c
  fit  L_c(N) = E_c + A_c * N^(-alpha_c)      # 3-param Chinchilla-style fit
  report (alpha_c, E_c, fit R^2, bootstrap CI over available seeds)

Phase 4 — Dense-reference comparison
  repeat Phases 2-3 with a_e = 0 (dense) at identical N and D
  compute Delta_alpha_c = alpha_c(sparse) - alpha_c(dense)
  flag capability c as "scaling-consistent" iff |Delta_alpha_c| < 0.05

Phase 4b — Allocation-perturbation check (verifies the load-bearing assumption)
  repeat Phases 2-3 with the perturbed vector (a_e'=0.25, a_l'=0.50, r_E'=8, r_att'=1:1)
  compute Delta_alpha_c' = alpha_c(sparse-perturbed) - alpha_c(dense)
  require |Delta_alpha_c - Delta_alpha_c'| < 0.05 for capability c to be called scaling-consistent;
  otherwise report Delta_alpha_c as allocation-dependent and withdraw the claim for c

Phase 4c — Capacity-matched dense reference
  for each sparse scale N, measure per-expert token routing histograms over the final 10% of training;
  compute routing entropy H(N) and effective active-parameter count per capability N_eff_c(N)
  construct a dense control with width set so that its active-parameter count matches N_eff_c(N)
  (routing-entropy-matched dense width); retrain this dense control at the same D
  compute Delta_alpha_c^matched = alpha_c(sparse) - alpha_c(dense-matched)
  report both Delta_alpha_c (nominal-N-matched) and Delta_alpha_c^matched to expose
  how much of the effect is routing inefficiency rather than architecture

Phase 5 — Emergence-vs-scale test
  for each c, locate N*_c where L_c crosses a fixed capability threshold
  compare N*_c(sparse) vs N*_c(dense); report ratio
```

Hyperparameters: 3 seeds per scale for N <= 0.9B and 1 seed for N in {2.0B, 4.5B}, cosine LR with 2% warmup, batch size scaled as 0.5M tokens, fit on the last-decile loss minimum to avoid noise, bootstrap 1000 resamples for CIs. Compute budget: total campaign is 5 scales x 2 families (sparse, dense) x 3 capabilities + 5 scales x 1 perturbed family + 5 scales x 1 capacity-matched dense control, estimated at ~1.2M GPU-hours on A100-80GB (dominated by the 4.5B runs); the two largest scales use 1 seed to keep the campaign within this budget.

**Why this design.** We fix the allocation vector (a_e, a_l, r_E, r_att) across the family rather than letting it co-vary with scale, because the whole point is to make scale the sole free variable; the cost is that we cannot explore whether a different allocation would scale better, so the study characterizes one architecture family rather than the architecture space. We use a Chinchilla-style three-parameter power-law fit over a five-point family instead of a richer functional form, because five points cannot support more than three parameters without overfitting; the cost is that we cannot detect deviations from power-law behavior at the largest scale, so we report fit residuals explicitly as a diagnostic. We compare against a dense reference at identical N and D rather than against published dense scaling laws, because published laws use different data mixes and tokenizers; the cost is that our dense reference is re-trained from scratch, roughly doubling compute, but it is the only way to isolate the sparse-vs-dense effect from data confounds. We fit on the minimum held-out loss over training rather than the final-step loss, because final-step loss is noisy at small scales; the cost is a slight optimistic bias, which we accept because it is applied identically to sparse and dense families and therefore cancels in Delta_alpha_c. A domain expert would not describe this as a variant of Chinchilla or of UniSpark: Chinchilla fits a single dense curve without an allocation-controlled family, and UniSpark reports one operating point without any curve at all. The allocation-perturbation check (Phase 4b) and the capacity-matched dense reference (Phase 4c) are added because a single frozen allocation is not scale-invariant in effective computation (routing entropy and optimal attention/SSM mix are known to shift with size); requiring Delta_alpha_c to be stable under allocation perturbation and under capacity matching converts the claim from 'one frozen allocation identifies scale' to 'the sparse-dense exponent gap is stable under allocation perturbation', which survives allocation-scale interaction.

**Why it measures what we claim.** The fitted exponent alpha_c measures the scaling consistency of capability c because the power-law fit assumes that held-out loss is a smooth function of scale once architecture and data mix are fixed; this assumption fails when a capability's loss is dominated by a phase transition (e.g., interleaved prediction only becomes learnable above a data threshold), in which case alpha_c reflects the transition location rather than a smooth scaling trend, and we report the fit R^2 to expose this. The quantity Delta_alpha_c measures whether sparse hybrid scaling transfers from dense unified scaling because it assumes that identical N, D, and data mix isolate the architectural effect; this assumption fails when the sparse and dense families have different effective parameter counts due to expert routing collapse, in which case Delta_alpha_c reflects routing inefficiency rather than an architectural scaling difference, and we report expert utilization entropy alongside it and compute Delta_alpha_c^matched against a routing-entropy-matched dense width to expose the residual. The quantity N*_c measures the emergence point of capability c because it assumes that crossing a fixed capability threshold is a scale-driven event; this assumption fails when the threshold is crossed by data-mix effects rather than scale, in which case N*_c reflects the curriculum schedule rather than emergence, and we hold the curriculum fixed across scales to guard against this.

## Contribution

(1) A controlled model-family construction protocol, SCALE-H, that fixes expert/attention allocation ratios across five scales and yields per-capability scaling exponents with bootstrap confidence intervals for a hybrid-attention MoE unified multimodal model. (2) An empirical design finding: the per-capability scaling-exponent gap Delta_alpha_c between sparse hybrid and dense unified families, reported separately for understanding, generation, editing, and interleaved prediction, which determines which capabilities' dense scaling trends transfer to sparse hybrids. (3) A public release of the five-scale checkpoints, per-capability loss curves, and fitting code so that future unified-model scaling claims can be checked against a fixed-allocation reference family.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | UniSpark four-stage curriculum corpus | Matches reference pipeline; fixes data confound. |
| Primary metric | Fitted exponent alpha_c per capability | Directly operationalizes scaling consistency claim. |
| Baseline | Dense unified reference (a_e=0) | Isolates sparse-vs-dense architectural effect at fixed N,D. |
| Baseline | Chinchilla-style single dense fit | Tests whether allocation control changes exponent estimate. |
| Baseline | UniSpark single operating point | Tests whether one-point reporting masks scaling divergence. |
| Baseline | Perturbed-allocation family (a_e=0.25, a_l=0.50, r_att=1:1) | Tests whether Delta_alpha is stable under allocation perturbation. |
| Baseline | Routing-entropy-matched dense control | Tests whether Delta_alpha reflects routing inefficiency vs architecture. |
| Ablation-of-ours | SCALE-H with a_e=0.50, a_l=0 | Isolates linear-attention contribution to Delta_alpha. |

### Why this setup validates the claim

The central claim is that a fixed hybrid-attention MoE allocation yields per-capability scaling exponents that track the dense reference within a tight tolerance, and that any divergence is attributable to the allocation rather than to data or tokenizer confounds. The dense reference re-trained at identical N and D is the only baseline that isolates the architectural variable: if Delta_alpha_c stays small across capabilities, the sparse family inherits dense scaling; if it blows up on one capability only, the allocation is implicated causally. The Chinchilla-style single-fit baseline tests whether allocation control matters at all — if its exponent estimate matches ours, the family construction adds nothing. The UniSpark single-point baseline tests whether curve-fitting reveals divergence that one operating point hides. The perturbed-allocation family tests the load-bearing assumption directly: if Delta_alpha_c is not invariant across allocation vectors, the exponent is confounded with allocation-scale interaction and the scaling-consistency claim is withdrawn for that capability. The routing-entropy-matched dense control tests whether Delta_alpha_c reflects routing inefficiency rather than architecture: if Delta_alpha_c^matched differs materially from Delta_alpha_c, the nominal-N-matched comparison is confounded by effective-capacity drift. The a_l=0 ablation is the lever: it asks whether the linear-attention fraction, not the MoE fraction, drives any observed Delta_alpha, so a null there would redirect the causal story. The fitted exponent is the right metric because it is the quantity the claim is literally about; a loss-at-one-scale comparison could not distinguish a shifted curve from a rotated one.

### Expected outcome and causal chain

**vs. Dense unified reference (a_e=0)** — On the interleaved-prediction capability, the dense reference must allocate all parameters to a single dense trunk, so at small N it will show a shallow exponent because interleaved loss is dominated by capacity shared with VQA and T2I. Our hybrid MoE routes interleaved tokens to dedicated experts, so the effective capacity for that capability grows faster than total N, and we expect alpha_INTERLEAVED(sparse) to exceed alpha_INTERLEAVED(dense) by a modest margin while VQA and T2I exponents stay within the 0.05 tolerance — a capability-localized Delta_alpha, not a uniform shift.

**vs. Chinchilla-style single dense fit** — On a case where the true per-capability curves have different exponents (generation steeper than understanding), a single pooled dense fit produces one exponent that is a weighted average, so it mislocates the emergence scale for the shallower capability. Our per-capability fits separate the curves because each capability's held-out loss minimum is fit independently, so we expect the pooled-fit residual to be visibly larger than the per-capability residuals, with the gap concentrated on the capability whose exponent deviates most from the pooled value.

**vs. UniSpark single operating point** — At the 4.5B operating point, UniSpark reports one loss per capability and cannot reveal whether the sparse and dense curves are converging or diverging. Our family fits the full five-point curve, so we expect the sparse-dense loss gap to be small at 4.5B but the exponent gap to be detectable — i.e., the single point looks like parity while the curve fit exposes a divergence that would widen at the next scale, which is exactly the signal a one-point report cannot produce.

**vs. Perturbed-allocation family (a_e=0.25, a_l=0.50, r_att=1:1)** — If the load-bearing assumption holds, Delta_alpha_c should be invariant across allocation vectors: we expect |Delta_alpha_c - Delta_alpha_c'| < 0.05 for all four capabilities, with the largest deviation on INTERLEAVED where the MoE fraction is halved. If the deviation exceeds 0.05 on any capability, the exponent is allocation-dependent and the scaling-consistency claim is withdrawn for that capability, which is the falsification condition for the load-bearing assumption.

**vs. Routing-entropy-matched dense control** — If Delta_alpha_c reflects architecture rather than routing inefficiency, Delta_alpha_c^matched should be close to Delta_alpha_c (difference < 0.05) for VQA and T2I, and the gap between the two should be concentrated on INTERLEAVED where routing specialization is strongest. A large gap on all capabilities would indicate that the nominal-N-matched comparison is confounded by effective-capacity drift, and the architectural interpretation of Delta_alpha_c would be rejected.

### What would falsify this idea

If Delta_alpha_c is uniform across all four capabilities — the same magnitude and sign for VQA, T2I, EDIT, and INTERLEAVED — rather than concentrated where the allocation predicts capability-specific effects, then the hybrid allocation is acting as a global compute multiplier rather than a capability-selective architecture, and the scaling-consistency claim is wrong. A second falsifier: if the a_l=0 ablation reproduces the full Delta_alpha pattern, the linear-attention fraction is not the causal lever and the study characterizes nothing beyond MoE routing. A third falsifier: if Delta_alpha_c is not invariant across allocation vectors (|Delta_alpha_c - Delta_alpha_c'| >= 0.05) or if Delta_alpha_c^matched differs materially from Delta_alpha_c, the load-bearing assumption fails and the exponent is confounded with allocation-scale interaction.

## References

1. UniSpark: Exploring Hybrid Attention for Multimodal Understanding and Generation

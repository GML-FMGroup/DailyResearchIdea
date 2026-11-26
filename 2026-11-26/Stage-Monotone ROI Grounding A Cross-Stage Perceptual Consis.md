# Stage-Monotone ROI Grounding: A Cross-Stage Perceptual Consistency Regularizer for Progressive Medical MLLM Curricula

## Motivation

Progressive medical MLLM curricula such as Sage-Med (Sage-Med: Scaling Medical Multimodal Intelligence) treat data curation, supervised fine-tuning, and reinforcement learning as separable modules, applying them sequentially and reporting only aggregate end-of-curriculum gains; nothing in the pipeline verifies that region-grounded perception is non-decreasing at each stage transition, so a later stage can silently overwrite ROI localization and description alignment learned earlier while text-side metrics still improve. This is a meta-level failure recurring across the curriculum-style tree: MedTrinity-25M similarly builds multigranular ROI annotations and hands them to a pretraining stage, then reports downstream VQA/report numbers (E1, E7) without any instrument that would reveal perceptual regression at the handoff, and MediConfusion shows that even flagship medical MLLMs are confused by image pairs that are trivially distinguishable to experts (E3, E10) — exactly the region-grounded perception that unverified stage transitions can erode. The root cause is structural: no curriculum stage carries a constraint that ties its ROI-level grounding representation to the one the previous stage achieved, so 'stage-wise gains' are measured on language-heavy aggregates and perceptual drift is invisible by construction.

## Key Insight

If ROI grounding is pinned to a frozen, stage-0 reference head on a held-out probe set and each transition is admitted only when the probe-set grounding agreement is non-decreasing, then perceptual regression becomes an observable, correctable quantity at the transition boundary rather than an invisible side effect buried in aggregate benchmark deltas.

## Method

**What it is.** We introduce **SM-Ground** (Stage-Monotone Grounding), a training-time regularizer and admission gate for progressive medical MLLM curricula. Its inputs are the model's ROI-level grounding outputs (region logits over a fixed patch grid plus region-conditioned description embeddings) on a held-out ROI probe set at the end of each curriculum stage; its outputs are a scalar stage-transition invariant loss added to the next stage's objective and a binary admission decision that blocks a stage transition when perceptual regression is detected.

**Load-bearing assumption (stated explicitly).** SM-Ground assumes that the probe-set grounding score G_s — computed against *external, label-anchored* criteria on a fixed, modality-stratified probe set — is a sufficient statistic for the perceptual capability a stage carries forward, so that non-decrease of G_s across consecutive stages is equivalent to non-regression of clinical ROI-grounding capability. This assumption is load-bearing: if G_s is not a sufficient statistic (e.g., because the probe set under-represents a modality introduced after stage 0), then the gate can admit a stage that improves probe-covered regions while degrading probe-uncovered regions. We therefore (i) define G_s against gold labels rather than against a frozen stage-0 head, and (ii) stratify the probe set by modality and report per-modality G_s so the sufficiency claim is auditable rather than assumed.

**How it works.**

```pseudocode
# Setup (once, before pretraining)
#   P = held-out ROI probe set: {(image_i, ROI box_i, gold_desc_i, modality_i)}_i=1..N
#       N = 4096, disjoint from all training records (Sage-Med-style 322.7M pool)
#       Stratified by modality: N_m >= 128 per modality m present at stage 0;
#       when a new modality m' is introduced at stage s, append N_m' = 512 fresh
#       probe examples for m' drawn from the stage-s training pool but held out
#       from stage-s gradients (probe set grows; earlier strata unchanged).
#   Gold labels: box_i is an expert-annotated ROI box; gold_desc_i is an
#       expert report sentence for that ROI.
#   Description-similarity metric: entity-F1 over clinical entities extracted
#       by RadGraph (chest X-ray / CT) or CheXbert (pathology / other), not cosine.
#   tau   = 0.005  (admissible probe-agreement slack per transition)
#   lambda = 0.3  (weight of invariant loss vs. stage objective)

for stage s in [pretrain, SFT_1, ..., SFT_k, RL]:
    train model M_s on stage-s data with objective L_stage

    # --- Cross-stage perceptual audit on P (no labels leak into training) ---
    for each probe (image_i, box_i, gold_desc_i, modality_i):
        r_i^s      = ROI_localize(M_s, image_i)          # patch-grid region logits
        d_i^s      = Desc_embed(M_s, image_i, r_i^s)     # region-conditioned desc embedding
        # Label-anchored scores (no reference head):
        a_i^s      = IoU(argmax r_i^s, box_i)            # localization vs. gold box
        c_i^s      = EntityF1(d_i^s, gold_desc_i)        # description vs. gold entities

    # Per-modality and aggregate grounding scores:
    for each modality m in P:
        A_s^m = mean_{i: modality_i=m} a_i^s
        C_s^m = mean_{i: modality_i=m} c_i^s
        G_s^m = 0.5 * A_s^m + 0.5 * C_s^m
    A_s   = mean_m A_s^m ; C_s = mean_m C_s^m ; G_s = 0.5*A_s + 0.5*C_s

    # --- Stage-transition invariant loss, applied during stage s+1 ---
    # Operand is the absolute label-anchored G difference, not agreement with a head.
    L_inv = relu(G_{s-1} - G_s + tau)                    # penalize any decrease > tau
    L_total^{s+1} = L_stage^{s+1} + lambda * L_inv

    # --- Admission gate: transition only if monotone within slack ---
    # Gate uses the aggregate G_s AND every per-modality G_s^m:
    if G_s < G_{s-1} - tau  OR  any_m (G_s^m < G_{s-1}^m - tau):
        rollback to M_{s-1}; re-run stage s with L_inv active
    else:
        G_{s-1} <- G_s ; G_{s-1}^m <- G_s^m for all m ; proceed to stage s+1
```

Hyperparameters are fixed before any stage runs: probe size N=4096 (with per-modality strata N_m>=128 at stage 0 and N_m'=512 for each newly introduced modality), slack tau=0.005, invariant weight lambda=0.3. There is no frozen reference head; the anchor is the gold-label criterion (IoU against expert boxes; entity-F1 against expert descriptions via RadGraph/CheXbert).

**Why this design.** We chose a *label-anchored criterion on a held-out probe set* over a frozen stage-0 reference head because a weak, randomly initialized linear-probe head is not semantically calibrated to clinical correctness — its embeddings are dominated by embedding-space geometry rather than description accuracy — so agreement with such a head would be a circular proxy; anchoring to expert gold boxes and entity-F1 makes the operand an absolute quality measure rather than a drift-toward-reference measure. We chose a *held-out ROI probe set* over in-training batch statistics because in-training measurements are confounded by the very data the stage is fitting, accepting the cost of a fixed probe set that cannot cover new modalities introduced mid-curriculum — mitigated by appending a fresh modality stratum (N_m'=512) whenever a new modality enters. We chose a *ReLU hinge with slack tau* over an exact-equality penalty because small stage-to-stage jitter is expected and penalizing it would fight legitimate refinement, accepting the cost that a regression of size < tau is tolerated. We chose an *admission gate with rollback* over post-hoc reporting because the whole point is to correct regression at the transition boundary rather than document it afterward, accepting the cost of extra compute when a stage is re-run. A domain expert would not describe this as a variant of standard multi-task regularization or of the Sage-Med curriculum: Sage-Med applies stage objectives sequentially and reports aggregate benchmark deltas (E4, E8), whereas SM-Ground adds a constraint whose operand is the *difference between consecutive stages' label-anchored grounding scores on a fixed, modality-stratified probe set*, which no stage objective in the curriculum contains.

**Why it measures what we claim.** The probe localization agreement A_s measures *ROI localization fidelity* because the assumption that the held-out probe set is distributionally representative of the regions the model must ground makes IoU against expert gold boxes an unbiased estimate of region-grounding quality; this assumption fails when the probe set under-represents a modality introduced after stage 0, in which case A_s reflects grounding on the probe's modality mix rather than on the new modality — which is why we append a fresh modality stratum and report per-modality A_s^m. The description-alignment agreement C_s measures *region-conditioned description alignment* because the assumption that entity-F1 against expert descriptions (RadGraph/CheXbert) is monotone in clinical description correctness makes C_s a valid alignment signal; this assumption fails when the entity extractor misses clinically salient findings, in which case C_s reflects entity-extraction coverage rather than full clinical accuracy. The stage grounding score G_s and its consecutive difference G_{s-1} - G_s measure *cross-stage perceptual consistency* because the assumption that probe-set grounding is a sufficient statistic for the perceptual capability a stage carries forward makes non-decrease of G equivalent to non-regression of perception; this assumption fails when a stage improves perception on probe-covered regions while degrading it on probe-uncovered regions, in which case G reflects probe-covered consistency only — which is why the gate checks every per-modality G_s^m, not just the aggregate. The gate threshold tau measures *admissible perceptual drift* because the assumption that stage-to-stage jitter below tau is noise rather than regression makes the hinge a faithful regression detector; this assumption fails when true regression is smaller than tau, in which case the gate admits a genuine but sub-threshold decline.

## Contribution

(1) SM-Ground, a cross-stage perceptual consistency regularizer and admission gate that couples consecutive curriculum stages through a frozen reference grounding head and a held-out ROI probe set, turning perceptual regression into an observable and correctable quantity at each transition boundary. (2) A design principle for progressive medical MLLM curricula: stage transitions should be admitted on monotone ROI-grounding agreement rather than on aggregate language-heavy benchmark deltas, because the two can move in opposite directions. (3) A reusable ROI probe protocol and reference-head calibration procedure that can be layered onto existing multigranular medical datasets without modifying their annotation pipelines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Reproducible 3-stage open curriculum: LLaVA-style pretrain on MedTrinity-25M (subset), SFT on a public VQA set (VQA-RAD + SLAKE), RL on PathVQA; plus Sage-Med multi-center suite for downstream | Small curriculum makes stage-wise G_s measurable; suite gives downstream coverage |
| Backbone | 7B LLaVA-style MLLM (fixed) | Feasible on open compute; matches curriculum scale |
| Primary metric | Stage-wise, modality-stratified ROI grounding score G_s^m trajectory on P | Directly measures the quantity the gate protects, per modality |
| Secondary metric | Downstream region-centric vs. text-heavy task subsets (VQA, grounding, segmentation vs. report fluency) | Tests predicted concentration of gain |
| Baseline | Sage-Med-style progressive curriculum (no gate) | Tests whether gating itself causes the gain |
| Baseline | Standard L2 / EWC multi-task regularizer | Tests whether generic regularization suffices |
| Baseline | Functional-space EWC: KL divergence between stage s and s-1 output distributions on P | Tests whether ROI-specific hinge reduces to generic functional regularization |
| Baseline | Post-hoc grounding audit only (no rollback) | Tests whether correction-at-transition beats documentation |
| Ablation-of-ours | SM-Ground with frozen random-init linear-probe reference head (the original H_ref) | Tests whether label-anchored criterion is load-bearing vs. weak reference |
| Ablation-of-ours | SM-Ground with aggregate-only gate (no per-modality check) | Tests whether modality stratification is load-bearing |
| Diagnostic | Pre-gate G_s trajectory on the 3-stage curriculum (no gate) | Establishes regression phenomenon and magnitude standalone |

### Why this setup validates the claim

The central claim is that perceptual regression across curriculum stages is real, detectable on a fixed label-anchored probe set, and correctable at the transition boundary. The reproducible 3-stage curriculum (pretrain on MedTrinity-25M subset, SFT on VQA-RAD+SLAKE, RL on PathVQA) is the right substrate because it is small enough to run multiple seeds and large enough to exhibit stage-wise drift: a curriculum that adds tasks stage by stage will, if the claim is true, show G_s^m dips when a new stage's objective dominates. The pre-gate diagnostic establishes the existence sub-claim directly — if regression is not real, the gate has nothing to correct and our method should be at parity. The no-gate Sage-Med baseline tests the same on the larger suite. The L2/EWC baseline tests the specificity sub-claim — if weight-space regularization already prevents perceptual drift, the probe-set hinge is redundant. The functional-space EWC baseline (KL on P) tests whether the ROI-specific hinge (IoU + entity-F1) is reducible to generic functional regularization. The post-hoc-audit baseline tests the correction sub-claim — if documenting regression after the fact recovers the same downstream performance as rollback, the gate's admission mechanism is unnecessary. The frozen-random-init-reference ablation tests the design sub-claim that a label-anchored criterion is what makes monotonicity non-vacuous. The aggregate-only-gate ablation tests whether per-modality stratification is load-bearing. G_s^m is the right primary metric because it is the exact operand of the invariant loss and gate; downstream benchmark deltas are the secondary consequence, and a gain there without a G_s trajectory difference would be unexplained by our mechanism.

### Expected outcome and causal chain

**vs. Sage-Med-style progressive curriculum (no gate)** — On a case where the SFT stage shifts the model toward report-generation fluency, the baseline produces a model whose region localization on probe-covered anatomy has silently decayed because the stage objective never references grounding and nothing constrains it. Our method instead detects the drop in A_s^m on the fixed probe set and applies L_inv during the next stage, because the hinge's operand is the consecutive-stage G difference. We expect the no-gate baseline to show a visible G_s^m dip at the stage boundary and a corresponding deficit concentrated on region-centric tasks (VQA, grounding, segmentation), while our method holds G_s^m flat and matches or exceeds on those subsets with parity on text-heavy subsets.

**vs. Standard L2 / EWC multi-task regularizer** — On a case where a new modality is introduced and the regularizer pulls weights toward the previous stage's parameters, the baseline produces a model that preserves parameter proximity but still loses region-specific grounding, because weight-space penalties do not reference the ROI output distribution and can be satisfied while region logits drift. Our method instead constrains the function-space quantity that matters — probe-set localization and description alignment — because the hinge is computed on the model's actual grounding outputs. We expect the regularizer to show partial G_s retention with a residual dip on the newly added modality's probe subset, while our method shows a tighter G_s trajectory and a larger downstream gap specifically on region-centric benchmarks.

**vs. Functional-space EWC (KL divergence on P)** — On a case where a stage changes the output distribution on P, the functional EWC baseline penalizes KL divergence between stage s and s-1 outputs and therefore preserves the *entire* output distribution, including text-side fluency, rather than the ROI-grounding component. Our method instead penalizes only the label-anchored grounding decrease (IoU + entity-F1), because the hinge operand is G_s, not the full distribution. We expect functional EWC to show G_s retention comparable to ours but a larger text-side penalty (over-regularization) and a smaller region-centric downstream gain, while our method shows a better region-centric/text-heavy trade-off.

**vs. Post-hoc grounding audit only (no rollback)** — On a case where a mid-curriculum stage causes a genuine perceptual regression, the baseline produces a documented report of the drop but proceeds to the next stage anyway, because audit-without-gate cannot alter the transition. Our method instead blocks the transition, re-runs the stage with L_inv active, and resumes from a non-regressed checkpoint, because the admission decision is coupled to the loss. We expect the audit-only baseline to show the same G_s dip as no-gate plus a recovery only at later stages, while our method shows the dip truncated at the boundary and a downstream advantage that is largest on the stages immediately following an admitted regression.

**vs. SM-Ground with frozen random-init linear-probe reference head (original H_ref)** — On a case where the reference head's embeddings collapse distinct descriptions to nearby vectors, the frozen-reference variant produces a G_s that is dominated by embedding-space geometry rather than clinical description accuracy, because cosine similarity to a weak anchor is not monotone in correctness. Our method instead uses entity-F1 against expert descriptions, because the anchor is the gold label. We expect the frozen-reference variant to show near-chance A_s (as a control) and a G_s trajectory that is flat or misleading, while our label-anchored method shows a G_s trajectory that tracks downstream region-centric performance.

**vs. SM-Ground with aggregate-only gate (no per-modality check)** — On a case where a new modality is introduced and the aggregate G_s is dominated by the large stage-0 strata, the aggregate-only gate produces a model that admits a stage despite a per-modality regression on the new modality, because the aggregate masks the dip. Our method instead checks every per-modality G_s^m and blocks the transition, because the gate is stratified. We expect the aggregate-only variant to show a flat aggregate G_s but a visible dip on the new modality's G_s^m and a downstream deficit on that modality's region-centric tasks, while our method shows a flat per-modality trajectory and a downstream gain on the new modality.

### What would falsify this idea

If the pre-gate diagnostic shows a flat G_s^m trajectory across stages for the no-gate baseline (no perceptual regression to correct), or if our downstream gain is uniform across region-centric and text-heavy task subsets rather than concentrated on the region-centric subsets predicted by the mechanism, then the central claim that stage-wise perceptual regression is a real, detectable, and correctable phenomenon is wrong. Likewise, if the frozen-random-init-reference ablation matches the label-anchored version, the label-anchored construction is not load-bearing and the design rationale collapses.

## References

1. Sage-Med: Scaling Medical Multimodal Intelligence

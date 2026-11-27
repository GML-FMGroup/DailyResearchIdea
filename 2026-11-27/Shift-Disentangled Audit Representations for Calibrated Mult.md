# Shift-Disentangled Audit Representations for Calibrated Multimodal Safety Intervention

## Motivation

Multimodal safety guards such as JudiAct predict diagnostic labels and handling actions from review-item representations that are assumed to be modality-stable, yet the ShiftDC analysis shows that adding the vision modality introduces a modality-induced activation shift toward a 'safer' direction, systematically distorting safety perception. As a result, the intervention threshold JudiAct learns is calibrated against a representation that moves whenever the image modality is present, so the audit's decision boundary is not invariant to the very perturbation that drives jailbreak success (E1, E8). This calibration-invariance gap recurs across guard-style and preference-tuning approaches: they inherit a text-only judgment surface without disentangling the modality-induced component.

## Key Insight

The modality-induced activation shift is a low-dimensional, input-conditioned displacement that can be estimated training-free from paired text-only and multimodal activations, and removing it before threshold calibration makes the intervention decision invariant to the vision-modality perturbation that distorts safety perception.

## Method

**What it is.** SDA-JudiAct (Shift-Disentangled Audit JudiAct) takes JudiAct's review-item hidden states together with their text-only counterparts, decomposes each multimodal review representation into a modality-stable component and a modality-induced shift, removes the shift component, and recalibrates the four-action intervention threshold on the shift-disentangled representation. Inputs: a multimodal query-image pair, its text-only twin (image replaced by a null/neutral placeholder), and the frozen JudiAct 4B backbone. Outputs: diagnostic labels, a handling action, and a calibrated intervention score.

**Load-bearing assumption (stated explicitly).** The method assumes that the modality-induced activation shift delta = h_mm - h_txt is a low-dimensional, input-conditioned displacement whose safety-relevant component lies along a single PCA direction D_safe estimated from paired text/multimodal activations, so that projecting out (delta·D_safe)D_safe yields a representation h_stable that is invariant to the vision modality for threshold calibration. This assumption is load-bearing: if the shift is not low-dimensional or not collinear with the safety-label gradient (e.g., nonlinear or multi-dimensional shift as in JOOD mixup), the projection removes only part of the shift and the residual orthogonal component is absorbed by tau_action, reproducing the calibration-invariance gap the method claims to close. The method therefore includes a measured invariance constraint (Phase 3) and a diagnostic verification (Phase 5) that directly test this assumption rather than asserting it.

**How it works.**

```pseudocode
# Phase 1: paired activation extraction (frozen backbone)
for each audit instance (q, v):
    h_mm = JudiAct.review_hidden(q, v)          # multimodal review-item state
    h_txt = JudiAct.review_hidden(q, v_null)    # text-only twin, v_null = blank image
    delta = h_mm - h_txt                         # raw modality-induced displacement

# Phase 2: training-free shift disentanglement (ShiftDC-style)
# Estimate the safety-relevant direction from a held-out calibration set of N=512 audits
D_safe = top-k PCA direction of {delta_i} aligned with safety-label gradient   # k=1
# Project out the safety-relevant component of delta
shift_safe = (delta . D_safe) * D_safe
h_stable = h_mm - shift_safe                     # disentangled review representation

# Phase 3: threshold recalibration on h_stable with measured invariance constraint
# Fit a scalar intervention score s = w . h_stable (logistic regression, L2=1e-3)
# Threshold tau chosen on held-out calibration split to maximize balanced accuracy
# over {allow, flag, refuse, escalate}, subject to a paired-twin agreement penalty:
#   L = -BalancedAcc(tau) + lambda * mean_i || h_stable(q_i, v_i) - h_stable(q_i, v_null) ||_2
# with lambda = 0.1, optimized by grid search over tau (4 per-action thresholds, monotone-ordered)
s = w . h_stable
action = argmax_action 1[s >= tau_action]        # per-action thresholds, monotone-ordered

# Phase 4: audit (unchanged JudiAct review-item verification)
labels = JudiAct.verify_review_items(q, v, h_stable)
return labels, action, s

# Phase 5: diagnostic verification of the rank-1 assumption (reported, not used for training)
# - variance captured by top-k PCA directions of {delta_i}, k=1..10
# - cosine alignment between D_safe and the safety-label gradient
# - residual modality sensitivity: mean_i || h_stable(q_i, v_i) - h_stable(q_i, v_null) ||_2
# - paired-twin agreement rate: fraction of calibration instances where argmax action on (q,v) equals that on (q,v_null)
```

**Why this design.** We chose a training-free PCA-plus-projection disentanglement over a learned independence head because the modality shift is estimated from paired text/multimodal activations of the *same* frozen backbone, so the displacement is a measured quantity rather than a statistical independence assertion the data cannot support — accepting the cost that the projection direction must be re-estimated per backbone and per calibration set. We chose to recalibrate only the scalar threshold on the disentangled representation rather than retraining JudiAct's review generator, because retraining would entangle the audit mechanism with the shift correction and destroy the diagnostic-label supervision JudiAct already provides — accepting the cost that residual shift components orthogonal to D_safe remain uncorrected. We chose per-action monotone-ordered thresholds over a single global threshold because the four handling actions have asymmetric error costs, and enforcing monotonicity prevents the calibration from inverting action severity — accepting the cost that the calibration split must be large enough to estimate four thresholds. We added a paired-twin agreement penalty to the threshold objective (lambda=0.1) so that invariance is a measured constraint on the calibration split rather than an assumed property of the projection; this penalty does not change the training procedure (still grid search over tau) but makes the residual modality sensitivity visible in the objective. A domain expert would not describe this as 'ShiftDC + JudiAct' because ShiftDC operates on the final output logits of a VLM, whereas SDA-JudiAct operates on intermediate review-item representations and recalibrates an intervention *decision rule*, not a generation distribution; the composition produces a threshold-invariance property that neither component alone establishes.

**Why it measures what we claim.** The projection residual h_stable measures modality-stable safety representation because we assume the safety-relevant direction D_safe estimated from paired text/multimodal deltas is the unique direction along which the vision modality systematically displaces safety perception; this assumption fails when the shift has a component orthogonal to the top PCA direction (e.g., shift varies nonlinearly with image content), in which case h_stable retains residual modality sensitivity that the threshold absorbs as noise. The per-action threshold tau_action measures calibrated intervention invariance because we assume the held-out calibration split is exchangeable with test instances in the shift-disentangled space; this assumption fails under distribution shift in image type (e.g., adversarial mixup inputs as in JOOD), in which case tau_action reflects the calibration split's shift distribution rather than a modality-invariant decision boundary. The balanced-accuracy objective measures decision calibration rather than raw accuracy because we assume the four actions have comparable per-class importance in the audit setting; this assumption fails when one action dominates the test distribution, in which case the threshold overfits the majority action. The paired-twin agreement penalty measures the residual modality sensitivity of h_stable directly as the L2 distance between h_stable(q,v) and h_stable(q,v_null), because we assume that a truly modality-invariant representation should yield identical decisions on the paired twins; this assumption fails when the text-only twin construction (blank image) itself introduces a distribution shift, in which case the penalty reflects twin-construction artifacts rather than modality invariance.

## Contribution

['A training-free shift-disentangled audit representation (SDA-JudiAct) that decomposes JudiAct review-item hidden states into modality-stable and modality-induced components using paired text/multimodal activations of a frozen backbone, and recalibrates per-action intervention thresholds on the disentangled representation.', 'A design principle: intervention-threshold calibration for multimodal safety guards should be performed on shift-disentangled representations rather than raw multimodal representations, because the modality-induced activation shift otherwise moves the decision boundary in a way that is invisible to label-level supervision.', 'An analysis of when the shift-disentanglement assumption holds (low-dimensional, safety-aligned shift direction) and when it breaks (orthogonal shift components, adversarial image perturbations), with a diagnostic for detecting the break.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | JudiBench (paired image/text audits) | Only benchmark with JudiAct-native review-item supervision |
| Primary metric | Request & Answer Macro-F1 | Balanced across four actions; insensitive to action skew |
| Secondary metric | Paired-twin agreement rate (fraction of instances where argmax action on (q,v) equals that on (q,v_null)) | Directly measures modality invariance of the decision rule |
| Secondary metric | Residual modality sensitivity: mean L2 distance between h_stable(q,v) and h_stable(q,v_null) | Quantifies uncorrected shift component |
| Baseline | Frozen JudiAct 4B backbone | Tests whether shift correction adds value over unmodified audit |
| Baseline | JudiAct + JOOD-style OOD mixup inputs | Tests robustness under adversarial modality shift |
| Baseline | JudiAct + naive global threshold recalibration | Tests whether shift disentanglement, not just recalibration, matters |
| Ablation-of-ours | SDA-JudiAct without Phase-2 projection (recalibration only) | Isolates contribution of shift removal from threshold fitting |
| Ablation-of-ours | SDA-JudiAct with k=1 vs. k=3 vs. k=5 PCA directions | Tests low-dimensionality assumption; if k=1 captures <50% variance, premise falsified |
| Ablation-of-ours | Text-only twin construction: blank image vs. neutral placeholder vs. caption-only | Sensitivity to twin design, which determines delta |

### Why this setup validates the claim

The central claim is that modality-induced activation shift, not threshold placement alone, degrades JudiAct's intervention decisions, and that removing the shift component before recalibrating yields a decision rule that is invariant to the vision modality. The frozen-backbone baseline tests the first sub-claim: any gain over it must come from the disentanglement-plus-recalibration pipeline, since the audit mechanism is identical. The naive-recalibration baseline is the critical discriminator: it shares our Phase-3 threshold fitting but skips Phase-2 projection, so if our gain comes from disentanglement rather than from refitting thresholds, our method must beat it specifically on instances where the modality shift is largest. The JOOD-style OOD baseline tests whether the correction generalizes to distribution-shifted image content, which is exactly where the stated assumption (shift lies along D_safe) is most at risk. The ablation (recalibration without projection) makes the causal attribution symmetric: any gap between it and the full method isolates the projection step. The k-sweep ablation directly tests the load-bearing rank-1 assumption: if k=1 captures <50% of delta variance, the low-dimensionality premise is falsified and the method must be reframed. The twin-construction ablation tests the sensitivity of delta to the text-only twin design, which is currently underspecified. The paired-twin agreement rate and residual modality sensitivity are reported as measured quantities, not assumed, so the invariance claim is empirically grounded. Macro-F1 is the right primary metric because it weights the four asymmetric actions equally, so a method that inflates the dominant action cannot masquerade as calibrated, and because JudiBench's native request/answer supervision makes the labels trustworthy rather than proxy.

### Expected outcome and causal chain

**vs. frozen JudiAct 4B backbone** — On a case where a benign image accompanies a mildly sensitive request, the baseline sees a modality-induced shift pushing its review state toward the "safer" direction, so it labels the request as safe and issues an `allow` where a `flag` is warranted, because it has no mechanism to separate the shift from the underlying safety signal. Our method subtracts the projected shift component before scoring, so the review representation reflects the text-only safety judgment, and the recalibrated threshold routes the same case to `flag`. We expect a clear Macro-F1 gap concentrated on image-present instances, with near-parity on text-only twins, since the correction should be inert when no modality shift exists. We also expect our paired-twin agreement rate to be substantially higher than the baseline's, and our residual modality sensitivity to be lower, confirming that the shift removal reduces the measured invariance gap.

**vs. JudiAct + naive global threshold recalibration** — On a case where the image strongly displaces the review state, the naive baseline fits thresholds on a representation that still contains the shift, so its decision boundary absorbs the shift as noise and misclassifies borderline cases in the direction of the calibration split's image distribution, producing `allow` on genuinely unsafe requests. Our method removes the shift first, so the threshold is fit on a representation where the safety signal is not competing with modality displacement, and the same case lands on `refuse`. We expect the largest gap precisely on the high-shift subset and near-zero gap on low-shift cases, which is the signature that disentanglement, not recalibration, drives the gain. The paired-twin agreement penalty in our Phase 3 should further widen this gap by explicitly minimizing residual modality sensitivity during threshold fitting.

**vs. JudiAct + JOOD-style OOD mixup inputs** — On an adversarial mixup image designed to raise model uncertainty, the baseline's shift is large and partly orthogonal to the top PCA direction, so its review state drifts and it produces `allow` on a request it should escalate, because the shift it cannot project out overwhelms the safety signal. Our method still removes the dominant shift component, recovering most of the text-only judgment, though residual orthogonal shift remains; we expect a substantial but not complete recovery on this subset — a smaller gap than the naive-recalibration comparison, which is consistent with our stated residual-shift limitation and would be a partial-failure signal rather than a clean win. The k-sweep ablation (k=1 vs. k=3 vs. k=5) will show whether increasing the subspace dimension recovers more of the shift on this subset; if k=1 captures <50% of delta variance, the low-dimensionality premise is falsified and the method should be reframed as 'threshold recalibration with a shift-aware regularizer' rather than 'disentanglement'.

### What would falsify this idea

If the gain over the frozen backbone is uniform across high-shift and low-shift subsets rather than concentrated where the modality shift is largest, then the disentanglement is not doing the causal work and the improvement is just threshold refitting. Equally, if the full method fails to beat the recalibration-only ablation on the high-shift subset, or if the top-1 PCA direction captures less than 50% of delta variance (falsifying the low-dimensionality premise), the Phase-2 projection contributes nothing beyond Phase-3 and the central claim is wrong.

## References

1. Guarding with Judgment: Learning to Intervene for Multimodal Safety

# Causal Admission of SAE Concepts via Held-Out Feature Ablation

## Motivation

The auditable-detector pipeline of the SAE-based detector paper admits every auto-interpreted SAE feature that passes a global fidelity threshold, which is a correlational filter: a feature can be confidently and faithfully labeled while being causally irrelevant to the detector's decisions, and the pipeline has no step that would detect this. Because the audit set is chosen before any intervention is run, the resulting 'auditable' detector certifies labels, not mechanisms, and downstream users cannot distinguish a concept the detector actually uses from a concept that merely co-occurs with its output.

## Key Insight

A labeled concept is causally necessary for a detector decision exactly when removing its activation on inputs where the concept is present, and only there, produces a larger output change than removing it on matched inputs where the concept is absent — a difference-in-differences that cancels the detector's generic sensitivity to any ablation and isolates concept-specific dependence.

## Method

**What it is.** CACHE (Causal Admission of Concepts via Held-out Excision) takes a fine-tuned detector $D$, its residual-stream SAE with auto-interpreted feature labels, and a held-out probe set $\mathcal{H}$ disjoint from SAE training and classifier training. It returns an admitted concept set $\mathcal{A}$ and, for each admitted concept, a causal-necessity score with a significance test.

**Load-bearing assumption (stated explicitly).** CACHE assumes that zeroing a single SAE feature coordinate at the decision token is a valid, concept-specific intervention: the SAE basis is locally faithful and monosemantic enough that ablating feature $k$ removes only the labeled concept $L_k$ and not a bundle of co-active, correlated concepts. Under this assumption the difference-in-differences $\tau_k$ isolates dependence on $L_k$. When the assumption fails (polysemantic or absorbing features), $\tau_k$ reflects joint dependence on whatever the feature actually encodes. CACHE therefore (i) defines $\tau_k$ as an excess over a matched placebo ablation rather than over a generic off-distribution shift, and (ii) reports $\tau_k$ as an upper bound on concept-specific necessity, with a co-activation partialling-out step required for admission.

**How it works.**

```pseudocode
Input: detector D, SAE {f_k}, labels L, held-out set H with detector labels y
Output: admitted set A, per-concept necessity scores tau_k, placebo-corrected

1. For each concept k with label L_k:
     H_k^+ = {x in H : L_k present in x}          # concept-positive slice
     H_k^- = match(H_k^+, on length, domain, y)    # concept-absent, detector-label-matched
     # exact matching: length bins of 16 tokens, domain label, detector label y
     # report covariate balance of H_k^+ vs H_k^- before step 4
2. For each x in H_k^+ ∪ H_k^-:
     a_k(x) = SAE activation of feature k at the detector's decision token
     x_abl  = x with a_k(x) set to 0 (feature ablation, no retraining)
     Delta_k(x) = |D(x) - D(x_abl)|               # detector output change
     # placebo: matched latent j with similar activation statistics but label-uncorrelated
     x_plc  = x with a_j(x) set to 0
     Delta_j(x) = |D(x) - D(x_plc)|
3. Necessity score (placebo-corrected DiD):
     tau_k = [mean_{x in H_k^+} Delta_k(x) - mean_{x in H_k^-} Delta_k(x)]
             - [mean_{x in H_k^+} Delta_j(x) - mean_{x in H_k^-} Delta_j(x)]
4. Significance: permutation test on tau_k with B=2000 label swaps within matched pairs;
     admit k to A iff tau_k > 0 and p_k < 0.01 after Benjamini-Hochberg correction over all k.
5. Co-activation partialling-out: regress Delta_k(x) on the activations of all other SAE latents
     co-active with k (|corr| > 0.3 on H_k^+ ∪ H_k^-); retain k only if the partial coefficient
     on a_k(x) remains positive and p < 0.01.
6. Report A and {tau_k, p_k}; concepts failing step 4 or step 5 are excluded from the audit.
```

**Why this design.** We chose feature ablation over activation patching from a donor input because ablation gives a well-defined null (zero the feature) that does not require constructing a semantically matched donor, accepting the cost that ablation can push the detector off-distribution; the matched concept-absent slice in step 1 absorbs that cost by measuring the same off-distribution shift under the null. We chose a difference-in-differences score $\tau_k$ over the raw mean $\Delta_k$ because raw ablation magnitude conflates concept-specific dependence with the detector's generic sensitivity to any residual perturbation, accepting the cost of needing matched negative examples; the matching in step 1 is what makes the subtraction valid. We chose a nonparametric permutation test over a parametric t-test because $\Delta_k$ is bounded and heavy-tailed near saturation, accepting the cost of $B=2000$ resamples per concept. We chose a held-out set disjoint from SAE and classifier training over reusing the audit set because in-sample ablation would reward features the classifier memorized, accepting the cost of a smaller probe set. The placebo subtraction in step 3 and the co-activation partialling-out in step 5 are added because the load-bearing assumption (monosemanticity) is empirically false for some features; they do not change the core DiD design but make $\tau_k$ an excess over a matched, label-uncorrelated latent and require admitted concepts to survive a co-activation control. A domain expert would not describe CACHE as a variant of the SAE-detector pipeline: that pipeline selects features by label fidelity, whereas CACHE selects by an interventional contrast that fidelity cannot produce.

**Why it measures what we claim.** $\Delta_k(x)$ measures the detector's dependence on feature $k$ at input $x$ because zeroing $a_k$ is a valid intervention on the SAE coordinate under the assumption that the SAE basis is locally faithful at the decision token; this assumption fails when a feature is polysemantic, in which case $\Delta_k$ reflects the detector's dependence on the dominant sub-concept rather than the labeled one. The matched-pair subtraction $\tau_k$ measures concept-specific causal necessity because matching on detector label and length makes $H_k^-$ a valid counterfactual for 'same decision, concept absent' under the assumption that no other concept correlated with $L_k$ differs systematically across the matched slices; this assumption fails when the auto-interpretation label $L_k$ is itself correlated with a second feature, in which case $\tau_k$ reflects the joint dependence and over-attributes necessity to $k$. The placebo correction in step 3 measures the excess of $\Delta_k$ over a matched, label-uncorrelated latent's ablation because the placebo shares activation statistics but not the label, under the assumption that the placebo is not co-active with $L_k$; this assumption fails when the placebo latent is itself correlated with $L_k$, in which case the correction removes part of the concept-specific effect. The co-activation partialling-out in step 5 measures whether $\tau_k$ survives controlling for co-active latents because it regresses $\Delta_k$ on the other co-active features, under the assumption that the co-active set is measured without error; this assumption fails when co-active latents are themselves polysemantic, in which case the partial coefficient reflects residual joint dependence. The permutation $p_k$ measures whether $\tau_k$ exceeds chance because label swaps within matched pairs preserve the marginal distributions of $\Delta_k$ under the null of no concept-specific effect; this assumption fails when matched pairs are not exchangeable (e.g., domain imbalance), in which case $p_k$ reflects the matching quality rather than the causal effect.

## Contribution

['A held-out interventional admission protocol (CACHE) that converts correlational SAE feature labels into causally certified concepts via matched-pair feature ablation and a difference-in-differences necessity score with permutation-based significance control.', 'The design principle that causal admission must be measured as a contrast between concept-present and detector-label-matched concept-absent slices, because raw ablation magnitude is dominated by generic detector sensitivity and would admit nearly every feature.', 'An audit artifact: for each admitted concept, a necessity score and corrected p-value that let users distinguish detector-used concepts from co-occurring labels.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AI-text detection with held-out domain shift | Tests OOD where memorization cannot mask causality |
| Primary metric | Necessity-score gap: admitted vs. rejected concepts | Directly reads out concept-specific causal dependence |
| Baseline | Fidelity-filtered SAE classifier | Tests whether label fidelity alone suffices |
| Baseline | Opaque fine-tuned detector | Tests whether audit adds signal beyond raw accuracy |
| Baseline | Stylometric classifier | Tests whether surface features explain admitted concepts |
| Baseline | Activation patching with semantically matched donor | Tests whether DiD yields distinct admitted set from causal mediation |
| Ablation-of-ours | CACHE with raw Δ_k (no matched subtraction) | Isolates the difference-in-differences contribution |
| Ablation-of-ours | CACHE without placebo correction (step 3) | Isolates the placebo-subtraction contribution |
| Ablation-of-ours | CACHE without co-activation partialling-out (step 5) | Isolates the co-activation control contribution |

### Why this setup validates the claim

The central claim is that concept-specific causal necessity, not label fidelity, identifies which SAE features the detector actually relies on. The fidelity-filtered SAE classifier tests the sub-claim that label fidelity is insufficient: if fidelity-selected features and CACHE-admitted features diverge, and only the latter track detector behavior under intervention, fidelity is not a proxy for necessity. The opaque detector tests whether the audit adds information beyond accuracy: CACHE must produce admitted concepts whose ablation degrades the detector on the same inputs where the opaque model is confidently wrong, otherwise the audit is decorative. The stylometric baseline tests whether admitted concepts reduce to surface linguistic cues: if CACHE's admitted set does not overlap stylometric features yet still predicts detector failures, the causal story survives. The activation-patching baseline with a semantically matched donor tests whether the DiD contrast yields a distinct admitted set rather than a cleaner version of the same ranking; if the two admitted sets are nearly identical, the DiD mechanism is not doing distinct work. The raw-Δ_k ablation is the load-bearing control: it removes the matched-pair subtraction, so if CACHE's advantage disappears without matching, the gain is attributable to the difference-in-differences design rather than to ablation magnitude. The placebo and co-activation ablations test the load-bearing assumption directly: if removing either step does not degrade precision, the polysemanticity correction is not doing the work we attribute to it. The necessity-score gap metric is chosen because it is the quantity the method claims to estimate; accuracy alone cannot distinguish admitted from rejected concepts, whereas the gap is exactly the predicted effect and is falsifiable if it collapses under the ablation.

### Expected outcome and causal chain

**vs. Fidelity-filtered SAE classifier** — On an OOD instance where a high-fidelity feature fires on a spurious correlation (e.g., a stylistic artifact present in training AI text but absent in held-out human text), the baseline admits that feature because its auto-interpretation label matches the concept, and the classifier trusts it. Our method instead ablates the feature on the matched concept-absent slice and finds near-zero Δ_k there, so τ_k stays small and the feature is rejected. We expect a visible gap in admitted-set composition concentrated on OOD-shifted concepts, with CACHE's admitted set showing lower detector-degradation under ablation on those same inputs.

**vs. Opaque fine-tuned detector** — On an instance where the detector is confidently wrong (high logit on a misclassified held-out example), the baseline offers no diagnostic because its residual stream is uninterpretable at the decision token. Our method intervenes on admitted features at that token and shows which concept's ablation flips the decision, localizing the error. We expect CACHE to identify at least one admitted concept whose ablation changes the detector output on the majority of confident-error cases, whereas the opaque baseline provides no such handle — the observable signal is a concentration of high-τ_k features on the error subset relative to the correct subset.

**vs. Stylometric classifier** — On an instance where the detector's decision is driven by a semantic concept (e.g., a topical or syntactic pattern) rather than a handcrafted surface feature, the stylometric baseline cannot represent that concept and misattributes the decision to a correlated surface cue. Our method instead admits the SAE feature whose ablation moves the detector and whose matched-pair contrast isolates it from generic perturbation sensitivity. We expect low overlap between CACHE's admitted set and stylometric feature importance, yet CACHE's admitted concepts to predict detector failures better than stylometric importance does — a dissociation pattern, not uniform improvement.

**vs. Activation patching with semantically matched donor** — On an instance where the donor input is not perfectly matched on co-active concepts, activation patching can attribute the decision to a concept that is merely correlated with the donor's residual stream. Our method's matched-pair subtraction and placebo correction isolate the concept-specific excess. We expect the two admitted sets to overlap substantially on high-τ_k features but to diverge on features with high polysemanticity (top-activating token diversity above the median), where CACHE rejects features that patching admits — a divergence concentrated on the polysemantic stratum.

**vs. CACHE with raw Δ_k (no matched subtraction)** — On an instance where the detector is generically sensitive to any residual perturbation, raw Δ_k is large for many features regardless of concept, so the ablation admits features indiscriminately. Our method's matched subtraction cancels that generic sensitivity, so τ_k isolates concept-specific dependence. We expect the ablation to admit a much larger, noisier set with poor precision on the OOD subset, while CACHE's admitted set stays small and its necessity-score gap remains large — the gain should be concentrated exactly where generic perturbation sensitivity is high.

**vs. CACHE without placebo correction (step 3)** — On an instance where a polysemantic feature absorbs a correlated concept, the un-corrected τ_k over-attributes necessity to the labeled concept. The placebo subtraction removes the shared off-distribution shift from a matched, label-uncorrelated latent. We expect the no-placebo ablation to admit a larger set with lower precision on the polysemantic stratum (top-activating token diversity above the median), while CACHE's admitted set remains stable across strata.

**vs. CACHE without co-activation partialling-out (step 5)** — On an instance where a second feature co-varies with L_k across the matched slices, the un-partialled τ_k reflects joint dependence. The partialling-out step retains only features whose partial coefficient on a_k(x) remains positive. We expect the no-partialling ablation to admit features that are co-active with the true concept but not causally necessary, with the precision gap concentrated on the high-co-activation stratum (|corr| > 0.3 with another latent).

### What would falsify this idea

If CACHE's admitted set and necessity-score gap are statistically indistinguishable from the fidelity-filtered baseline across all subsets, or if the gain is uniform rather than concentrated on the OOD-shifted and confident-error subsets where the predicted failure modes occur, the central claim that causal necessity outperforms fidelity is wrong. A second falsifier: if removing the matched-pair subtraction (raw Δ_k), the placebo correction, or the co-activation partialling-out does not degrade precision on the polysemantic or high-co-activation strata, the corresponding mechanism is not doing the work we attribute to it.

## References

1. Building Auditable AI Text Detectors with Sparse Autoencoders

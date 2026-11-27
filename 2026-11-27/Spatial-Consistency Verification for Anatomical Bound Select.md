# Spatial-Consistency Verification for Anatomical Bound Selection in 3D Medical Vision-Language Models

## Motivation

Medical VLMs that rely on anatomical bound selection, such as Radev, assume the frozen 3D encoder's organ features are spatially faithful without any verification loop; consequently, a miscalibrated bounds predictor can cause the query to attend to 2D slice features outside the true anatomy, and the language backbone consumes these tokens as if they were organ evidence. This structural assumption — inherited from the VILA-M3 ancestor assumption that expert-model output is a fixed oracle never re-checked against the VLM's own spatial grounding — means that spatial faithfulness is never tested at inference, so errors in bound selection propagate silently into clinical decisions.

## Key Insight

The frozen encoder's organ-level feature manifold provides a reference distribution against which any selected token set can be tested for spatial consistency; if selected 2D slice tokens, re-projected into that manifold, deviate systematically from the encoder's own organ features, the bound selection is provably inconsistent with the anatomy the encoder represents.

## Method

**What it is** — SC-Verify (Spatial-Consistency Verifier) is a closed-loop verification module attached to Radev's anatomical bound selector. Its inputs are (i) the candidate 2D slice tokens selected by the query within predicted anatomical bounds and (ii) the frozen 3D encoder's organ-level feature map; its output is a per-token consistency score that re-weights the selected token set before it reaches the language backbone.

**How it works** — The verifier operates in three phases inside the Radev forward pass:

```pseudocode
# Phase 1: Re-projection
for each selected 2D slice token t_i at spatial location (x_i, y_i, z_i):
    p_i = Proj(t_i)                      # linear projection to encoder feature dim (Proj: 2-layer MLP, hidden=256, GeLU)
    e_i = Interp3D(F_organ, x_i, y_i, z_i)  # trilinear interpolation of frozen 3D organ features
    d_i = 1 - cosine_similarity(p_i, e_i)   # per-token deviation

# Phase 2: Consistency scoring
s_i = exp(-d_i / tau)                    # tau = 0.1, temperature for sharpness
w_i = s_i / (1 + s_i)                    # bounded re-weight in [0, 0.5]
tilde_t_i = w_i * t_i + (1 - w_i) * StopGrad(Proj^{-1}(e_i))  # pull toward manifold

# Phase 3: Counterfactual bound perturbation training
for each training sample with ground-truth organ mask M:
    bounds_clean = predicted bounds
    bounds_shifted = bounds_clean shifted by delta ~ U(5, 15) voxels outside M
    tokens_clean = Select(bounds_clean); tokens_shifted = Select(bounds_shifted)
    L_inv = || mean_i d_i(tokens_clean) - mean_i d_i(tokens_shifted) ||_1  # invariance: clean should have low deviation
    L_flag = BCE(flag(tokens_shifted), 1) + BCE(flag(tokens_clean), 0)     # flag = 1 if mean d_i > theta (theta = 0.3)
    L_total = L_task + lambda1 * L_inv + lambda2 * L_flag   # lambda1 = 0.5, lambda2 = 0.2
```

**Why this design** — We chose a closed-loop re-projection verifier over a learned independence head (which would match Anti-Pattern 1) because the frozen encoder's organ features provide a structural reference that does not require disentangling coupled variables: the encoder's spatial manifold is fixed and known, so deviation from it is a well-defined quantity rather than a statistical independence claim. We chose counterfactual bound perturbation over simply training on clean bounds because the verifier must learn to flag out-of-anatomy selections that never appear in clean training data, accepting the cost that perturbation magnitude (5–15 voxels) must be tuned to the organ scale. We chose a bounded re-weighting scheme (w_i in [0, 0.5]) over hard rejection because hard rejection would create non-differentiable gates and discard partially correct tokens, accepting the cost that severely misselected tokens are down-weighted rather than eliminated; the language backbone's attention can still recover if enough consistent tokens remain. We chose cosine deviation in the encoder feature space over raw pixel-space deviation because the frozen encoder's organ features are the actual representation the language backbone was trained to consume, accepting the cost that the projection matrix Proj must be learned jointly with the verifier. This design is not a variant of retrieval-based verification (Anti-Pattern 2) because the verifier does not retrieve a nearest neighbor from a library; it computes a counterfactual deviation under bound perturbation, which requires intervening on the selection mechanism itself.

**Why it measures what we claim** — The per-token deviation d_i measures spatial inconsistency between selected tokens and the encoder's organ manifold because the assumption that trilinear interpolation of frozen organ features is a faithful spatial reference holds when the encoder's feature map is spatially smooth at the organ scale; this assumption fails when the encoder's features are noisy at fine scales, in which case d_i reflects encoder noise rather than bound misselection. The counterfactual invariance loss L_inv measures the verifier's sensitivity to out-of-anatomy selection because the assumption that shifting bounds outside the ground-truth mask necessarily increases deviation holds when the organ mask is accurate; this assumption fails when the mask itself is noisy, in which case L_inv penalizes correct selections near organ boundaries. The flag classifier measures the operational consistency signal because the assumption that a threshold theta=0.3 on mean deviation separates clean from shifted selections holds when the deviation distributions are separated; this assumption fails when organ features are heterogeneous across slices, in which case the flag reflects slice-level variability rather than bound error.

**(D) Linking assumption and diagnostic** — The verifier's core linking assumption A is that the frozen encoder's organ feature map F_organ is spatially smooth and organ-discriminative at the interpolation scale, so that d_i reflects bound misselection rather than encoder noise. To test A, we add a feature-smoothness diagnostic: for each organ, compute the mean absolute difference of F_organ between adjacent voxels within the organ mask (intra-organ smoothness) and across the organ boundary (inter-organ contrast). We report the fraction of voxels where intra-organ smoothness exceeds a threshold (e.g., adjacent-voxel difference < 0.1 in normalized feature space) and the separation between clean and shifted d_i distributions (e.g., AUC of d_i for classifying clean vs. shifted selections). If the smoothness fraction is below 0.8 or the d_i AUC is below 0.7, the verifier's applicability is gated: we report this as a precondition and do not claim SC-Verify works on that organ.

## Contribution

["A closed-loop spatial-consistency verifier (SC-Verify) that re-projects selected 2D slice tokens into the frozen 3D encoder's organ-feature manifold and computes per-token deviation scores, providing the first mechanism that tests the spatial faithfulness assumption embedded in Radev's frozen-encoder-plus-selector design.", 'A counterfactual bound-perturbation training objective that teaches the verifier to flag out-of-anatomy selections by comparing deviation statistics under clean versus shifted anatomical bounds, enabling the verifier to generalize to miscalibrated bounds not seen during clean training.', 'A bounded re-weighting scheme that down-weights inconsistent tokens before they reach the language backbone, preserving differentiability while reducing the influence of out-of-anatomy evidence on clinical decisions.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Radev volumetric CT benchmark with organ masks | Provides ground-truth anatomy for perturbation |
| Primary metric | Decision accuracy on out-of-bound perturbed cases | Directly tests robustness to misselection |
| Baseline | Radev (unmodified anatomical bound selector) | Tests whether verification adds anything |
| Baseline | Radev + hard-rejection gate | Tests bounded re-weighting vs elimination |
| Baseline | Radev + pixel-space deviation verifier | Tests encoder-space over pixel-space reference |
| Baseline | Radev + SC-Verify with fixed identity-aligned projection (Proj replaced by frozen encoder's own decoder/upsampling path) | Tests whether the learned projection is necessary or whether the manifold-consistency intervention alone suffices |
| Ablation-of-ours | SC-Verify without counterfactual perturbation loss | Isolates the contribution of Phase 3 |
| Diagnostic | Feature-smoothness and d_i separation analysis | Validates the linking assumption A before claiming the verifier measures spatial inconsistency |

### Why this setup validates the claim

The central claim is that a frozen-encoder re-projection verifier, trained with counterfactual bound perturbation, catches out-of-anatomy token selections that Radev's selector cannot. The unmodified Radev baseline tests the core sub-claim: if SC-Verify adds nothing over the selector alone, the verifier is inert. The hard-rejection baseline tests the bounded-re-weighting design choice — if hard rejection matches or beats bounded re-weighting, our stated rationale for accepting down-weighting over elimination is wrong. The pixel-space verifier baseline tests whether the encoder feature space is the right reference; if pixel deviation works equally well, the projection machinery is unjustified. The fixed-projection baseline tests whether the learned Proj is necessary; if it matches SC-Verify, the novelty is the manifold-consistency intervention itself, not the learned projection. The no-perturbation ablation isolates Phase 3: without counterfactual training, the verifier should never learn to flag shifted selections, so any gain from SC-Verify should collapse. The diagnostic directly tests the linking assumption A: if the encoder's feature map is not smooth at the organ scale or if d_i does not separate clean from shifted selections, the verifier's core signal is invalid. Decision accuracy on perturbed cases is the right primary metric because the predicted effect is specifically about out-of-bound selections, not clean-case performance; a metric aggregated over clean cases would dilute the signal. Together these form a falsifiable test: the gain must be concentrated on perturbed cases, must require perturbation training, must not be reproducible by simpler verifiers, and must be accompanied by a validated smoothness/separation diagnostic.

### Expected outcome and causal chain

**vs. Radev (unmodified)** — On a case where the query's attention drifts toward a neighboring organ at the predicted bound edge, the baseline selects slice tokens outside the true organ because its bound predictor has no internal consistency check, so the language backbone receives off-manifold tokens. Our method instead computes cosine deviation between projected tokens and the interpolated organ manifold, down-weighting the drifted tokens before they reach the backbone. We expect a noticeable accuracy gap on perturbed cases concentrated at organ boundaries, with near-parity on clean central-organ cases.

**vs. Radev + hard-rejection gate** — On a case where the bound is only partially shifted, so a subset of selected tokens remain anatomically valid, the hard-rejection baseline discards the entire token set because its gate is binary and non-differentiable, losing correct evidence. Our method instead re-weights per token with w_i in [0, 0.5], retaining partially correct tokens while suppressing deviant ones. We expect our method to outperform hard rejection specifically on partial-shift cases, with the gap shrinking on severe full-shift cases where rejection and down-weighting converge.

**vs. Radev + pixel-space deviation verifier** — On a case where the organ has heterogeneous intensity (e.g., contrast-enhanced versus unenhanced regions), the pixel-space baseline flags valid tokens as deviant because raw intensity deviation reflects acquisition variation rather than spatial misselection. Our method instead measures deviation in the frozen encoder's organ feature space, which abstracts over intensity and encodes structure. We expect parity on homogeneous organs but a clear advantage on heterogeneous-intensity subsets, where the pixel verifier's false-flag rate inflates.

**vs. Radev + SC-Verify with fixed identity-aligned projection** — On a case where the learned Proj adapts to the encoder's feature statistics, the fixed-projection baseline may underperform because it cannot align the 2D token space to the 3D organ manifold. However, if the fixed-projection baseline matches SC-Verify, the learned Proj is not the source of the gain, and the contribution is the counterfactual perturbation protocol and manifold-consistency intervention. We expect the learned Proj to provide a modest advantage on organs with high feature heterogeneity, but the core gain to persist even with fixed projection.

### What would falsify this idea

If SC-Verify's accuracy gain over unmodified Radev is uniform across clean and perturbed cases rather than concentrated on out-of-bound selections, the claimed mechanism is wrong. Likewise, if the no-perturbation ablation matches full SC-Verify, counterfactual bound perturbation is not doing the work we claim. Additionally, if the feature-smoothness diagnostic shows that the frozen encoder's organ feature map is not smooth at the organ scale (intra-organ adjacent-voxel difference > 0.1 for more than 20% of voxels) or if d_i does not separate clean from shifted selections (AUC < 0.7), the linking assumption A fails and the verifier's core signal is invalid.

## References

1. Decision-Native Multimodal Clinical Modeling via Anatomical Grounding and Confidence Calibration

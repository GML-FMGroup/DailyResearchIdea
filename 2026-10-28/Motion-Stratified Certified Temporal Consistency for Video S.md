# Motion-Stratified Certified Temporal Consistency for Video Style Transfer

## Motivation

Current diffusion-transformer video style transfer methods, such as VISTA, learn a single temporal operator that is only valid on the training motion distribution; under occlusion and large motion they have neither a consistency invariant nor a coverage certificate, so temporal style consistency is unguaranteed exactly where it matters. This is a structural limitation of the field's convergent action toward end-to-end learned temporal operators: the operator's validity is defined by the training distribution, not by any certified property of the motion patterns it encounters.

## Key Insight

A per-pixel motion-pattern classifier can partition the motion-pattern space into strata where a Lipschitz-style temporal consistency bound is enforced by construction, while an explicit uncertified stratum falls back to a conservative identity-preserving propagation, so that the operator's consistency is certified exactly where it is otherwise unconstrained.

## Method

# MS-CTOC: Motion-Stratified Certified Temporal Consistency Operator
# Input: noisy latent video x_t ∈ R^{T×H×W×C}, style reference s, clean video v, timestep t
# Output: denoised latent video x_{t-1}, per-pixel certificate mask m ∈ {0,1}^{T×H×W}

1. Motion-pattern classifier g_φ: 3-layer MLP (input dim = 3 motion features [optical flow magnitude, occlusion score, local variance of flow], hidden dims 128, 64, GeLU activation, output dim K=4 stratum logits) applied per pixel to features computed from v and x_t; outputs stratum logits z ∈ R^{T×H×W×K} with K=4 strata (K-1=3 certified + 1 uncertified fallback).
2. Certified strata: for each certified stratum k∈{1..K-1}, temporal attention A_k = softmax(Q K^T / √d) is masked to same-stratum pixels and normalized; gating weight w_k = softmax(z)_k.
3. Certified temporal operator: x_{t-1}^{cert} = ∑_{k=1}^{K-1} w_k · (I + α_k L_k)^{-1} x_t, where L_k is the graph Laplacian of the k-th stratum's motion-pattern graph (edges between pixels with same stratum assignment and flow-magnitude difference < 0.1) and α_k is a learnable per-stratum step size (initialized to 0.1). This is a single-step diffusion of the latent along the stratum graph, which satisfies a Lipschitz bound with constant 1/(1+α_k λ_min(L_k)).
4. Uncertified fallback: x_{t-1}^{uncert} = x_t (identity-preserving propagation).
5. Final output: x_{t-1} = m ⊙ x_{t-1}^{cert} + (1-m) ⊙ x_{t-1}^{uncert}, where m = 1[argmax_k z ∈ certified strata].
6. Training loss: L = L_diffusion + λ_robust · L_DRO, where L_DRO = max_{π ∈ Δ(K)} ∑_k π_k · E_{pixels in stratum k}[||x_{t-1}^{cert} - x_{t-1}^{target}||^2], with π reweighted by stratum difficulty (π_k ∝ 1/ (1 + exp(- (L_k - μ)/σ)) where L_k is the per-stratum loss and μ,σ are running mean and std of losses across strata); λ_robust=0.1.
7. Certificate enforcement: for each certified stratum k, add penalty L_cert = max(0, ||∇_x x_{t-1}^{cert}||_2 - 1/(1+α_k λ_min(L_k)))^2, ensuring the certified Lipschitz bound holds on the training distribution.
8. Calibrated certificate mask (explicit load-bearing assumption): The certificate is valid only if the classifier's stratum assignment is calibrated to true motion-pattern structure. We replace the hard argmax mask m with a calibrated mask m_cal produced by split-conformal calibration on a held-out motion-stratified calibration set of 512 clips (disjoint from training and test). For each pixel, we compute a nonconformity score s = 1 - max_{k∈certified} softmax(z)_k (i.e., one minus the maximum certified-stratum probability). We choose a threshold τ such that the empirical coverage of pixels with s ≤ τ on the calibration set is at least 1-δ (δ=0.05), using the standard conformal quantile: τ = the ⌈(n+1)(1-δ)⌉-th smallest s among calibration pixels that are truly in certified strata (ground-truth motion labels from optical flow and occlusion masks). The calibrated mask is m_cal = 1[s ≤ τ]. The certified operator's Lipschitz bound is then conditional on the calibrated set: for pixels with m_cal=1, the bound 1/(1+α_k λ_min(L_k)) holds with probability at least 1-δ over the draw of the calibration set. Pixels with m_cal=0 are routed to the identity fallback. This calibration step is performed once after training and before test; it does not alter the training procedure.

**Why this design** — We chose a per-pixel motion-pattern classifier over a global video-level classifier because occlusion and large motion are spatially localized; a global classifier would smear certified and uncertified regions, making the certificate vacuous. We chose a graph-diffusion operator with a closed-form Lipschitz constant over a generic temporal attention layer because the latter's Lipschitz constant is not analytically controllable without spectral normalization that would cripple its expressiveness; the graph-diffusion form gives a per-stratum bound that is both tight and trainable. We chose a distributionally robust loss over motion-pattern strata over a standard empirical risk loss because the latter would optimize average performance and ignore the worst-case stratum, which is exactly where the uncertified fallback triggers; the DRO loss forces the certified strata to cover the difficult motion patterns. We chose an identity-preserving fallback over a learned fallback because a learned fallback would itself be unconstrained and reintroduce the very problem we are solving. This design is not a variant of VISTA or FLUX.1 Kontext: those methods learn a single unconstrained temporal operator, whereas our operator's temporal consistency is a structural property of the graph-diffusion form, certified by the classifier's partition, and enforced by the DRO loss — a domain expert would not describe our method as 'VISTA plus a classifier' because the classifier's output is not a routing signal but a certificate that changes the operator's mathematical form.

**Why it measures what we claim** — The per-pixel motion-pattern classifier g_φ measures the operator's validity region because we assume the motion features (flow magnitude, occlusion score, local variance) are sufficient statistics for the operator's error; this assumption fails when the classifier misassigns pixels (e.g., due to domain shift or noisy flow), in which case g_φ reflects the training distribution's motion patterns rather than the true error. The calibrated mask m_cal measures the certificate's coverage because we assume the calibration set is representative of the test motion distribution; this assumption fails when the test distribution shifts, in which case m_cal reflects the calibration set's coverage rather than the test set's. The stratified temporal warp error measures consistency gain in certified strata because we assume the ground-truth motion labels (from optical flow and occlusion masks) are accurate; this assumption fails when ground-truth labels are noisy, in which case the metric reflects label noise rather than operator performance.

## Contribution

['A motion-pattern-adaptive temporal consistency operator (MS-CTOC) whose temporal attention is gated by a per-pixel motion-pattern classifier that outputs a certified coverage set, with an explicit uncertified fallback stratum that preserves identity.', 'A distributionally robust training loss over motion-pattern strata that enforces the certified Lipschitz bound exactly on the strata where the learned operator is otherwise unconstrained, turning the certificate from a post-hoc check into a structural property of the operator.', 'A concrete instantiation of the operator on a diffusion transformer backbone for video style transfer, with a per-pixel certificate mask that identifies where temporal style consistency is guaranteed.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Motion-stratified video style benchmark (VISTA + KontextBench-style clips) | Contains localized occlusion and large-motion regions |
| Primary metric | Temporal warp error stratified by ground-truth motion stratum (from optical flow and occlusion masks) | Detects consistency gains exactly where certified operator acts, without circularity |
| Baseline | VISTA | Single unconstrained temporal operator, no certificate |
| Baseline | FLUX.1 Kontext | Flow-matching editor, consistency degrades across turns |
| Baseline | RelationAdapter | Relation transfer without motion-stratified certification |
| Ablation-of-ours | MS-CTOC w/o DRO loss (uniform stratum weighting) | Isolates whether robustness comes from DRO |
| Ablation-of-ours | MS-CTOC w/o calibration (hard argmax mask) | Isolates whether calibration is necessary for non-vacuous certificate |

### Why this setup validates the claim

The central claim is that temporal consistency is a structural property of the motion-stratified graph-diffusion operator, not a learned regularity, and that the certificate is non-vacuous. The stratified temporal warp error is the right metric because it separates pixels by ground-truth motion stratum (derived from optical flow and occlusion masks, not from g_φ's output), so the evaluation is not circular: if our certified operator genuinely controls error only where the certificate is issued, the gain must be concentrated in certified strata and vanish in the uncertified fallback stratum. VISTA tests whether a single unconstrained temporal operator can match this pattern — it should not, because it lacks any mechanism to differentiate strata. FLUX.1 Kontext tests whether a strong flow-matching editor already achieves consistency through scale; if it does, our structural claim is undermined. RelationAdapter tests whether relation-aware conditioning substitutes for certification. The DRO ablation is the decisive control: if removing worst-stratum reweighting does not shrink the certified-stratum gain, then the DRO loss is not the causal driver and our design rationale fails. The calibration ablation tests whether the calibrated mask is necessary for the certificate to be non-vacuous: if the hard argmax mask yields the same stratified gain, then calibration is not needed and the load-bearing assumption is not addressed.

### Expected outcome and causal chain

**vs. VISTA** — On a case where a fast-moving foreground object occludes a static background, VISTA produces temporal flicker and style bleed across the occlusion boundary because its single unconstrained temporal operator averages attention over all pixels, smearing certified and uncertified motion. Our method instead partitions pixels by motion stratum and applies a per-stratum graph diffusion whose Lipschitz constant is bounded, so the occlusion boundary is handled by the fallback identity while the fast object stays in a certified stratum. We expect a noticeable temporal warp error gap on high-occlusion clips but near parity on slow, uniform-motion clips.

**vs. FLUX.1 Kontext** — On a case where style reference must persist across a multi-turn edit with large camera motion, Kontext produces drift in character and style identity because its flow-matching path has no per-pixel certificate and accumulates error across turns. Our method instead re-certifies each pixel's motion stratum at every denoising step and confines diffusion to same-stratum graphs, so error cannot propagate across stratum boundaries. We expect our advantage to grow with turn count and motion magnitude, appearing as a widening gap on multi-turn large-motion subsets and parity on single-turn static subsets.

**vs. RelationAdapter** — On a case where the source-target relation involves localized occlusion that RelationAdapter's relation conditioning does not spatially resolve, it produces inconsistent style transfer in the occluded region because its conditioning is global rather than per-pixel certified. Our method instead routes occluded pixels to the identity-preserving fallback and certifies the rest, so the failure region is explicitly excluded from the operator's guarantee rather than silently mis-stylized. We expect a localized error reduction concentrated in occluded regions with little change elsewhere.

### What would falsify this idea

If the temporal consistency gain over VISTA is uniform across ground-truth motion strata rather than concentrated in certified strata with parity in the uncertified fallback, then the certificate is vacuous and the structural claim is wrong. Likewise, if removing the DRO loss leaves the certified-stratum gain unchanged, the robustness mechanism is not doing the causal work we attribute to it. Additionally, if the calibrated mask does not improve the concentration of gain in certified strata compared to the hard argmax mask, then the calibration step is not addressing the load-bearing assumption and the certificate remains unverified.

## References

1. VISTA: Triplet-Supervised Video Style Transfer with Diffusion Transformers
2. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
3. Composer: Creative and Controllable Image Synthesis with Composable Conditions
4. RelationAdapter: Learning and Transferring Visual Relation with Diffusion Transformers
5. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
6. In-Context LoRA for Diffusion Transformers
7. OmniGen: Unified Image Generation
8. Boosting Latent Diffusion with Flow Matching

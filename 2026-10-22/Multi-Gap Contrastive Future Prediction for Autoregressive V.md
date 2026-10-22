# Multi-Gap Contrastive Future Prediction for Autoregressive Video Generation

## Motivation

Existing autoregressive video generation methods such as FCP (Looking Ahead) supervise future evolution using a single fixed temporal gap, which cannot guarantee that the learned latent representation separates true evolution dynamics from noise and short-term appearance changes. This shared residual gap across all branches is critical for robust autoregressive video generation, where distinguishing true evolution from noise is essential. The limitation is structural: a single gap provides only one contrastive signal, so the model can minimize the FCP loss by encoding short-term appearance or noise rather than evolution dynamics.

## Key Insight

Supervising future chunk prediction at multiple temporal gaps with a contrastive loss that treats different gaps as different views of the same evolution process forces the latent representation to encode gap-invariant evolution dynamics while discarding gap-specific noise and short-term appearance, because only true evolution is consistent across gaps.

## Method

**What it is**: Multi-Gap Contrastive Future Prediction (MGCFP) is an auxiliary training objective for autoregressive video diffusion models. It takes a context chunk and predicts future chunks at multiple temporal gaps (short, medium, long), using a contrastive loss to align representations of the same evolution across gaps while separating them from noise and short-term changes.

**Load-bearing assumption (stated explicitly)**: The component of the latent representation that is invariant across temporal gaps {1,2,4} is the true evolution dynamics, and the gap-specific residual is noise or short-term appearance, so that cross-gap InfoNCE alignment isolates dynamics rather than any other gap-consistent factor. This assumption fails when the video contains periodic/repetitive motion, constant-velocity translation, or static background, which are also trivially consistent across gaps; in that case the contrastive objective is equally minimized by encoding those gap-invariant factors. To mitigate this failure, we replace cross-gap negatives with negatives that are explicitly matched on gap-consistent nuisance factors (time-warped or phase-shifted futures, and dynamics-scrambled counterfactuals), and we use other-gap futures only as positives under a consistency term, not as negatives.

**How it works**:
```python
# Given: context chunk C_t, future chunks F_{t+g} for g in {g1, g2, g3} = {1, 2, 4}
# Model: autoregressive video diffusion model with encoder E and predictor P
# Training loop:
for each training video with chunks {C_1, ..., C_T}:
    sample context chunk C_t
    for each gap g in {1, 2, 4} (short, medium, long):
        sample future chunk F_{t+g}
        # Encode context and future
        z_context = E(C_t)
        z_future_g = E(F_{t+g})
        # Predict future representation from context
        z_pred_g = P(z_context, g)  # gap-conditioned predictor: 2-layer MLP, hidden=256, GeLU
        # Construct hard negatives matched on gap-consistent nuisance factors:
        #   - time-warped future: F_{t+g} warped by a random temporal scaling in [0.5, 2.0]
        #   - phase-shifted future: F_{t+g} shifted by a random phase in [0, T/2]
        #   - dynamics-scrambled counterfactual: F_{t+g} with physical parameters shuffled (e.g., gravity, friction)
        #   - noise-augmented version: F_{t+g} + Gaussian noise (σ=0.1)
        # Other-gap futures {F_{t+g'} for g'!=g} are used ONLY as positives under a consistency term,
        #   not as negatives.
        # Contrastive loss: pull z_pred_g towards z_future_g, push away from hard negatives
        L_contrast_g = InfoNCE(z_pred_g, z_future_g, negatives={E(F_{t+g}^{warped}), E(F_{t+g}^{phase}), E(F_{t+g}^{scrambled}), E(F_{t+g}+noise)})
        # Consistency term: align z_pred_g with z_future_g' for g'!=g (no repulsion)
        L_consistency_g = mean_{g'!=g} ||z_pred_g - z_future_g'||^2
    # Total loss: sum over gaps + standard denoising loss
    L_total = L_denoise + λ * mean_g (L_contrast_g + 0.1 * L_consistency_g)
```

**Why this design**: We chose multiple temporal gaps over a single fixed gap because a single gap provides only one contrastive signal, allowing the model to minimize the loss by encoding short-term appearance or noise that happens to correlate with the gap; multiple gaps force the representation to be consistent across different temporal scales, which is a structural property that only true evolution dynamics satisfy. We chose a contrastive loss with negative samples from other gaps and noise-augmented versions over a simple regression loss because regression alone does not explicitly penalize the model for encoding gap-specific information; the contrastive objective creates a repulsive force that separates evolution from noise. We chose a gap-conditioned predictor over a shared predictor because the evolution dynamics may manifest differently at different temporal scales, and conditioning on the gap allows the predictor to focus on the appropriate scale while the contrastive loss ensures the underlying representation is gap-invariant. The trade-off is increased training complexity and the need to tune the gap set and loss weighting, but this is justified by the stronger separation guarantee. This design is not a variant of Diffusion Forcing (which uses independent per-token noise levels) because our contrastive loss operates across temporal gaps rather than noise levels, and it is not a variant of FCP (which uses a single gap) because we explicitly enforce multi-gap consistency.

**Why it measures what we claim**: The contrastive loss L_contrast_g measures the separation of evolution dynamics from noise and short-term appearance because the InfoNCE objective with hard negatives (time-warped, phase-shifted, dynamics-scrambled futures, and noise-augmented versions) forces the representation z_pred_g to be close to the true future representation z_future_g and far from representations of nuisance factors; this assumes that true evolution dynamics are the only component consistent across different temporal gaps, which fails when the video contains repetitive or periodic motion where short-term appearance changes are also consistent across gaps, in which case the contrastive loss may reflect gap-invariant appearance rather than evolution. The gap-conditioned predictor P(z_context, g) measures the model's ability to extrapolate evolution at a specific temporal scale because conditioning on g allows the predictor to specialize, assuming that the evolution dynamics at gap g are predictable from the context; this assumption fails when the future is inherently stochastic or when the context lacks sufficient information, in which case the predictor may rely on noise. The use of multiple gaps {1,2,4} measures the multi-scale consistency of evolution because different gaps capture different temporal extents, assuming that true evolution manifests consistently across these scales; this assumption fails when the video contains events at a specific temporal scale that are not present at others, in which case the contrastive loss may penalize legitimate scale-specific dynamics.

## Contribution

['A novel training objective, Multi-Gap Contrastive Future Prediction (MGCFP), that supervises autoregressive video diffusion models at multiple temporal gaps with a contrastive loss to separate evolution dynamics from noise and short-term appearance changes.', 'A design principle that multi-gap contrastive supervision yields a more structured latent representation than single-gap supervision, as evidenced by improved physical plausibility and reduced noise sensitivity in generated videos.', 'An empirical demonstration that MGCFP can be integrated into existing autoregressive video diffusion training without additional pretrained encoders, using only training videos for supervision.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Physion-style physical dynamics video set | Contains varied temporal scales and stochastic futures. |
| Dataset | Synthetic dataset with gap-invariant appearance but non-gap-invariant dynamics (e.g., moving object with periodic texture) | Tests whether cross-gap negatives are trivially satisfied by appearance. |
| Primary metric | Physical plausibility (human 2AFC) | Directly detects evolution-vs-noise separation. |
| Secondary metric | CKA between E(F_{t+g}) across g | Measures gap-invariance of target encoder directly. |
| Baseline | Standard teacher-forced AR diffusion | No contrastive multi-gap objective. |
| Baseline | Single-gap FCP | One temporal gap only. |
| Baseline | Diffusion Forcing | Per-token noise, no gap contrast. |
| Baseline | Multi-scale regression (predict future at gaps {1,2,4} with L2 loss) | Tests if contrastive component is just multi-scale supervision. |
| Ablation-of-ours | MGCFP with gaps {1,1,1} (single gap) | Isolates multi-gap consistency effect. |
| Ablation-of-ours | MGCFP with cross-gap negatives (original) vs. hard negatives (ours) | Tests if hard negatives improve separation. |
| Ablation-of-ours | MGCFP with λ=0.1, 0.5, 1.0 | Sensitivity to loss weighting. |
| Ablation-of-ours | MGCFP with gap sets {1,2,4}, {1,2,3}, {1,3,6} | Robustness to gap choice. |

### Why this setup validates the claim

The central claim is that enforcing consistency of predicted future representations across multiple temporal gaps separates true evolution dynamics from noise and short-term appearance, and that this separation improves physical plausibility of long-horizon autoregressive video generation. The Physion-style dataset is the right testbed because it contains both deterministic physical events (e.g., falling, collision) that should be gap-invariant and stochastic or repetitive events (e.g., bouncing, periodic motion) where the assumption may fail, allowing us to see whether the gain is concentrated where the mechanism predicts. The synthetic dataset with gap-invariant appearance but non-gap-invariant dynamics directly tests the load-bearing assumption: if cross-gap negatives are trivially satisfied by appearance, our method will show no advantage over the multi-scale regression baseline on this dataset. The primary metric, physical plausibility via human 2AFC, directly measures whether generated futures respect evolution dynamics rather than merely looking sharp; FID or pixel error would not distinguish a plausible physical trajectory from a noisy but visually crisp one. The secondary metric, CKA between E(F_{t+g}) across g, verifies that the target encoder is evolution-dominated before relying on the contrastive loss. The standard teacher-forced AR diffusion baseline tests the sub-claim that any auxiliary future-prediction objective helps; the single-gap FCP baseline tests whether multiple gaps are necessary beyond one gap; the Diffusion Forcing baseline tests whether the gain is just from noise-level conditioning rather than gap contrast; the multi-scale regression baseline tests whether the contrastive component is just multi-scale supervision in disguise. The ablation with gaps {1,1,1} collapses our method to a single-gap contrastive objective, isolating the contribution of multi-gap consistency. The ablation with cross-gap negatives vs. hard negatives tests whether the principled negative construction is necessary. The λ and gap-set sweeps test robustness. If the mechanism is correct, we expect the largest improvement over all baselines on videos with deterministic, multi-scale dynamics, and parity or slight degradation on strongly periodic or stochastic subsets where gap-invariance is violated. This pattern would be a falsifiable signature: a uniform gain across all subsets would suggest the improvement comes from something other than the claimed separation.

### Expected outcome and causal chain

**vs. Standard teacher-forced AR diffusion** — On a case where the context shows an object mid-fall and the future at gap 4 requires extrapolating acceleration, the baseline produces a future that either freezes the object or drifts into noise because its training signal only enforces per-chunk denoising without any explicit pressure to encode evolution across time. Our method instead pulls the predicted gap-4 representation toward the true future representation and pushes it away from hard negatives (time-warped, phase-shifted, dynamics-scrambled futures, and noise-augmented encodings), so the model must encode the acceleration trend rather than appearance. We expect a noticeable gap in physical plausibility on deterministic-motion subsets (e.g., falling, collision) but near parity on stochastic subsets (e.g., random flicker), with the overall 2AFC preference favoring our method by a clear margin on the former.

**vs. Single-gap FCP** — On a case where short-term appearance change (e.g., lighting flicker) correlates with a fixed gap of 1, the single-gap baseline can minimize its contrastive loss by encoding that flicker as the “future” signal, so its long-horizon rollouts show plausible short-term texture but physically wrong trajectories. Our method instead uses gaps {1,2,4} with hard negatives and a consistency term, forcing the representation to be consistent only for components that persist across scales, so flicker is separated out. We expect our method to outperform single-gap FCP specifically on long-horizon physical plausibility (gap-4 predictions) while matching it on short-horizon appearance, producing a widening advantage as rollout length increases.

**vs. Diffusion Forcing** — On a case where the model must generate a long rollout with variable noise levels, Diffusion Forcing produces futures that are locally coherent but drift in global dynamics because its conditioning is on per-token noise levels, not on temporal-gap consistency. Our method instead adds a contrastive loss across temporal gaps that directly aligns predicted evolution representations, so the global trajectory remains anchored. We expect our method to show better long-horizon stability (fewer physical violations after several chunks) while Diffusion Forcing may match on short rollouts, yielding a crossover advantage for ours as horizon grows.

**vs. Multi-scale regression** — On a synthetic dataset where appearance is gap-invariant but dynamics are not, the multi-scale regression baseline can minimize its L2 loss by predicting the gap-invariant appearance component, ignoring dynamics. Our method, with hard negatives that share appearance but differ in dynamics, must separate dynamics from appearance to minimize the contrastive loss. We expect our method to outperform multi-scale regression on physical plausibility on this synthetic dataset, demonstrating that the contrastive component is not just multi-scale supervision in disguise.

### What would falsify this idea

If our method’s gain in physical plausibility is uniform across all subsets—including the periodic and stochastic subsets where gap-invariance is predicted to fail—rather than concentrated on deterministic multi-scale dynamics, then the central claim that multi-gap contrast separates evolution from noise is wrong. Similarly, if the single-gap ablation {1,1,1} matches our multi-gap performance on long-horizon physical plausibility, or if the multi-scale regression baseline matches our method on the synthetic dataset, the multi-gap design is not doing the claimed work.

## References

1. Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation
2. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
3. Masked Diffusion Transformer is a Strong Image Synthesizer
4. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
5. Diffusion-LM Improves Controllable Text Generation
6. Continuous diffusion for categorical data
7. DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models
8. Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise

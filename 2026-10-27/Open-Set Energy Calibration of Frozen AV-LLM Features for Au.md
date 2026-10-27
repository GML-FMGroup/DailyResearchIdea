# Open-Set Energy Calibration of Frozen AV-LLM Features for Audio-Visual Deepfake Detection

## Motivation

Existing audio-visual deepfake detectors, including AV-FABLE's NFSR readouts, train and evaluate on a fixed, closed set of manipulation types and assume the frozen AV-LLM feature substrate is already separable for unseen forgery mechanisms. This assumption is structurally unsound: frozen features encode manipulation evidence (as NFSR demonstrates) but are never certified for open-set separability, so readout heads trained on known manipulation types can assign high confidence to novel forgeries whose signatures lie outside the training distribution. The field's trajectory toward foundation-model-based forensics makes this a recurring bottleneck across the lineage, not a local flaw of one tree.

## Key Insight

The frozen AV-LLM feature space admits a manipulation-type-invariant energy geometry: because forgery signatures are localized directions in this space, a Mahalanobis energy calibrated on held-out manipulation types yields a decision boundary that is monotone in distributional distance for unseen mechanisms, a property that closed-set softmax readouts provably lack.

## Method

## AV-OSEC: Audio-Visual Open-Set Energy Calibration

**(A) What it is.** AV-OSEC is a post-hoc layer that takes frozen AV-LLM segment embeddings (audio and visual streams) and outputs a per-segment novelty score plus a calibrated detection decision. It learns a manipulation-type-agnostic energy surface via leave-one-manipulation-type-out (LOMO) training and recalibrates online using unlabeled long-form video segments. **Load-bearing assumption (stated explicitly):** the frozen AV-LLM feature space admits a manipulation-type-invariant, unimodal class-conditional Gaussian geometry such that a Mahalanobis energy calibrated on held-out manipulation types is monotone in distributional distance for unseen forgery mechanisms. This assumption is not taken for granted: it is operationalized by a per-class Gaussian mixture (K=4 components, fitted by EM with covariance shrinkage lambda=0.1) in place of a single Gaussian, and by a LOMO cross-type ranking loss that explicitly trains held-out-type energy to exceed seen-type energy; the assumption is verified empirically by the diagnostic in (D) (held-out-type energy ranking vs. measured MMD distance) and by the ablation that removes the ranking term.

**(B) How it works.**

```pseudocode
# Inputs: frozen AV-LLM encoder f; training set D = {(v_i, y_i, m_i)} where
#   y_i in {real, fake}, m_i is manipulation type; unlabeled long-form video U.
# Hyperparameters: temperature T=1.0, energy margin gamma=2.0, EMA rate beta=0.9,
#   covariance shrinkage lambda=0.1, replay buffer size B=512,
#   GMM components K=4, ranking weight rho=1.0, ranking margin delta=1.0.

1. Embed: z_i = [f_audio(v_i); f_visual(v_i)] in R^d  (frozen, no fine-tuning)

2. LOMO energy training:
   for each held-out manipulation type m* in {types}:
       fit class-conditional Gaussian mixtures on real and fake embeddings from D \ {m*}
         {mu_real^k, Sigma_real^k}_{k=1..K}, {mu_fake^k, Sigma_fake^k}_{k=1..K}
         via EM (K=4, 50 iterations, covariance shrinkage lambda=0.1)
       define energy E(z) = -T * logsumexp( -D_real(z)/T, -D_fake(z)/T )
         where D_c(z) = -T * logsumexp_k( -0.5 * (z - mu_c^k)^T (Sigma_c^k)^{-1} (z - mu_c^k) )
       train a linear calibration head g(E(z); theta) on D \ {m*} with
         loss = BCE(g, y) + gamma * max(0, E_fake - E_real + margin)
                + rho * max(0, delta - (E_heldout - E_seen))
         where E_heldout is the mean energy of held-out-type segments (from a
         small held-out-type validation split, 10% of m* segments) and E_seen is
         the mean energy of seen-type fake segments; this cross-type ranking term
         explicitly enforces held-out-type energy > seen-type energy.
       store theta_{m*}
   final theta = mean over held-out folds

3. Online test-time recalibration (unlabeled long-form segments):
   for each incoming segment z_t from U:
       compute E_t = E(z_t) using current mu, Sigma
       update {mu_real^k, Sigma_real^k}, {mu_fake^k, Sigma_fake^k} via EMA with rate beta
         using a replay buffer of the B most recent segments, weighted by
         pseudo-labels from g(E_t) > 0.5; report pseudo-label class ratio
         (fraction real vs. fake) as a diagnostic.
       update theta via one gradient step on the entropy of g(E_t)

4. Output: per-segment novelty score s_t = E_t and calibrated decision p_t = g(E_t; theta)
```

**(C) Why this design.**

We chose a Mahalanobis energy over a learned discriminative novelty head because the frozen AV-LLM feature space is high-dimensional with limited manipulation-type coverage, and a Gaussian class-conditional energy has a closed-form decision boundary that does not require fitting a separate classifier for each unseen type; the cost is that the Gaussian assumption may be violated for multi-modal forgery distributions, which we mitigate with covariance shrinkage (lambda=0.1) and a per-class Gaussian mixture (K=4) that captures multi-modality. We chose leave-one-manipulation-type-out training over standard cross-validation because the goal is to explicitly optimize the readout to preserve separability margins for held-out forgery mechanisms, accepting the cost of reduced training data per fold and higher variance in the calibration head; the cross-type ranking term (rho=1.0, delta=1.0) makes this objective explicit rather than assumed. We chose online EMA recalibration with a replay buffer over full test-time adaptation because long-form videos exhibit distribution shift across time (e.g., changing lighting, codec artifacts) and full adaptation would require labels; the replay buffer bounds memory and the EMA rate beta=0.9 provides a stability-plasticity trade-off, accepting the cost that abrupt shifts may be under-adapted. A domain expert would not describe this method as a variant of standard open-set recognition (e.g., OpenMax, energy-based OOD) because the LOMO objective explicitly trains the calibration head to preserve margins for held-out manipulation types, a manipulation-type-invariant property that standard OOD methods do not enforce.

**(D) Why it measures what we claim.**

The Mahalanobis energy E(z) measures distributional distance from the training feature manifold because the Gaussian mixture class-conditional assumption makes E(z) equivalent to the negative log-likelihood of z under the real/fake mixture; this assumption fails when unseen manipulation types produce feature distributions that are non-Gaussian (e.g., multi-modal), in which case E(z) reflects distance to the nearest Gaussian mode rather than true novelty. The LOMO training loss measures manipulation-type-invariant separability because holding out an entire manipulation type forces the calibration head to rely on features that generalize across types; this assumption fails when held-out types share low-level artifacts with training types, in which case the loss reflects within-type separability rather than cross-type invariance. The cross-type ranking term measures the monotonicity property because it directly penalizes held-out-type energy not exceeding seen-type energy; this assumption fails when the held-out-type validation split is too small or unrepresentative, in which case the ranking term reflects noise rather than true cross-type ordering. The online EMA recalibration measures test-time distribution shift because the unlabeled segments provide a running estimate of the current feature distribution; this assumption fails when the unlabeled stream is dominated by real videos, in which case the recalibration biases the energy surface toward the real manifold and reduces sensitivity to novel forgeries (diagnosed by reporting the pseudo-label class ratio).

**(E) Contribution.**
(1) AV-OSEC, a post-hoc open-set calibration layer for frozen AV-LLM features that combines Mahalanobis energy scoring with leave-one-manipulation-type-out training to explicitly optimize for unseen forgery mechanisms. (2) A design principle showing that manipulation-type-invariant separability can be enforced without retraining the feature extractor by calibrating a closed-form energy surface on held-out types. (3) An online recalibration procedure using unlabeled long-form video segments that adapts to distribution shift while preserving open-set sensitivity.

## Contribution

(1) AV-OSEC, a post-hoc open-set calibration layer for frozen AV-LLM features that combines Mahalanobis energy scoring with leave-one-manipulation-type-out training to explicitly optimize for unseen forgery mechanisms. (2) A design principle showing that manipulation-type-invariant separability can be enforced without retraining the feature extractor by calibrating a closed-form energy surface on held-out types. (3) An online recalibration procedure using unlabeled long-form video segments that adapts to distribution shift while preserving open-set sensitivity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AV-FABLE long-form AV deepfake benchmark | Long-form segments expose temporal distribution shift and unseen manipulation types |
| Primary metric | AUC on held-out manipulation types | Directly measures cross-type separability, not within-type memorization |
| Baseline | Zero-shot AV-LLM prompting | Tests whether frozen features already separate unseen types |
| Baseline | Supervised NFSR readout on frozen features | Tests whether standard discriminative readout generalizes across types |
| Baseline | Energy-based OOD (OpenMax-style) | Tests whether generic open-set scoring suffices without LOMO |
| Baseline | LOMO-trained OpenMax (same training, different energy) | Isolates what the energy geometry adds beyond the training scheme |
| Ablation-of-ours | AV-OSEC without LOMO (standard CV) | Isolates the contribution of manipulation-type-invariant training |
| Ablation-of-ours | AV-OSEC without cross-type ranking term (rho=0) | Isolates the contribution of the explicit ranking loss |
| Ablation-of-ours | AV-OSEC with single Gaussian (K=1) | Tests the multi-modal Gaussian mixture assumption |
| Ablation-of-ours | AV-OSEC without online EMA (static) | Separates offline LOMO from online recalibration |
| Diagnostic | Pseudo-label class ratio over time | Detects real-dominated stream failure mode |
| Diagnostic | Held-out-type energy vs. MMD distance correlation | Verifies monotonicity assumption empirically |

### Why this setup validates the claim

The central claim is that a closed-form Mahalanobis energy surface, trained with leave-one-manipulation-type-out and recalibrated online, generalizes to unseen forgery mechanisms better than readouts that never see a type held out. AV-FABLE is the right testbed because it supplies long-form videos with multiple manipulation types and temporal distribution shift, so held-out-type AUC can be computed cleanly. The zero-shot AV-LLM baseline tests whether the frozen feature space is already sufficient without any readout; the supervised NFSR readout tests whether a standard discriminative head closes the gap; the energy-based OOD baseline tests whether generic open-set scoring suffices without LOMO; the LOMO-trained OpenMax baseline tests whether the specific Mahalanobis energy geometry adds anything beyond the LOMO training scheme itself. The ablation (same energy surface, standard cross-validation instead of LOMO) is the diagnostic: if our gain comes from type-invariant training, removing LOMO should collapse held-out-type AUC toward the NFSR baseline, while leaving within-type AUC roughly intact. The cross-type ranking ablation (rho=0) tests whether the explicit monotonicity term is necessary. The single-Gaussian ablation (K=1) tests the multi-modal assumption. The static-vs-EMA ablation separates the offline and online contributions. AUC on held-out types is the right metric because it is threshold-free and directly reflects separability of unseen mechanisms, which is exactly the property the method claims to preserve. The diagnostic correlating held-out-type energy with MMD distance between held-out and training type embeddings empirically verifies the monotonicity assumption: if the assumption holds, higher energy should correspond to larger MMD.

### Expected outcome and causal chain

**vs. Zero-shot AV-LLM prompting** — On a case where a held-out manipulation type (e.g., a lip-sync forgery unseen in calibration) produces plausible per-modality features but an inconsistent audio-visual temporal alignment, zero-shot prompting produces near-chance or unstable judgments because it has no calibrated decision surface and relies on the LLM's prior over forgery vocabulary. Our method instead projects the segment embedding onto a class-conditional Gaussian mixture energy surface whose margin was explicitly trained to survive type holdout, so the inconsistent alignment lands far from both real and fake manifolds. We expect a large held-out-type AUC gap (roughly 0.15–0.25) with near-parity on seen types.

**vs. Supervised NFSR readout** — On a case where the held-out type shares low-level codec artifacts with a training type, the NFSR readout latches onto those within-type cues and produces confident but wrong labels on the new type because its discriminative boundary was fit only to seen-type statistics. Our method instead uses a closed-form energy whose decision boundary is set by class-conditional covariances and a cross-type ranking loss trained across held-out folds, so shared low-level cues cannot dominate the score. We expect the NFSR readout to show a sharp AUC drop on the held-out-type subset while AV-OSEC degrades gracefully, yielding a noticeable gap concentrated on that subset.

**vs. Energy-based OOD (OpenMax-style)** — On a case where the unlabeled long-form stream drifts (changing lighting, re-encoding) mid-video, the static OpenMax energy produces miscalibrated scores because its tail model is frozen at training time and cannot track the shifting feature distribution. Our method instead applies EMA recalibration with a bounded replay buffer, so the Gaussian means and covariances track the drift while the LOMO-trained margin preserves open-set sensitivity. We expect OpenMax to show rising false-positive rates on real segments after the drift point, while AV-OSEC holds a stable operating point, producing a moderate but consistent AUC advantage on the drifted portion.

**vs. LOMO-trained OpenMax** — On a case where the held-out type's signature lies in a direction the covariance never saw, the LOMO-trained OpenMax (same training scheme, but Weibull tail model instead of Mahalanobis energy) may still fail because its tail model is fit to seen-type activation norms and does not explicitly penalize held-out-type energy. Our method instead uses the Mahalanobis energy with the cross-type ranking term, which directly enforces held-out-type energy > seen-type energy. We expect AV-OSEC to outperform LOMO-trained OpenMax on held-out-type AUC, with the gap concentrated on types whose MMD distance from training types is largest, confirming that the energy geometry adds value beyond the training scheme.

### What would falsify this idea

If AV-OSEC's gain over the NFSR readout and OpenMax is uniform across seen and held-out manipulation types rather than concentrated on held-out types, the claim that LOMO enforces type-invariant separability is wrong. Likewise, if removing LOMO (the ablation) leaves held-out-type AUC unchanged, or if the cross-type ranking term (rho=0) does not reduce held-out-type AUC, the central mechanism is not doing the work we attribute to it. Additionally, if the correlation between held-out-type energy and MMD distance is near zero, the monotonicity assumption is empirically false.

## References

1. AV-FABLE: Audio-Visual Manipulation Forensics in Long-Form Videos

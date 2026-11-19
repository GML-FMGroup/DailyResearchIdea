# Tail-Coverage Critics for Parallel Single-Pass Image Generation

## Motivation

Parallel single-pass generators such as Jeveration condition all spatial positions on a shared random state to obtain global coherence, but this conditioning only guarantees that each individual draw is internally consistent — it provides no mechanism that forces the aggregate distribution of draws to cover the tails of the learned discrete image-code prior. The structural root cause is that Jeveration inherits the DnD-Transformer assumption that sequential autoregressive factorization supplies code-distribution fidelity, yet when sequential autoregression is replaced by parallel prediction, that factorization no longer exists and the tail of the code prior is silently dropped by mode-seeking parallel decoding. As a result, Jeveration reports only 'competitive' image quality with no guarantee that rare codes and rare compositions remain reachable across draws.

## Key Insight

A critic trained to estimate each generated image's probability under the frozen discrete code prior turns tail support into a per-sample, differentiable training signal, so the parallel decoder can be penalized exactly when a draw falls outside the prior's high-density region — a constraint that shared-random-state conditioning alone cannot express because it operates within a single draw, not across the draw distribution.

## Method

**What it is.** We introduce TCC (Tail-Coverage Critic), a training-time regularizer for Jeveration-style parallel single-pass image generation. TCC takes as input a batch of generated images (produced by the frozen VLM backbone under shared-random-state conditioning), maps each image to its discrete code sequence via the frozen VQ encoder, and outputs a scalar coverage penalty that is added to the standard image-supervision loss, plus a per-sample coverage mask used for stratified replay.

**How it works.**

```pseudocode
# Frozen components: VLM backbone F, VQ encoder E, VQ codebook C (K codes)
# Learned prior: p_theta(code_seq) — a small autoregressive model over codes,
#   pre-trained on the code sequences of the training set (Jeveration's
#   learned discrete code distribution), then FROZEN during TCC training.
# Trainable: parallel decoder D (Jeveration head), critic Q_phi.
# Critic architecture: 2-layer MLP, hidden=256, GeLU activation, input = flattened
#   code sequence z (length L, embedding dim 64 per code), output = scalar log-prob estimate.
# Held-out real-code density: a second frozen autoregressive model p_real(code_seq)
#   trained on a disjoint 10% split of the training code sequences (same architecture as p_theta).

Input: image batch x, shared random state s ~ N(0, I), coverage threshold tau = 0.15,
       critic warmup steps W = 2000, penalty weight lambda = 0.5,
       off-support guard threshold gamma = 0.05 (5th percentile of p_real on held-out real codes)

for each training step t:
    # 1. Parallel generation under shared state
    x_hat = D(F, s)                       # one forward pass, all positions

    # 2. Map generated image to discrete codes
    z_hat = E(x_hat)                      # code sequence, length L

    # 3. Exact prior log-probability (cheap for a single sequence)
    logp_exact = log p_theta(z_hat)       # scalar per sample, computed by frozen AR prior

    # 4. Critic estimates log-probability under the frozen code prior
    logp_hat = Q_phi(z_hat)               # scalar per sample; trained to match
                                          # log p_theta(z_hat) via MSE on a
                                          # replay buffer of real code sequences

    # 5. Off-support guard: reject samples whose code sequence has low likelihood
    #    under the held-out real-code density p_real
    logp_real = log p_real(z_hat)         # scalar per sample
    in_support = 1[ logp_real >= gamma ]  # 1 if generated code lies in real-code support

    # 6. Coverage penalty: penalize samples whose estimated prior log-prob
    #    falls below the tau-quantile of the prior's own log-prob distribution
    #    (computed once from 50k samples of p_theta, stored as logp_tau)
    mask = 1[ logp_hat < logp_tau ] * in_support   # tail indicator, gated by support
    L_cov = mean( mask * (logp_tau - logp_hat) )

    # 7. Total loss
    L = L_img(x_hat, x) + lambda * L_cov
    update D via gradient of L; update Q_phi via MSE to log p_theta on
    fresh real code sequences (critic is trained online, not frozen)
```

Hyperparameters: tau = 0.15 (fraction of prior mass treated as tail), lambda = 0.5, critic warmup W = 2000 steps during which lambda = 0 so the critic is calibrated before penalizing, off-support guard threshold gamma = 0.05 (5th percentile of p_real on held-out real codes), critic architecture 2-layer MLP with hidden=256 and GeLU, code embedding dim 64.

**Why this design.** We chose a learned critic Q_phi over exact evaluation of p_theta on generated codes because p_theta is an autoregressive model over length-L sequences and exact log-probability is cheap for a single sequence but the critic lets us amortize and smooth the estimate across the batch, accepting the cost that Q_phi introduces approximation error (mitigated by warmup and by the MSE objective to p_theta). We chose a quantile-thresholded hinge penalty over a full KL or MMD between generated and prior code distributions because the motivation is specifically tail support, not mean-matching — a KL would be dominated by head mass and would not penalize a decoder that ignores the tail, accepting the cost that the hinge is non-smooth at the threshold (handled by the mask being treated as a constant in the backward pass). We chose to freeze p_theta during TCC training rather than co-train it with D because co-training would let the prior drift toward the decoder's mode-seeking distribution, defeating the coverage target, accepting the cost that the prior may be slightly mismatched to the frozen VLM backbone. A domain expert would not describe TCC as a variant of Jeveration: Jeveration's shared-random-state conditioning is a within-draw coherence mechanism, whereas TCC is an across-draw support constraint enforced through a critic on the frozen code prior — the two operate on different statistical objects and neither subsumes the other.

**Why it measures what we claim.** The quantity `logp_hat` measures tail coverage of the discrete code prior because the assumption that a critic trained by MSE to `log p_theta(z)` on real code sequences is a consistent estimator of the prior log-probability for generated codes makes `logp_hat` equivalent to the true prior log-probability; this assumption fails when the generated code distribution lies far outside the real-code training support of Q_phi (extrapolation regime), in which case `logp_hat` reflects the critic's extrapolation bias rather than true tail support. The quantity `mask = 1[logp_hat < logp_tau]` measures whether a draw falls in the tail of the prior because the assumption that the tau-quantile of the prior's own log-prob distribution is a valid tail boundary makes the indicator equivalent to a tail-membership test; this assumption fails when the prior is multimodal with heavy tails, in which case the indicator reflects a single global quantile rather than per-mode tail coverage. The quantity `L_cov` measures the degree of tail violation because the assumption that the hinge magnitude `(logp_tau - logp_hat)` is monotone in severity of tail violation makes the penalty equivalent to a coverage deficit; this assumption fails when `logp_hat` is saturated by critic error, in which case `L_cov` reflects critic noise rather than coverage deficit. The quantity `in_support = 1[logp_real >= gamma]` measures whether a generated code lies within the real-code support because the assumption that the held-out real-code density p_real assigns high likelihood to exactly the region where Q_phi was trained makes the indicator equivalent to a support-overlap test; this assumption fails when p_real itself is overconfident or when generated codes are near the support boundary, in which case `in_support` reflects the sharpness of p_real rather than true support overlap.

## Contribution

['(1) TCC, a tail-coverage critic regularizer that converts the frozen discrete code prior into a per-sample differentiable training signal for parallel single-pass image generators, with a quantile-thresholded hinge penalty and critic warmup schedule.', '(2) A design principle for parallel (non-autoregressive) image generation: distributional coverage of the code prior must be enforced by an across-draw constraint because the within-draw shared-random-state conditioning that provides coherence cannot express tail support.', '(3) A stratified replay buffer of real code sequences used to keep the critic calibrated against the frozen prior, enabling stable training without co-training the prior.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Text-to-image prompts with rich text/graphic elements (e.g., ISG-Bench text/graphic subset, 1,150 samples) | Stresses rare code-tail modes where coverage fails |
| Primary metric | Tail-recall of generated code sequences vs. frozen prior (fraction of draws with logp_exact below logp_tau) | Directly measures the support property TCC targets |
| Secondary metric | GenEval score (object presence, counting, color, position) and T2I-CompBench rare-glyph accuracy | Community-relevant outcome for tail coverage |
| Baseline | Jeveration (single-pass, no regularizer) | Isolates effect of TCC on shared-state parallel generation |
| Baseline | Jeveration + KL-to-prior regularizer (weight 0.5) | Tests whether tail-specific penalty beats mean-matching |
| Baseline | Unified autoregressive image model (e.g. ELM-style) | Tests whether single-pass parallel decoding is the lever |
| Ablation-of-ours | TCC with critic disabled (exact p_theta) | Isolates critic approximation cost from coverage effect |
| Ablation-of-ours | TCC with off-support guard disabled (gamma = 0) | Tests whether support gating changes tail-recall |
| Ablation-of-ours | TCC with critic trained on mixture of real and generated codes (density-ratio objective) | Tests whether critic calibration improves tail-recall |

### Why this setup validates the claim

The central claim is that a tail-covering critic regularizer raises the probability that Jeveration-style parallel draws land in the low-density tail of the frozen code prior, without degrading mean image quality. The dataset is chosen because prompts with text and graphical elements are exactly where tail code modes (rare glyph and layout configurations) are needed, so tail-recall is the metric that would move if the mechanism works. Jeveration-without-TCC is the baseline that tests the sub-claim that shared-state conditioning alone does not enforce across-draw support. The KL-regularized baseline tests the sub-claim that a quantile-thresholded hinge is necessary rather than any prior-matching term, because KL is head-dominated. The unified autoregressive baseline tests whether the effect is specific to parallel single-pass decoding rather than generation generally. The exact-prior ablation tests whether the critic's approximation is benign, which is the load-bearing assumption in the "why it measures what we claim" argument. The off-support-guard ablation tests whether the support-overlap assumption is necessary for the critic to act as a faithful prior log-probability estimator. The mixture-trained critic ablation tests whether a density-ratio objective reduces extrapolation bias. Together these form a falsifiable test: if tail-recall improves only in the TCC arm and concentrates on tail-heavy subsets, the causal story holds.

### Expected outcome and causal chain

**vs. Jeveration (no regularizer)** — On a prompt requiring rare glyph layouts, the baseline's parallel predictions collapse onto high-density code modes because shared-state conditioning enforces within-draw coherence but imposes no across-draw support constraint, so tail modes are systematically underdrawn. Our method adds a hinge penalty whenever the critic's log-prob estimate falls below the tau-quantile, pushing the decoder to cover those tail modes. We expect a clear tail-recall gap on text/graphic-heavy subsets with near-parity on natural-image subsets where the prior is unimodal.

**vs. Jeveration + KL-to-prior regularizer** — On the same rare-glyph prompt, the KL baseline is dominated by head mass, so it barely changes tail draw frequency while slightly over-regularizing common modes. Our quantile-thresholded hinge instead ignores head mass entirely and penalizes only tail violations, so we expect the KL arm to show marginal tail-recall gain and a small head-quality cost, while TCC shows a larger tail-recall gain with no head-quality cost.

**vs. Unified autoregressive image model** — On a prompt needing coherent global layout, the autoregressive baseline generates sequentially and can cover tail modes but is far slower and its coverage comes from sampling variance rather than an explicit support constraint. Our method achieves comparable tail-recall in a single parallel pass, so we expect a speed advantage of roughly two orders of magnitude at matched tail-recall, with the autoregressive model showing no tail-recall advantage once compute is controlled.

**vs. TCC with exact prior (critic disabled)** — On generated codes far outside the real-code training support, the critic may extrapolate, so the ablation tests whether the critic's approximation is benign. We expect the exact-prior arm to show slightly higher tail-recall and lower variance, with the gap small enough to confirm the critic is a faithful surrogate rather than the source of any gain.

**vs. TCC with off-support guard disabled (gamma = 0)** — On generated codes that fall outside the real-code support, the unguarded critic may assign unreliable log-probabilities, so the penalty signal is noisy exactly where it should act. Our guarded version rejects those samples, so we expect the guarded arm to show higher tail-recall and lower variance than the unguarded arm, with the gap concentrated on prompts that produce rare glyph layouts.

**vs. TCC with mixture-trained critic (density-ratio objective)** — On generated codes that lie near the support boundary, a critic trained only on real codes may extrapolate, whereas a critic trained on a mixture of real and generated codes with a log-density-ratio objective is calibrated on both. We expect the mixture-trained critic to show higher tail-recall and lower calibration error (measured as MSE between Q_phi(z) and log p_theta(z) on held-out generated codes) than the real-only critic, with the gap largest for samples near the support boundary.

### What would falsify this idea

If the tail-recall gain from TCC is uniform across all subsets rather than concentrated on the tail-heavy text/graphic subsets where the predicted failure mode occurs, the mechanism is not doing what we claim. Likewise, if the exact-prior ablation shows a large gap over the critic arm, or if the off-support guard ablation shows no difference, the critic approximation — not tail coverage — is driving the result.

## References

1. Jeveration: System One Models Now Generate

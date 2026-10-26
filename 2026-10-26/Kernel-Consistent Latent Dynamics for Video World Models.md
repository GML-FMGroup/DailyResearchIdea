# Kernel-Consistent Latent Dynamics for Video World Models

## Motivation

Both branches of latent video world modeling cap their achievable accuracy and mutual-information objective at the capacity of a fixed neural approximator: DreamGen's diffusion generator can only reproduce dynamics its finite network and dataset encode, so physical-law adherence degrades outside the training distribution, and Contrastive World Models' Deep InfoMax bound is computed against a frozen feature space whose value cannot increase as the latent belief sharpens, because the estimator itself is a fixed-capacity function. The structural problem is that the transition operator and its objective are both parameterized by the same bounded network, so there is no mechanism by which a sharper posterior can translate into a tighter objective or more accurate prediction.

## Key Insight

Replacing the parametric transition with a nonparametric kernel-weighted operator whose bandwidth is tied to the posterior precision makes predictive accuracy and the transition's mutual-information term depend only on the spread of the belief, not on the capacity of any backbone, so refining the posterior strictly shrinks both the prediction error and the MI gap.

## Method

**What it is.** KCLD (Kernel-Consistent Latent Dynamics) is a video world model whose latent transition is a nonparametric Nadaraya-Watson kernel operator over a growing memory of encoded latent-state/action pairs, coupled to the posterior by a precision-matched bandwidth with a density floor. Inputs: a sequence of frames and actions; output: a predicted next latent, a predictive distribution, and a belief-refinement signal used to sharpen the posterior.

**How it works.**
```pseudocode
# Encoder: fixed pretrained ViT patch encoder (I-JEPA-style), frozen
z_t = Encoder(o_t)                     # patch-mean latent, dim d=32 (fixed for kNN MI)

# Posterior over latent state: Gaussian, precision Lambda_t
mu_t, Lambda_t = Posterior(z_{<=t}, a_{<t})   # diagonal + low-rank (rank=4)

# Kernel transition operator (nonparametric, no learned transition weights)
for each memory key (z_i, a_i, z_i') in M:      # M grows, no capacity cap; reservoir cap 1e5
    w_i = exp( -||z_t - z_i||^2 / (2 h_t^2) - ||a_t - a_i||^2 / (2 h_a^2) )
    # bandwidth: precision prior with density floor
    h_t^2 = max( c / trace(Lambda_t), h_min^2 )   # c = 1.0 fixed; h_min = k-th NN distance in M
z_hat_{t+1} = sum_i w_i z_i' / sum_i w_i         # Nadaraya-Watson estimate
Sigma_hat    = sum_i w_i (z_i' - z_hat)^2 / sum_i w_i   # local spread

# Consistency regularizer: one-sided penalty for over-confidence only
L_cons = || max(0, Lambda_t^{-1} - Sigma_hat) ||_F^2   # penalizes Sigma_hat < Lambda_t^{-1}

# MI term: kernel-based, capacity-free
L_mi = - I_hat( z_hat_{t+1} ; z_{t+1} )        # kNN kernel MI estimator, k=5, d=32

# Belief refinement: posterior update uses kernel prediction as prior mean
mu_{t+1}, Lambda_{t+1} = BayesUpdate(mu=z_hat_{t+1}, prior_prec=Sigma_hat^{-1}, obs=z_{t+1})

# Memory insertion: append (z_t, a_t, z_{t+1}) after each step, with dedup radius r=0.05
```

**Why this design.** We chose a Nadaraya-Watson kernel operator over a learned transition network because the kernel estimate's error is governed by bandwidth and memory density rather than parameter count, accepting the cost that memory grows with trajectory length and requires a dedup radius (r=0.05) to stay tractable. We chose to tie the bandwidth h_t to the posterior precision trace rather than tune it as a free hyperparameter because a fixed bandwidth would decouple the transition from the belief and reproduce the very capacity-bounded objective we are replacing, accepting the cost that the posterior must be maintained as an explicit Gaussian rather than an implicit recurrent state. We chose a kNN kernel mutual-information estimator over the Deep InfoMax contrastive bound used in Contrastive World Models because the latter is computed against a frozen feature space whose value cannot move as the belief sharpens, accepting the cost that the kNN estimator is noisier in high dimensions and needs the low-rank posterior structure to keep d small. A domain expert would not describe KCLD as a variant of Dreamer's RSSM or of the CWM contrastive objective, because neither of those methods has a transition whose bandwidth is a function of posterior precision, and neither has an MI term that is recomputed against a growing nonparametric memory rather than a fixed feature extractor.

**Why it measures what we claim.** The bandwidth h_t^2 = max(c/trace(Lambda_t), h_min^2) measures belief sharpness because a Gaussian posterior's trace of precision is exactly the inverse of its mean predictive variance under the assumption that the latent posterior is well-calibrated Gaussian and unimodal; this assumption fails when the true posterior is multimodal, in which case h_t reflects the average spread of the mixture rather than the local spread around the dominant mode and the kernel over-smooths. The one-sided consistency loss ||max(0, Lambda_t^{-1} - Sigma_hat)||_F^2 measures the alignment between the transition's local predictive spread and the posterior's claimed uncertainty because the Nadaraya-Watson local residual variance is an unbiased estimate of predictive variance under the assumption that memory points are locally i.i.d. samples of the transition; this fails when memory is temporally correlated, in which case Sigma_hat underestimates true spread and the regularizer over-sharpens. The kNN MI estimate measures the information the predicted latent carries about the true next latent because kNN estimators converge to the true MI under the assumption that the joint density is smooth at the k-th neighbor scale; this fails when the latent manifold is thin, in which case the estimate reflects local density rather than mutual information. The posterior update with kernel prior mean measures belief refinement because Bayes' rule with a Gaussian prior is exact under the assumption that the kernel prediction is an unbiased prior mean; this fails under distribution shift where the kernel extrapolates by weighted averaging of distant memories, in which case the update reflects stale memory rather than current dynamics.

## Contribution

1. KCLD, a latent video world model whose transition operator is a nonparametric kernel-weighted estimator with bandwidth coupled to posterior precision, removing the fixed-capacity transition network used by DreamerV3-style RSSMs and Contrastive World Models. 2. A precision-matched consistency regularizer that forces the transition's local predictive spread to track the posterior's claimed uncertainty, giving a training signal that sharpens as the belief sharpens rather than saturating at a fixed feature space. 3. A kernel-based mutual-information estimator over a growing deduplicated memory that replaces the Deep InfoMax bound, providing an objective whose value moves with belief refinement rather than with backbone parameter count.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Distractor-heavy DMC + Atari-100k subset | Tests sharpening under correlated, noisy latents |
| Primary metric | kNN MI between predicted and true next latent (k=5, d=32) | Directly measures claimed information gain |
| Baseline | Dreamer (RSSM, pixel reconstruction) | Learned transition with implicit recurrent belief |
| Baseline | Contrastive World Model (Deep InfoMax) | Fixed-feature MI bound that cannot sharpen |
| Baseline | Momentum-prediction baseline | Non-adaptive predictive spread, no belief coupling |
| Baseline | Hypernetwork transition (precision-conditioned variance) | Learned transition with precision-conditioned bandwidth |
| Ablation-of-ours | KCLD with fixed bandwidth h (no precision tie) | Isolates the bandwidth-precision consistency claim |
| Ablation-of-ours | KCLD with hard L_cons equality (original) | Tests one-sided penalty vs. equality |

### Why this setup validates the claim

The central claim is that coupling transition bandwidth to posterior precision yields a belief that sharpens with memory, and that a capacity-free kernel MI term captures this sharpening where a fixed-feature contrastive bound cannot. The distractor-heavy dataset is the right testbed because distractors inflate posterior uncertainty in a controlled way, so the predicted sharpening effect has a concrete trigger. The primary metric, kNN MI between predicted and true next latent, is chosen because it is the exact quantity the method claims to maximize and the quantity the contrastive baseline bounds against a frozen space; if our mechanism works, this metric should rise as memory accumulates while the baseline's bound plateaus. Dreamer tests whether a learned recurrent transition can match precision-tied sharpening; the CWM baseline tests whether a fixed-feature MI bound is the actual bottleneck; the momentum baseline tests whether any adaptive spread is needed at all; the hypernetwork baseline tests whether the nonparametric memory adds value beyond precision-conditioned variance. The fixed-bandwidth ablation is the decisive control: it removes only the precision tie, so any gap between it and full KCLD isolates the specific causal mechanism rather than generic kernel capacity. The hard-equality ablation tests whether the one-sided penalty is necessary to avoid over-sharpening. Together these form a falsifiable test: if the gain is uniform across all baselines and the ablation matches full KCLD, the precision-bandwidth coupling is doing no work.

### Expected outcome and causal chain

**vs. Dreamer (RSSM)** — On a case where a distractor enters the frame and posterior uncertainty spikes, Dreamer's learned recurrent transition keeps a roughly fixed effective spread because its transition weights were fit under the training distribution and cannot re-sharpen when the belief is temporarily diffuse. Our method instead shrinks h_t as trace(Lambda_t) recovers, pulling the kernel estimate toward the nearest sharpened memories, so we expect a visible MI gap on distractor-heavy episodes but near-parity on clean episodes where both methods are already confident.

**vs. Contrastive World Model** — On a case where the latent belief sharpens over a long trajectory, the CWM contrastive bound is computed against a frozen feature space, so its MI estimate saturates at the value encoded in that fixed extractor and cannot register the sharpening. Our method recomputes kNN MI against a growing memory whose local density tracks the sharpened posterior, so we expect the MI curve to keep climbing with trajectory length while CWM's flattens, with the gap widening on the longest episodes.

**vs. Momentum-prediction baseline** — On a case where dynamics switch abruptly (action changes regime), the momentum baseline predicts by extrapolating recent latents and produces a stale, over-smoothed next latent because it has no mechanism to weight memories by action similarity. Our method's kernel weights include the action term ||a_t - a_i||^2, so it retrieves memories from the matching action regime, and we expect a concentrated error reduction on regime-switch segments with little difference on smooth segments.

**vs. Hypernetwork transition** — On a case where the posterior is sharp and well-calibrated, the hypernetwork baseline can condition its variance on trace(Lambda_t) but its variance is still produced by a fixed-capacity network, so it cannot exploit the growing memory to reduce error beyond its training distribution. Our method's error depends on memory density and bandwidth, so we expect KCLD to outperform the hypernetwork as memory size increases, with the gap largest on the longest episodes.

**vs. fixed-bandwidth ablation** — On a case where the posterior is well-calibrated and sharp, the fixed-bandwidth variant over-smooths because its h never shrinks, averaging in distant memories whose residuals inflate Sigma_hat and trigger the consistency loss to fight the posterior. Full KCLD shrinks h with trace(Lambda_t) down to the density floor h_min, so Sigma_hat tracks Lambda_t^{-1} and L_cons drops; we expect the ablation to show elevated consistency loss and flattened MI specifically on low-uncertainty segments, confirming the precision tie is the active ingredient.

**vs. hard-equality ablation** — On a case where the posterior is overconfident (miscalibrated), the hard-equality variant drives h_t to zero because L_cons forces Sigma_hat to equal Lambda_t^{-1}, collapsing the kernel to a nearest-neighbor copy and inflating prediction error. Full KCLD's one-sided penalty only penalizes over-confidence (Sigma_hat < Lambda_t^{-1}) and the density floor h_min prevents collapse, so we expect lower error on miscalibrated segments.

### What would falsify this idea

If the MI gain over the contrastive baseline is uniform across trajectory lengths rather than concentrated on long, sharpening trajectories, or if the fixed-bandwidth ablation matches full KCLD on low-uncertainty segments, then the precision-bandwidth coupling is not the causal mechanism and the central claim is wrong.

## References

1. Contrastive World Models
2. DreamGen: Unlocking Generalization in Robot Learning through Video World Models
3. HunyuanVideo: A Systematic Framework For Large Video Generative Models
4. Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos
5. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
6. Mastering diverse control tasks through world models
7. Context Autoencoder for Self-supervised Representation Learning
8. Masked Siamese Networks for Label-Efficient Learning

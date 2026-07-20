# Noise-Adversarial Training for Label-Robust Vision-Language-Action Models

## Motivation

Auto-labeling pipelines in vision-language-action models, as used in Xiaomi-Robotics-1, introduce language label noise that degrades policy quality, especially under varying noise levels. Existing methods assume scaling data volume mitigates noise, but structurally fail to guarantee invariance because they do not explicitly model noise robustness. This leaves a persistent gap where policy quality cannot be ensured across different auto-labeling conditions.

## Key Insight

Treating auto-labeling noise as an adversarial perturbation in the language embedding space forces the policy to rely on invariant task-relevant features from vision and action consistency, rather than surface label patterns.

## Method

### Noise-Adversarial Vision-Language-Action Training (NAVLA)

**(A) What it is:** NAVLA is a training procedure that adds an adversarial perturbation to the language embedding during training to make the policy robust to label noise. It inputs trajectories (images, actions, auto-generated language labels) and outputs a policy parameter update.

**(B) How it works:**
```python
# Hyperparameters: epsilon (noise budget, e.g., 0.1, set to 95th percentile of validation noise distances), alpha (step size, 0.01), K (iterations, 3), lambda (trade-off, 0.5)
# Language encoder: CLIP text transformer (ViT-B/32), frozen
# Policy: 2-layer transformer, 8 heads, hidden size 256
calibration_set = 512 trajectories with clean and noisy labels  # used to set epsilon
For each batch (I, A, L):
    # I: images, A: actions, L: language labels
    h = language_encoder(L)                  # language embedding
    # 1. Clean loss
    pred_clean = policy(I, h)
    L_clean = MSE(pred_clean, A)
    # 2. Adversarial perturbation on h
    delta = 0
    for k in range(K):
        pred_adv = policy(I, h + delta)
        L_adv = MSE(pred_adv, A)
        grad = gradient(L_adv, delta)
        delta = delta + alpha * sign(grad)   # PGD step
        delta = clip_by_norm(delta, epsilon) # L2 ball projection
    # 3. Adversarial loss
    pred_adv_final = policy(I, h + delta)
    L_adv_final = MSE(pred_adv_final, A)
    # 4. Combined loss
    total_loss = L_clean + lambda * L_adv_final
    update policy parameters via total_loss
# Post-training calibration: measure proportion of noisy embeddings within epsilon-ball
```

**(C) Why this design:** We chose adversarial training over data augmentation because adversarial perturbations directly optimize for worst-case noise, while augmentation only covers seen noise patterns, accepting the cost of increased training time (K extra forward/backward passes). We used a bounded L2 perturbation in the language embedding space rather than discrete label flipping because the continuous space allows gradient-based optimization, but at the cost of requiring a differentiable language encoder. We set the noise budget epsilon to the 95th percentile of L2 distances between clean and noisy embeddings on a validation set of 512 trajectories, rather than learning it, because a learned budget could collapse to zero; this choice trades adaptivity for stability. The trade-off hyperparameter lambda balances clean and adversarial losses: a high lambda risks overfitting to adversarial examples, while a low lambda provides insufficient robustness. This differs from standard adversarial training (Madry et al.) which perturbs input images; our perturbation targets language embeddings to specifically model label noise, not visual perturbations.

**(D) Why it measures what we claim:** The adversarial perturbation delta in language embedding space measures the worst-case label noise that maximally degrades policy accuracy, under the assumption that the true label noise distribution is contained within an L2 ball of radius epsilon centered at the clean embedding; this assumption fails when the noise is structured or correlated with visual features, in which case the adversarial training may target unrealistic perturbations. To verify this assumption, after training we compute the proportion of held-out noisy examples whose L2 distance from their clean embedding falls within epsilon; if this proportion is high (e.g., >95%), the assumption is empirically supported. The final adversarial loss L_adv_final measures the policy's robustness to such worst-case noise, assuming that minimizing this loss enforces invariance to label perturbations; this assumption fails if the policy can memorize both clean and adversarial examples without learning invariant features, in which case the loss would be low but the policy brittle to unseen noise patterns. The clean loss L_clean ensures the policy remains accurate on the original labels, preventing overfitting to adversarial examples.

## Contribution

(1) A noise-adversarial training framework (NAVLA) that explicitly models auto-labeling noise as adversarial perturbations in language embedding space for vision-language-action policies. (2) A design principle that policy robustness to label noise can be achieved without additional manual data, by leveraging the adversarial training paradigm with a bounded perturbation constraint. (3) Empirical demonstration that the method maintains policy quality invariance across varying noise levels, validated on the Xiaomi-Robotics-1 dataset (hypothetical).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | BridgeData v2 (noisy subsets) | Diverse tasks with auto-generated labels and real noise |
| Primary metric | Success rate | Directly measures task completion |
| Secondary metric | L2 ball coverage (% noisy embeddings within epsilon) | Validates assumption about noise distribution |
| Baseline | Standard VLA (no adversarial) | Baseline for effect of adversarial training |
| Baseline | VLA with label smoothing (α=0.1) | Alternative robustness approach |
| Baseline | VLA with random perturbation (ε=0.1, uniform) | Random perturbation baseline |
| Baseline | VLA with action-space adversarial training (ε=0.1, K=3) | Compare perturbation space (language vs. action) |
| Ablation-of-ours | NAVLA without adversarial (λ=0) | Isolates adversarial component |

### Why this setup validates the claim
BridgeData v2 provides diverse manipulation tasks with automatically generated language labels that inherently contain labeling noise, allowing evaluation on realistic noise levels. The primary metric (success rate) directly measures task performance under noise. Standard VLA isolates the effect of adversarial training, while label smoothing and random perturbation represent alternative robustness methods. The action-space adversarial training baseline tests whether perturbing language embeddings is uniquely beneficial. The L2 ball coverage metric empirically checks the core assumption that real noise lies within the epsilon-ball. The ablation quantifies the adversarial component’s contribution. This combination creates a falsifiable test: if NAVLA outperforms all baselines specifically on noisy subsets and the coverage metric exceeds 95%, the claim holds; if not, the method is not providing targeted robustness.

### Expected outcome and causal chain

**vs. Standard VLA** — On a case where the language label is incorrect (e.g., "lift mug" instead of "push mug"), the standard policy will attempt the wrong action because it overfits to the noisy label. Our method instead computes an adversarial perturbation that simulates worst-case label noise, training the policy to be invariant to such errors. Thus we expect a noticeable gap on subsets with high labeling error (e.g., auto-generated captions) but similar performance on clean, human-labeled subsets.

**vs. VLA with label smoothing** — Label smoothing reduces overconfidence but does not explicitly handle worst-case noise. On a case with severe label noise (e.g., multiple conflicting labels), label smoothing leads to degraded performance because it only reduces confidence without enforcing invariance. Our method directly optimizes for worst-case perturbations, so we expect it to significantly outperform on high-noise subsets while performing similarly on low-noise ones.

**vs. VLA with random perturbation** — Data augmentation with random perturbations covers only seen noise patterns. On a case where the noise is systematic (e.g., a consistent bias in captioning), augmentation fails to prepare the policy. Our method models worst-case perturbations, so we expect better generalization to unknown noise distributions. The gain should be concentrated on subsets where augmentation does not match the true noise distribution.

**vs. VLA with action-space adversarial training** — Perturbing actions directly targets action noise, not language label noise. On a case where language labels are noisy but actions are clean, action-space adversarial training provides no benefit. Our method specifically targets language noise, so we expect it to outperform on subsets with high language label noise, while both methods perform similarly on clean-language subsets.

**vs. NAVLA without adversarial (ablation)** — Removing the adversarial loss (λ=0) reduces to standard VLA. The comparison shows the performance gain directly attributable to adversarial training. We expect the full method to outperform the ablation on noisy subsets.

### What would falsify this idea
If the improvement from NAVLA is uniform across both clean and noisy subsets, rather than concentrated on high-noise subsets, then the central claim that adversarial training provides targeted robustness to label noise is false. Additionally, if the L2 ball coverage metric is low (<50%), the assumption that real noise lies in an L2 ball is violated, undermining the method's foundation.

## References

1. Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories

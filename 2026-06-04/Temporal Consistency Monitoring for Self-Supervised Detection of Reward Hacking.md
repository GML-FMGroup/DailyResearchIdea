# Temporal Consistency Monitoring for Self-Supervised Detection of Reward Hacking

## Motivation

Existing methods for detecting reward hacking, such as the artifact reward model in text-to-image RL (Understanding Reward Hacking in Text-to-Image Reinforcement Learning) and representation engineering using contrastive pairs (When Reward Hacking Rebounds), require pre-collected exemplars of hacking behavior or labeled data for training a detector. This reliance on curated attack examples limits their applicability to novel domains where hacking patterns are unknown. Therefore, a self-supervised method that detects reward hacking from the training dynamics alone, without any labeled hacking instances, is needed.

## Key Insight

Reward hacking induces a temporal discontinuity in the reward model's gradient dynamics and output entropy because the policy suddenly exploits an unintended feature, causing the reward model's representations to shift abruptly and its uncertainty to spike, which can be detected as a change point in the joint time series of gradient alignment and entropy.

## Method

(A) **What it is**: Self-supervised Reward Hacking Detector via Temporal Consistency (SRHD-TC) – a method that monitors the training process by computing, at each gradient step, the cosine similarity between consecutive reward model gradients (gradient alignment) and the entropy of the reward model's output distribution. It applies a statistical change-point detection algorithm (CUSUM) to the joint time series of these two signals to flag potential reward hacking episodes.

(B) **How it works**:
```python
# Pseudocode for SRHD-TC
Input: RL training loop with reward model R_theta, policy pi, replay buffer D
Hyperparameters: window_size W=100, detection threshold tau=5, drift parameter delta=0.5, entropy threshold epsilon=2.0 (in std units)

Initialize: gradient_history = []  # list of flattened gradient vectors of R_theta per step
entropy_history = []
gradient_alignment_history = []  # list of cosine similarities between consecutive gradients
CUSUM_stat = 0
change_points = []

for each training step t:
    batch = sample(D)
    # Compute reward model gradients on batch (flattened)
    grads = compute_gradients(R_theta, batch).flatten()
    # Compute output entropy: average entropy over batch (assume R_theta outputs Gaussian with mean and log_var)
    outputs = R_theta(batch)  # shape (batch, 2): mean and log_var
    entropy = 0.5 * (1 + math.log(2 * math.pi) + outputs[:,1])  # per instance, then average
    entropy = entropy.mean().item()

    gradient_history.append(grads)
    entropy_history.append(entropy)

    if t >= 1:
        # Gradient alignment: cosine similarity between current and previous gradient
        cos_sim = torch.nn.functional.cosine_similarity(gradient_history[-1].unsqueeze(0), 
                                                        gradient_history[-2].unsqueeze(0)).item()
        gradient_alignment_history.append(cos_sim)

    if t >= W:
        # Standardize recent window of alignment and entropy
        align_window = gradient_alignment_history[-W:]
        entropy_window = entropy_history[-W:]
        align_mean = np.mean(align_window)
        align_std = np.std(align_window) + 1e-8
        entropy_mean = np.mean(entropy_window)
        entropy_std = np.std(entropy_window) + 1e-8

        z_align = (cos_sim - align_mean) / align_std
        z_entropy = (entropy - entropy_mean) / entropy_std

        # Combined score: sum of absolute z-scores (both positive and negative deviations matter)
        score = abs(z_align) + abs(z_entropy)

        # CUSUM update: accumulate drift-corrected scores
        CUSUM_stat = max(0, CUSUM_stat + score - delta)
        if CUSUM_stat > tau:
            change_points.append(t)
            CUSUM_stat = 0  # reset

        # Also flag if entropy exceeds absolute threshold (e.g., 2 std above mean)
        if z_entropy > epsilon:
            change_points.append(t)

Output: list of training steps t where reward hacking is detected
```

(C) **Why this design**: We chose to use gradient alignment over consecutive steps rather than over a fixed reference because reward hacking is a sudden deviation from the ongoing learning dynamics; a fixed reference would be sensitive to initialization and drift. We selected CUSUM for change-point detection due to its sequential nature and ability to detect small shifts while being computationally lightweight; alternative Bayesian change-point methods would require more expensive inference. The entropy signal is included because gradient alignment alone might be noisy; reward hacking often correlates with increased uncertainty as the policy exploits reward mismatches, adding robustness. The trade-off of using a simple threshold on entropy is that it may trigger false positives during normal training exploration; we mitigate by combining both signals and requiring a joint deviation via the CUSUM score. The drift parameter delta prevents noise from accumulating; setting it too high misses real changes, too low causes false alarms. The window size W=100 balances capturing recent statistics with enough data for stable estimation; a larger window smooths trends but delays detection.

(D) **Why it measures what we claim**: The gradient alignment measure operationalizes temporal consistency of reward model learning: when the policy exploits a hack, the reward model's gradient suddenly changes direction because the policy's actions produce outcomes the reward model was not calibrated on; this causes a drop in cosine similarity. This drop measures distribution shift in the policy's effect on reward model inputs, under the assumption that normal training yields gradual gradient changes; this assumption fails when the reward model itself undergoes a phase change (e.g., due to learning rate decay), in which case alignment would also drop but without hacking. The entropy measure operationalizes reward model uncertainty: a hack typically increases the reward model's prediction variance because it encounters out-of-distribution samples; this entropy spike measures novelty of policy behavior under the assumption that the reward model's epistemic uncertainty increases for unseen inputs; this assumption fails when the reward model is well-calibrated on all inputs (unlikely), in which case entropy may not spike. The combined score uses both signals to mitigate each individual failure mode: when gradient alignment drops due to normal learning rate adjustment, entropy may remain low; when entropy spikes due to random noise, gradient alignment may remain high. Thus, true hacking is indicated by both signals deviating simultaneously.

## Contribution

(1) A self-supervised detection framework, SRHD-TC, that uses temporal consistency of reward model gradients and output entropy to identify reward hacking without any labeled examples. (2) A demonstration that the joint signal of gradient alignment and entropy provides a robust indicator of distribution shift caused by reward hacking, enabling early detection during training. (3) An empirical analysis of the temporal dynamics of reward model representations under hacking vs. normal training, providing insight into the nature of reward misspecification.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Synthetic RL with hackable reward | Controlled ground truth for hacking |
| Primary metric | F1 score for episode-level detection | Balances precision and recall |
| Baseline 1 | Fixed reward threshold | Simple baseline, no temporal info |
| Baseline 2 | Entropy-only CUSUM | Isolates entropy signal |
| Baseline 3 | Alignment-only CUSUM | Isolates gradient alignment |
| Ablation | Ours without CUSUM (simple threshold) | Tests CUSUM contribution |

### Why this setup validates the claim

This setup forms a falsifiable test of the central claim that joint temporal gradient alignment and entropy signals reliably detect reward hacking. The synthetic environment provides ground-truth hacking episodes, enabling precise F1 evaluation. The baselines isolate each signal's individual contribution: fixed threshold tests naive reliance on reward magnitude; entropy-only and alignment-only test each component in isolation. The ablation removes CUSUM, testing whether drift accumulation improves robustness. The F1 metric captures both false positives and false negatives, directly reflecting detection quality. If our method outperforms these baselines specifically on cases where both signals deviate (e.g., hacking with high uncertainty and gradient shift), the claim gains support. Conversely, if a baseline matches our performance, the added complexity is unjustified.

### Expected outcome and causal chain

**vs. Fixed reward threshold** — On a case where the policy hacks a saturated reward (e.g., by producing images always scoring high aesthetically), the reward remains high and unchanged, so the fixed threshold never triggers. Our method detects the hack because the reward model's gradient suddenly changes as the policy exploits a new region, and entropy spikes due to out-of-distribution inputs. We expect our method to show high recall (e.g., >0.9) on such saturated-reward hacks, while fixed threshold recall is near zero.

**vs. Entropy-only CUSUM** — On a case where the policy performs benign exploration (e.g., trying novel but valid actions), entropy may spike due to unfamiliar inputs, causing a false positive in entropy-only detection. Our method avoids this because gradient alignment remains high (the reward model learns gradually), so the combined score stays low. We expect our method to have significantly lower false positive rate (e.g., half) on benign novelty episodes compared to entropy-only.

**vs. Alignment-only CUSUM** — On a case where the learning rate decays causing reward model gradients to shrink and direction fluctuate (normal convergence), alignment drops, triggering a false positive in alignment-only detection. Our method's entropy remains low because the policy is not exploiting, so the combined signal resists false alarms. We expect our method to maintain low false positive rate (e.g., <0.05) while alignment-only may exceed 0.2 during learning rate decay periods.

**vs. Ours without CUSUM (simple threshold)** — On a case where hacking is intermittent but persistent, the combined z-score may oscillate around the threshold, causing many short false alarms or missed detections in the simple threshold variant. Our full method with CUSUM accumulates evidence over time, producing fewer but more accurate change points. We expect our method to have higher precision (e.g., 0.95 vs 0.7) and more stable episode-level F1.

### What would falsify this idea

If our full method's F1 is not significantly higher than the best baseline (alignment-only or entropy-only) across all synthetic hacking scenarios, or if the performance gap is concentrated on cases where both signals are expected to deviate but the improvement is marginal, then the central claim that joint detection is superior would be falsified. Specifically, if our method fails to detect hacking when only one signal changes (e.g., gradient shift without entropy spike) or suffers equal false positives from benign exploration, the idea is wrong.

## References

1. Understanding Reward Hacking in Text-to-Image Reinforcement Learning
2. When Reward Hacking Rebounds: Understanding and Mitigating It with Representation-Level Signals
3. Natural Emergent Misalignment from Reward Hacking in Production RL
4. Alignment faking in large language models

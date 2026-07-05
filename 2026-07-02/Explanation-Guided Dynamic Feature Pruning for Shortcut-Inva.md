# Explanation-Guided Dynamic Feature Pruning for Shortcut-Invariant Lightweight IIoT Intrusion Detection

## Motivation

Existing lightweight intrusion detection models for IIoT, as shown in "Cross-Domain Generalization Failure in Lightweight Intrusion Detection Models for IIoT Networks", rely on coarse port-category shortcuts that fail to generalize across datasets. This shortcut reliance arises because standard training procedures allow the model to latch onto spurious correlations (e.g., common benign port patterns) that are dataset-specific, while ignoring robust traffic behaviors. The structural root cause is that loss minimization does not penalize reliance on features with high predictive power only within the training distribution. We need a training procedure that actively enforces shortcut invariance without adding inference-time overhead, which is critical for resource-constrained IIoT devices.

## Key Insight

Pruning neurons with high attribution to shortcut features during training forces the model to learn alternative, generalizable representations because the attribution signal directly exposes features that have no causal relationship with the intrusion label across domains.

## Method

(A) **What it is**: Explanation-Guided Dynamic Feature Pruning (EGDFP) is a training procedure that takes a lightweight intrusion detection model and a source dataset, and outputs a shortcut-invariant model. It uses an attribution method (e.g., Integrated Gradients) to identify features (or neurons) that are dominantly used as shortcuts, and then dynamically prunes (zeroes out) those features during training. Input: model, source data, attribution threshold τ, pruning rate ρ. Output: trained model with fewer shortcut-dependent pathways. (B) **How it works**:
```pseudocode
Initialize model with weights w
For each training epoch:
  1. Train on source dataset for N steps with standard cross-entropy loss
  2. Every K steps, compute attributions A(x) = IntegratedGradients(model, x) for a batch of source samples
  3. For each input feature dimension f, compute mean absolute attribution over batch: avg_attr[f]
  4. Rank features by avg_attr descending
  5. Define shortcut set S = top ρ fraction of features with highest avg_attr
  6. Apply pruning mask: for each sample, zero out the input features in S (or for neuron-level: gate those neurons to 0)
  7. Continue training with masked inputs
After training, remove pruning (use original model) or fine-tune without mask.
```
Hyperparameters: ρ = 0.1, K = 10 batches, attribution sample size = 128. (C) **Why this design**: We chose dynamic pruning during training rather than static feature selection (e.g., filter methods) because static selection cannot adapt as the model's reliance evolves; the trade-off is increased computation for periodic attribution computation. We used Integrated Gradients over simpler gradient×input because IG satisfies completeness and sensitivity, providing more reliable attributions at the cost of higher compute per sample. Pruning input features directly (rather than adding a regularization penalty) forces the model to never see the shortcut value during training, which is a stronger intervention than a penalty that weights could still bypass; the cost is potential loss of information if the attribution mistakenly identifies a causal feature as shortcut. We perform pruning every K steps rather than every step to amortize attribution cost while still enabling adaptation; the risk is that the model may re-learn shortcuts between pruning intervals. (D) **Why it measures what we claim**: The mean absolute attribution on source data <avg_attr[f]> measures the model's reliance on feature f in making predictions, because Integrated Gradients by construction satisfies completeness: attributions sum to the logit difference from baseline. This relies on the assumption that the baseline (zero) is an appropriate reference; this assumption fails when the model's behavior at zero input is not representative (e.g., if zero is far from natural data), in which case attributions may over- or under- emphasize features. The pruning mask operationally enforces shortcut invariance by removing the features with highest attribution, forcing the model to rely on remaining features; this assumes that shortcut features are among the highest-attribution ones. This assumption fails if a shortcut feature has low attribution because the model uses it in a nonlinear way (e.g., interactions), in which case pruning will miss the shortcut and the model may still generalize poorly. The periodic repetition (every K steps) ensures that as the model's reliance shifts, newly emerged shortcuts are also pruned; this assumes that shortcut features remain consistently high-attribution across training phases, which could fail if the model discovers a new shortcut that initially has low attribution but becomes dominant later.

## Contribution

(1) A novel training procedure (EGDFP) that uses explanation-guided dynamic feature pruning to enforce shortcut invariance in lightweight IIoT intrusion detection models. (2) An empirical demonstration that models trained with EGDFP achieve significantly higher cross-dataset generalization (e.g., on the Gotham 2025 dataset) compared to standard training and static feature selection baselines, without increasing inference complexity. (3) A diagnostic analysis identifying which features (e.g., port-category, protocol type) typical lightweight models rely on as shortcuts, confirming and extending findings from prior work.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CICIDS2017 (source), CSE-CIC-IDS2018 (target) | Cross-dataset, realistic IIoT traffic scenarios |
| Primary metric | Target-domain F1-score | Measures generalization, balanced class performance |
| Baseline 1 | Standard training (no pruning) | Baseline without any shortcut mitigation |
| Baseline 2 | Static feature pruning (prune once) | Tests if dynamic adaptation is needed |
| Ablation-of-ours | EGDFP with gradient × input | Evaluates importance of Integrated Gradients |

### Why this setup validates the claim
This experimental design directly tests the central claim that dynamic, attribution-guided feature pruning during training improves cross-dataset generalization by removing shortcuts. The cross-dataset pair (CICIDS2017→CSE-CIC-IDS2018) forces models to rely on generalizable patterns rather than dataset-specific correlations. Using target-domain F1-score as the primary metric captures both precision and recall, essential for imbalanced intrusion detection. Standard training (Baseline 1) establishes the baseline shortcut reliance; static pruning (Baseline 2) isolates the benefit of adaptivity; the gradient×input ablation (Ablation) isolates the need for reliable attributions (IG's completeness). If our method outperforms all baselines on target-domain F1, and the improvement is concentrated on subsets where shortcuts are known to differ (e.g., port-based attacks), then the claim is supported. If not, the idea is falsified.

### Expected outcome and causal chain

**vs. Standard training** — On a case where source dataset has a strong correlation between TCP port 80 and benign traffic, but target dataset uses port 80 for both benign and attack, standard training learns to rely on port 80 as a shortcut, causing low recall on port-80 attacks in target. Our method dynamically prunes port 80 feature (high attribution) during source training, forcing the model to use other patterns, so we expect a noticeable F1 improvement on port-related attacks (e.g., +15%) while parity on others.

**vs. Static feature pruning** — On a case where shortcuts are not present early in training but emerge as the model learns (e.g., a feature combination becomes predictive in source but not target), static pruning removes only initially high-attribution features, missing later shortcuts. Our method re-evaluates attributions every K steps, detecting and pruning newly emerging shortcuts, so we expect higher target F1 on late-emerging shortcut subsets (e.g., +10% on composite features) while matching static on others.

**vs. Gradient×input ablation** — On a case where a shortcut feature is nonlinearly used (e.g., squared value of packet length), gradient×input fails to assign high attribution (saturation), missing the shortcut. Our method uses Integrated Gradients which satisfies completeness, correctly identifying the shortcut even with nonlinearities, so we expect our method to show higher F1 on such nonlinearly-used shortcut subsets (e.g., +8%) compared to the ablation.

### What would falsify this idea
If our method's target-domain F1 improvement over static pruning is negligible (≤1%) or uniform across all feature subsets (not concentrated on predicted shortcut features), then the claim that dynamic adaptation is needed would be false. Similarly, if the gradient×input ablation performs equally well, then IG's completeness is not critical.

## References

1. Cross-Domain Generalization Failure in Lightweight Intrusion Detection Models for IIoT Networks
2. Gotham Dataset 2025: A Reproducible Large-Scale IoT Network Dataset for Intrusion Detection and Security Research
3. Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study
4. Machine Learning on Public Intrusion Datasets: Academic Hype or Concrete Advances in NIDS?
5. Evaluation of Inter-Dataset Generalisability of Autoencoders for Network Intrusion Detection

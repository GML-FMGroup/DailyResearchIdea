# AdaSoft: Difficulty-Adaptive Label Softening for Mixup Augmentation in Biometric Recognition

## Motivation

Existing soft augmentation methods (e.g., Soft Augmentation) and mixing-based augmentations (e.g., StarMixup) apply a fixed or augmentation-magnitude-based label softening, ignoring the inherent difficulty of each training sample. In biometric recognition, sample quality varies widely due to occlusion, noise, or pose variation, and treating all samples uniformly leads to poor calibration—a problem highlighted by AGVBench where high-accuracy mixing methods like PuzzleMix suffer from high expected calibration error. The root cause is that the softening parameter is not informed by sample complexity, causing overconfidence on hard samples and underconfidence on easy ones.

## Key Insight

Sample difficulty, measured by the model's prediction entropy from a single forward pass, provides a computationally lightweight and theoretically grounded signal for adjusting label softening because entropy directly quantifies the model's uncertainty, which is the quantity label softening aims to regulate.

## Method

[Current method]
**(A) What it is:** AdaSoft is a data augmentation module that per-sample adapts the label softening magnitude in MixUp-based augmentations using prediction entropy as a difficulty measure. Its inputs are a batch of samples with labels, and outputs are the augmented samples with per-sample softened labels.  

**(B) How it works (pseudocode):**  
```python
for batch (X, y) in dataloader:
    # Step 1: Compute prediction entropy for each sample in the batch
    logits = model(X)  # forward pass without gradient
    probs = softmax(logits)
    H = -sum(probs * log(probs + 1e-8), dim=1)  # entropy per sample
    
    # Step 2: Normalize H to [0,1] using running mean/std (momentum=0.9)
    H_norm = (H - running_mean) / (running_std + 1e-8)
    H_clipped = clip(H_norm, -3, 3)  # outlier robust
    
    # Step 3: Compute per-sample softening coefficient α
    β = 2.0  # scaling hyperparameter
    τ = 0.0  # threshold (in normalized space)
    α = sigmoid(β * (H_clipped - τ))  # α ∈ (0,1)
    
    # Step 4: Apply MixUp on a pair (i, j) (for simplicity, uniform random pair)
    λ = random.beta(α, α)  # mixing coefficient sampled from Beta(α, α) — note: α varies per sample
    x_tilde = λ * X[i] + (1-λ) * X[j]
    y_tilde = λ * y[i] + (1-λ) * y[j]
    
    # Step 5: Apply entropy-adaptive label softening
    α_i = α[i]  # softening for sample i (the primary sample in the pair)
    # Alternatively, combine both α[i] and α[j]; we simplify to use α of the first sample
    y_soft = (1 - α_i) * y_tilde + α_i * uniform_vector  # uniform_vector = ones(C)/C
    
    # Use x_tilde, y_soft for training loss (e.g., cross-entropy)
    loss = cross_entropy(model(x_tilde), y_soft)
    # Backprop and update model, also update running mean/std for H
```

**(C) Why this design:**  
We chose prediction entropy over gradient magnitude (a common difficulty measure) because entropy requires only one forward pass without backprop, making it computationally cheaper and avoiding interference with the main gradient computation—accepting the cost that entropy may be less sensitive to target-specific difficulty (e.g., adversarial examples). We use a sigmoid mapping with a threshold (β=2, τ=0) instead of a linear mapping because it provides a smooth transition between low and high softening regimes, avoiding abrupt changes that could destabilize training. The decision to apply softening only to the primary sample in the pairing (rather than a symmetric scheme) reduces the number of entropy computations per pair (only one forward pass) at the risk of missing difficulty information from the secondary sample; we found simplifying trade-off acceptable for efficiency. We update running statistics for entropy normalization using exponential moving average (momentum=0.9), which adapts to distribution shifts over training epochs without requiring a separate validation set.

**(D) Why it measures what we claim:**  
We assume prediction entropy is a proxy for sample difficulty under the condition that the model's softmax probabilities are well-calibrated; if calibration degrades, entropy may not track true difficulty, and we mitigate this with running normalization but do not fully resolve it. The entropy H measures sample difficulty because, under the ideal case where the model's softmax probabilities reflect true class probabilities (i.e., model is well-calibrated), higher entropy indicates greater uncertainty and thus a harder sample; this assumption fails when the model is miscalibrated (e.g., early in training), in which case entropy may measure overconfidence rather than true difficulty. However, by using a running normalization that adapts to the current model state, we mitigate calibration drift. The sigmoid mapping α = σ(β*(H_ norm - τ)) translates normalized entropy into a softening coefficient that increases with difficulty (higher H → higher α), operationalizing the motivation that harder samples need stronger label smoothing. The uniform_vector target for high entropy samples prevents the model from committing to overly confident predictions on uncertain inputs, directly addressing the calibration failure reported in AGVBench. The Beta(α, α) sampling for λ ties the mixing strength to sample difficulty, as α appears in both the label mixing and softening—a deliberate coupling that ensures augmentation magnitude and label reliability are consistent: hard samples produce more uniform mixing, reducing the influence of uncertain labels. To further mitigate the impact of miscalibration, we note that applying temperature scaling to logits before entropy computation could enhance robustness, but we leave this extension for future work.

## Contribution

(1) AdaSoft, a novel difficulty-adaptive label softening mechanism that adjusts the softening magnitude per training sample based on prediction entropy, designed for MixUp-based augmentations in biometric recognition. (2) The design principle that sample difficulty derived from the model itself can serve as a computationally efficient calibrator for label smoothing, improving calibration without sacrificing accuracy. (3) An open-source implementation and evaluation protocol on two palm-vein datasets (from the AGVBench suite) to facilitate reproducibility.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | FV-USM vein dataset | High overfitting risk in vein recognition |
| Primary metric | Expected Calibration Error (ECE) | Directly measures confidence reliability |
| Baseline 1 | Standard Augmentation (no MixUp) | Isolates benefit of MixUp framework |
| Baseline 2 | Standard MixUp (fixed α=1.0) | Tests need for adaptive softening |
| Baseline 3 | Label Smoothing (fixed ε=0.1) | Compares to fixed uniform softening |
| Ablation | AdaSoft w/o normalization | Tests necessity of running statistics |

### Why this setup validates the claim

The FV-USM vein dataset exhibits high intra-class variability and limited sample size, making overconfident predictions common. ECE captures miscalibration directly. Standard Augmentation provides a pure baseline without mixing; Standard MixUp tests the benefit of mixing alone; Label Smoothing tests fixed softening; Ablation tests the entropy normalization component. This combination disentangles the contributions of mixing, adaptive softening, and normalization, forming a falsifiable test: if our method fails to reduce ECE compared to fixed MixUp or smoothing, the central claim that entropy-adaptive softening improves calibration is wrong.

### Expected outcome and causal chain

**vs. Standard Augmentation (no MixUp)** — On a difficult vein sample with poor illumination, standard training yields overconfident wrong predictions because it sees limited variations and enforces hard labels. Our method applies MixUp with softened labels, reducing overconfidence by interpolating with other samples and smoothing targets. We expect AdaSoft to show lower ECE (e.g., 0.03 vs. 0.08) on the hardest quartile of samples.

**vs. Standard MixUp (fixed α=1.0)** — On an easy sample (clear vein pattern), fixed MixUp still applies strong interpolation (λ~0.5), diluting the correct label and hurting accuracy. Our method adapts α to be low for easy samples (low entropy), preserving the original label. We expect AdaSoft to achieve higher clean accuracy (e.g., 95% vs. 92%) while maintaining similar ECE.

**vs. Label Smoothing (fixed ε=0.1)** — On a sample of moderate difficulty, fixed smoothing applies uniform softening regardless of actual uncertainty, potentially over-smoothing confident predictions. Our method uses entropy to adjust softening dynamically, softening only where needed. We expect AdaSoft to have lower ECE on medium-difficulty samples (e.g., 0.04 vs. 0.06).

**vs. AdaSoft w/o normalization** — Without normalization, entropy values can be scale-shifted across training epochs, causing inconsistent softening. With running normalization, adaptivity remains stable. We expect the normalized version to converge faster (e.g., 10 epochs earlier) and achieve lower final ECE (e.g., 0.025 vs. 0.04).

### What would falsify this idea

If AdaSoft's ECE is not consistently lower than Standard MixUp across all difficulty quartiles, or if the ablation without normalization performs equally well, then the central claim that entropy-adaptive softening combined with normalization improves calibration is invalidated.

## References

1. AGVBench: A Reliability-Oriented Benchmark of Data Augmentation for Vein Recognition
2. StarMixup: A More Suitable Mixup Method for Palm-Vein Identification

# RobustNVS: Domain-Invariant Feature Extraction for Novel View Synthesis under Extreme Imaging Conditions

## Motivation

Existing novel view synthesis methods like MetaView rely on a geometry perception network that assumes standard imaging conditions; under low-light or motion blur, monocular depth estimates become unreliable, leading to inconsistent geometry and rendering artifacts. Depth Anything 3, trained on clean academic datasets, similarly lacks robustness. The root cause is that feature extraction frontends do not explicitly account for domain shifts caused by real-world degradation, allowing degradation-specific patterns to propagate into the geometry prior.

## Key Insight

Adaptive instance normalization can align feature distributions across degradation conditions by matching mean and variance, effectively decoupling content from nuisance variations, provided the degradation simulator covers the target domain.

## Method

### (A) What it is
RobustNVS augments the MetaView feature extraction frontend with an adaptive instance normalization (AdaIN) layer and a learnable degradation simulator. Input: a single image; output: domain-invariant feature maps fed to MetaView's geometry network and diffusion model.

### (B) How it works (training loop)
```python
for batch in data_loader:
    x = batch['clean_image']
    # Learnable degradation simulator: MLP takes noise vector z ~ N(0,I) and outputs degradation parameters (gamma, kernel_sigma)
    gamma, sigma = degradation_simulator(z)  # gamma in [0.1, 0.5], sigma in [0, 5]
    # Apply low-light and motion blur differentiable operations
    degraded = apply_lowlight(x, gamma)  # pixel-wise scaling
    kernel = gaussian_kernel(sigma) if sigma>0 else identity
    degraded = apply_blur(degraded, kernel)
    # Feature extraction with AdaIN
    f = feature_extractor(degraded)
    # Compute target statistics from a reference batch (e.g., clean images)
    mu_ref, sigma_ref = compute_running_stats(reference_batch)
    # AdaIN: normalize f to have reference statistics
    f_norm = AdaIN(f, mu_ref, sigma_ref)
    # Feed f_norm to MetaView's geometry network and diffusion
    loss = MetaView_train_step(f_norm, camera_params, target_view)
    # Update feature extractor, AdaIN parameters, and degradation simulator
    loss.backward()
    optimizer.step()
    update_reference_stats()
```
Hyperparameters: AdaIN uses exponential moving average for reference stats (momentum=0.99). Degradation simulator is a 2-layer MLP with hidden size 128.

### (C) Why this design
We chose AdaIN over simple batch normalization (BN) because BN normalizes across a batch, which mixes different degradation types and fails to produce a consistent target distribution; AdaIN allows per-image statistics to be aligned to a clean reference, providing explicit domain invariance at the cost of requiring a reference distribution that may not be representative of all clean images. We chose a learnable degradation simulator over a fixed random sampler because the simulator can adapt to produce challenging samples that the feature extractor finds hardest, improving robustness more efficiently; however, this introduces an adversarial training dynamic that may destabilize early training. We chose to apply degradation only during training, not inference, because at test time we assume the input can be any condition and the AdaIN layer should handle it by matching to the reference statistics learned from clean images; but if test degradation is outside the simulator's distribution, invariance may fail. Additionally, we embed AdaIN after the first few convolutional layers rather than later in the network, as lower-level features are more affected by degradation, and normalizing early prevents propagation of domain-specific patterns; this sacrifices some high-level semantic invariance but simplifies optimization.

### (D) Why it measures what we claim
**Load-bearing assumption:** Degradation effects are captured primarily by the first and second moments of feature distributions (mean and variance), so matching these via AdaIN suffices for domain invariance.

**Calibration/verification:** To validate this assumption, we compute the skewness and kurtosis of feature maps from a calibration set of 512 clean and degraded images. If the skewness and kurtosis differ significantly (e.g., >0.5 difference), we flag that higher-order statistics are not aligned and report the discrepancy as a limitation.

The AdaIN operation (feature statistics matching) measures domain invariance because we assume that degradation effects are captured primarily by the first and second moments of feature distributions (mean and variance); this assumption fails when degradation induces non-linear distortions (e.g., severe clipping due to low-light) that alter higher-order cumulants (skewness, kurtosis), in which case the metric reflects only partial invariance and residual degradation-specific patterns remain. The learnable degradation simulator measures coverage of the target domain because it is trained to produce features that the AdaIN layer cannot fully normalize; this assumption fails when the simulator overfits to a narrow set of degradation types and does not explore the full spectrum of real-world conditions, in which case the metric reflects invariance only to simulated degradations and not to unseen ones. The reference statistics computed from a clean batch measure the desired clean-domain distribution because we assume that clean images share a common feature distribution; this assumption fails when clean images vary widely in content (e.g., indoor vs. outdoor scenes have different feature statistics), in which case the reference statistics become a compromise that may not align any specific image well, causing content distortion.

## Contribution

(1) A robust feature extraction frontend for novel view synthesis that integrates adaptive instance normalization with a learnable degradation simulator, enabling domain-invariant feature extraction under low-light and motion blur conditions. (2) Empirical demonstration that invariant features significantly improve geometry consistency and rendering quality compared to standard MetaView when tested on degraded inputs, with analysis of the limitations of moment-based invariance. (3) A training framework that adversarially simulates degradations to expose the feature extractor to challenging conditions, improving robustness without requiring paired clean-degraded data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RealEstate10K | Large viewpoint changes, realistic scenes. |
| Primary metric | PSNR | Measures pixel-wise reconstruction fidelity. |
| Baseline 1 | MetaView | Direct base method without our additions. |
| Baseline 2 | VGGT | Prior SOTA with implicit geometry. |
| Ablation 1 | RobustNVS w/o AdaIN | Tests necessity of adaptive normalization. |
| Ablation 2 | RobustNVS w/o deg. sim. | Tests necessity of learnable degradation simulator. |
| Resources | 200 GPU hours on NVIDIA A100 (40GB) | Ensures reproducibility and feasibility. |

### Why this setup validates the claim

This combination tests the central claim that domain-invariant features via AdaIN and a learnable degradation simulator improve robustness to image degradation in novel view synthesis. By comparing against MetaView, we isolate the effect of our additions, while VGGT provides a strong baseline with different assumptions. The ablations (w/o AdaIN and w/o deg. sim.) control for each component. Using RealEstate10K with simulated test-time degradation (low-light and blur) ensures variations in degradation types. PSNR directly reflects pixel accuracy; if our method truly normalizes features, it should maintain higher PSNR on degraded inputs compared to baselines, while staying competitive on clean scenes. This design allows falsification: if no degradation-specific improvement is observed, the claim fails. Additionally, we compute skewness and kurtosis of features from a calibration set of 512 images to verify the load-bearing assumption about first/second moments.

### Expected outcome and causal chain

**vs. MetaView** — On a case with heavy motion blur and low-light, MetaView extracts features with shifted statistics, leading to incorrect geometry estimation and blurry rendered novel views because it has no mechanism to account for degradation. Our method normalizes these features to a clean reference using AdaIN, preserving geometric accuracy and producing sharper outputs. We expect a noticeable PSNR gap (3-5 dB) on the degraded subset but parity on clean images.

**vs. VGGT** — On a severely low-light input, VGGT's pretrained depth priors may fail due to poor edge detection, causing geometry hallucination. Our degradation simulator trains on diverse degradations, so the feature extractor learns to handle such cases, while AdaIN aligns statistics. We expect our method to outperform VGGT on heavy degradations (2-4 dB gain) but be competitive on mild or clean scenes.

### What would falsify this idea

If our method's PSNR gain over MetaView is uniform across all test images (both clean and degraded), or if it performs worse on degraded images, then the domain invariance claim is disproved—indicating either AdaIN damages clean features or the degradation simulator fails to induce useful invariance.

## References

1. MetaView: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors
2. Depth Anything 3: Recovering the Visual Space from Any Views

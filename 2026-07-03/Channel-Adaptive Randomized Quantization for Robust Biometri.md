# Channel-Adaptive Randomized Quantization for Robust Biometric Augmentation

## Motivation

Existing data augmentations like MixUp (as evaluated in AGVBench) achieve high accuracy in vein recognition but suffer from poor calibration and adversarial vulnerability, while randomized quantization assumes uniform channel redundancy, ignoring that informative channels (e.g., ridge patterns) require higher precision than background noise. StarMixup's interpolation mixes all channels equally, losing fine-grained features. A channel-wise precision allocation based on redundancy (e.g., variance) could simultaneously improve accuracy, calibration, and robustness by preserving discriminatory information while augmenting noisy ones.

## Key Insight

Channel redundancy, measured by variance, inversely correlates with informativeness; thus, allocating more quantization levels to low-redundancy channels preserves discriminative features while aggressive quantization on redundant channels introduces beneficial noise.

## Method

**A) What it is** — Channel-Adaptive Randomized Quantization (CARQ) is a per-channel augmentation that samples quantization bit-widths from a distribution whose mean is a function of channel redundancy, producing augmented images with diverse precision while respecting information content.

**B) How it works** (pseudocode)
```python
def carq(image, num_levels=256, sigma=1, pretrained_model=None):
    # image: (C, H, W) float tensor
    C = image.shape[0]
    # Step 1: Compute per-channel redundancy statistic
    variance = torch.var(image.view(C, -1), dim=1)  # shape (C,)
    # Step 2: Map redundancy to quantization budget (lower variance → higher budget)
    budget = num_levels * (1 - (variance - variance.min()) / (variance.max() - variance.min() + 1e-8))
    budget = torch.clamp(budget, 2, num_levels).int()  # at least 2 levels
    # Step 3: Sample actual bit-width from a narrow Gaussian centered at budget
    bit_width = torch.normal(budget.float(), sigma=sigma).clamp(2, num_levels).int()
    # Step 4: Apply uniform quantization per channel
    # For each channel c, quantize to 2^bit_width[c] levels
    augmented = torch.zeros_like(image)
    for c in range(C):
        levels = 2 ** bit_width[c]
        qmin, qmax = image[c].min(), image[c].max()
        scale = (qmax - qmin) / (levels - 1)
        q = torch.round((image[c] - qmin) / scale) * scale + qmin
        # Add uniform noise with width equal to half scale (stochastic rounding effect)
        noise = torch.rand_like(q) * scale - scale/2
        augmented[c] = q + noise
    return augmented
```
Hyperparameters: `num_levels=256` (max bit-depth), `sigma=1` (sampling spread).

**C) Why this design** — We chose variance over entropy as the redundancy statistic because variance is cheaper to compute (O(CWH) vs O(CWH*log(bins))) and empirically correlates with channel informativeness in vein images (uniform regions have low variance). We use a narrow Gaussian to sample actual bit-width instead of a deterministic mapping to retain stochasticity (key for augmentation diversity), accepting the cost that the Gaussian may occasionally over- or under-assign levels, potentially reducing benefit on some samples. We add uniform noise after quantization (stochastic rounding) to maintain gradient flow and prevent zero-gradient regions, at the cost of slightly increased variance. Three design decisions with trade-offs: (1) variance over entropy: faster but less precise in non-Gaussian channels; (2) Gaussian sampling over hard threshold: diversity but less consistent effect; (3) post-quantization noise over dithering: simpler but may introduce bias.

**D) Why it measures what we claim** — The computational quantity `variance` measures channel redundancy because in natural images, low variance indicates uniform texture (high redundancy) while high variance indicates edges/features (low redundancy); this assumption fails when a channel has high variance due to noise (e.g., sensor noise) rather than informative patterns, in which case variance overestimates informativeness and the method would allocate excessive precision to noise, reducing augmentation benefit. The quantity `bit_width` operationalizes adaptive precision: higher bit width (more levels) for low-redundancy channels preserves discriminatory features (measure of fidelity), while lower bit width for high-redundancy channels introduces quantization noise that acts as augmentation (measure of regularization). The `uniform noise` after quantization operationalizes stochasticity: it prevents overfitting to a fixed quantization grid (measure of diversity). Each component has a named failure mode: variance as redundancy proxy fails when noise dominates; bit-width sampling fails if Gaussian spread is too small (no diversity) or too large (unstable); post-quantization noise fails if scale is too large, washing out signal.

**Load-bearing assumption (explicit):** Channel variance inversely correlates with informativeness, meaning low-variance channels are redundant and can tolerate coarse quantization while high-variance channels contain discriminative features and require high precision. **Calibration procedure:** Prior to using CARQ on a new dataset, we validate this assumption by computing variance per channel and correlating it with class separability (Fisher discriminant ratio) of features extracted from a pretrained CNN (e.g., ResNet-18) on a held-out validation set of 512 images. If the Pearson correlation is below 0.3, we recommend an alternative redundancy statistic such as local entropy (sliding window of 5×5, normalized to [0,1]) or edge density (Canny edge pixels per channel). A scatter plot of variance vs. class separability will be reported in the supplementary material, highlighting channels where the assumption fails (e.g., high variance due to noise).

## Contribution

(1) A novel channel-adaptive randomized quantization algorithm that adjusts quantization bit-width per channel based on a lightweight redundancy statistic (variance). (2) Empirical demonstration that CARQ improves clean accuracy, calibration (lower ECE), and adversarial robustness (higher PGD accuracy) over uniform randomized quantization and mixing-based augmentations in palm- and finger-vein recognition across multiple backbones. (3) An analysis of channel redundancy patterns in vein images, showing that informative channels (vessel patterns) have lower variance and benefit from higher precision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | FV-USM (finger-vein) | Vein images have structured redundancy; we also validate assumption on a held-out calibration set of 512 samples. |
| Primary metric | Recognition accuracy (Rank-1) | Direct measure of augmentation benefit. |
| Baseline 1 | No augmentation | Shows absolute improvement from CARQ. |
| Baseline 2 | Uniform random quantization (same total noise magnitude per channel as CARQ's average quantization noise) | Tests per-channel adaptation necessity and isolates quantization effect. |
| Baseline 3 | Deterministic variance-based quantization (same mapping as CARQ but without Gaussian sampling) | Tests stochasticity benefit. |
| Ablation 1 | CARQ without post-quantization noise | Tests noise contribution. |
| Ablation 2 | Uniform channel-wise noise (same total magnitude as CARQ's quantization noise, applied directly to pixels without quantization) | Isolates quantization effect from noise effect. |

### Why this setup validates the claim

This combination directly tests the three sub-claims: variance-based adaptivity, stochastic sampling, and post-quantization noise. FV-USM contains finger-vein images with channels that have varying redundancy (uniform background vs. vascular patterns), making variance a plausible proxy. The calibration step (correlation of variance with class separability) provides empirical justification for the load-bearing assumption; if correlation < 0.3, we switch to local entropy as redundancy statistic. By comparing against no augmentation, we gauge overall benefit; against uniform random quantization (with matched noise magnitude per channel), we test whether per-channel adaptation matters; against deterministic variance-based quantization, we test whether stochasticity is key. Ablation 1 (no noise) isolates the noise effect, and Ablation 2 (uniform noise per channel) separates quantization from noise. Recognition accuracy captures the core goal of improving feature discrimination and generalization, and failure of any sub-component would manifest as a specific performance pattern: e.g., if uniform random beats CARQ, adaptivity is harmful; if deterministic beats CARQ, stochasticity is unnecessary; if Ablation 2 beats CARQ, quantization noise is detrimental.

### Expected outcome and causal chain

**vs. No augmentation** — On a low-variance channel (e.g., uniform background), the baseline overfits to redundant features, leading to poor generalization under lighting changes. Our method applies coarse quantization to that channel (fewer levels), injecting noise that regularizes the representation. We expect CARQ to show a noticeable accuracy improvement (e.g., 1-2% absolute) on the overall test set, driven by better handling of uniform regions.

**vs. Uniform random quantization** — On a high-variance channel (e.g., sharp vein ridges), uniform random may assign too few levels, losing crucial details, or too many, wasting budget. Our method adapts bit-width based on variance: it allocates high precision to informative channels, preserving discriminability. We expect CARQ to outperform uniform random particularly on images with strong vein contrast, potentially showing a larger margin than on low-contrast samples.

**vs. Deterministic variance-based quantization** — On a borderline channel where variance is ambiguous (e.g., slightly textured background), deterministic assignment always picks one bit-width, limiting diversity across training epochs. Our Gaussian sampling allows occasional over- or under-provisioning, which acts as a stronger regularizer. We expect CARQ to yield higher average accuracy with lower variance across runs compared to the deterministic version.

**vs. Ablation 2 (uniform noise per channel)** — If CARQ's per-channel quantization is essential, it will outperform uniform noise; if uniform noise matches CARQ, then quantization is not necessary and the benefit comes solely from noise injection. We expect CARQ to perform better, confirming that the quantization grid adapted to channel characteristics matters.

### What would falsify this idea

If CARQ performs no better than uniform random quantization on high-variance channels, or if the ablation (no noise) matches full CARQ in accuracy, then the central claims about adaptivity and stochastic noise are invalidated. Additionally, if the calibration scatter plot shows a Pearson correlation < 0.3 between variance and class separability, the load-bearing assumption fails and CARQ should not be applied as-is; a negative result on FV-USM would also falsify the idea for vein recognition.

## References

1. AGVBench: A Reliability-Oriented Benchmark of Data Augmentation for Vein Recognition
2. StarMixup: A More Suitable Mixup Method for Palm-Vein Identification

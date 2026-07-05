# Structure-Aware Normalization for GRPO-based Diffusion RL

## Motivation

Existing GRPO implementations, such as those in Qwen-Image-2.0-RL, apply uniform clipping across timesteps, ignoring the natural increase in ratio variance with timestep in diffusion models. Even GRPO-Guard, which normalizes per timestep, uses a fixed variance target and batch statistics, failing to leverage the predictable trend of variance growth. This structural mismatch leads to over-optimization at high-variance timesteps and under-update at low-variance ones, requiring manual tuning of clipping hyperparameters.

## Key Insight

The ratio variance in diffusion RL grows exponentially with timestep, enabling a parametric model that adapts normalization targets per timestep without relying on noisy batch estimates.

## Method

### (A) What it is
Structure-Aware Normalization (SAN) extends GRPO by modeling per-timestep ratio variance as an exponential function of timestep (σ_t = a·exp(b·t/T)), fitted during a warmup phase, then normalizing ratios to unit variance before clipping. Input: policy π_θ, reference policy π_ref, reward model. Output: normalized ratio for GRPO loss.

### (B) How it works
```pseudocode
# Warmup phase (e.g., 1000 trajectories)
Collect D samples with initial policy
For each timestep t: compute empirical variance var_t of r_t = π_θ/π_ref
Fit σ_t = a * exp(b * t/T) using least squares

# Training loop
For each batch:
  Generate trajectories with current π_θ
  For each timestep t:
    r_t = π_θ(a_t|s_t) / π_ref(a_t|s_t)
    r_t_norm = 1 + (r_t - 1) / σ_t  # normalize to unit variance
    r_t_clip = clamp(r_t_norm, 1-ε, 1+ε)  # ε=0.2
    L_t = -min(r_t_norm * A_t, r_t_clip * A_t)
  Update π_θ with mean(L_t)
```

### (C) Why this design
We chose an exponential parametric model over free per-timestep estimation (e.g., sliding window) because it captures the monotonic trend with only two parameters, reducing variance from batch estimates, at the cost of assuming exact exponential growth which may miss fine-grained fluctuations. We normalize to unit variance before clipping rather than directly adjusting clipping bounds (e.g., ε_t = c·σ_t) because normalization keeps the distribution centered and symmetric, ensuring that the fixed ε=0.2 operates on a consistent scale; adjusting bounds without normalization would shift the ratio's mean and bias updates. We use a separate warmup phase instead of online estimation to avoid unstable early ratios corrupting variance estimates, though this incurs a one-time computational overhead of ~1000 additional trajectories. Alternative approaches like adaptive clipping based on ratio percentiles were avoided because they introduce additional hyperparameters and can be unstable in small batches.

### (D) Why it measures what we claim
The fitted σ_t measures the expected fluctuation magnitude of importance ratios at timestep t because it is derived from empirical variances of a representative warmup distribution; this equivalence holds under the assumption that the ratio distribution's shape (not just variance) remains consistent across training, which fails if the policy shifts significantly from the warmup policy, in which case σ_t may under- or over-estimate true variance. The normalized ratio r_t_norm measures the deviation from the mean relative to timestep-specific variability, ensuring updates are scaled proportionally; this assumes the ratio distribution is approximately symmetric and unimodal, which fails when distribution is heavily skewed (e.g., from extreme rewards), in which case normalization may not center and clipping may become asymmetric. The clamping operation measures the controlled deviation from the normalized mean; this relies on the assumption that unit variance is the appropriate target for stable updates, which is violated if the optimal variance differs per task (e.g., tasks with noisy rewards require lower variance), in which case the fixed ε=0.2 may still cause over-optimization.

## Contribution

(1) Structure-Aware Normalization (SAN), a method that uses an exponential parametric model of per-timestep ratio variance to adapt GRPO clipping for diffusion models, eliminating manual tuning of hyperparameters. (2) Empirical finding that ratio variance in diffusion RL grows exponentially with timestep, providing a simple and interpretable prior for normalization. (3) A plug-in module for GRPO that can be applied to any diffusion model without architectural changes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | COCO Captions (text-to-image) | Standard benchmark for image alignment |
| Primary metric | HPSv3 (human preference) | Measures human-judged alignment; prone to over-optimization |
| Baseline 1 | Standard GRPO | Baseline without ratio normalization |
| Baseline 2 | GRPO-Guard | Adaptive clipping based on ratio percentiles |
| Baseline 3 | PPO with adaptive ε (ε_t ∝ σ_t) | Adjusts clipping but not normalization |
| Ablation-of-ours | SAN with constant σ (no timestep) | Tests exponential fit vs. constant |

### Why this setup validates the claim

The central claim is that modeling per-timestep ratio variance with an exponential function and normalizing before clipping mitigates over-optimization. COCO Captions provides diverse timesteps and prompts, enabling evaluation across the diffusion horizon. Standard GRPO tests the impact of no normalization; GRPO-Guard tests an alternative adaptive strategy that does not center the distribution; PPO with adaptive ε tests changing clipping bounds without normalization. The ablation isolates the exponential fit component. HPSv3 directly captures human preference, which is known to degrade under over-optimization. If SAN consistently achieves higher HPSv3 while maintaining text alignment, it validates that variance normalization is effective and that the exponential parametrization provides additional benefit over simpler alternatives.

### Expected outcome and causal chain

**vs. Standard GRPO** — On a prompt requiring long dependency (e.g., "a bird with detailed feathers"), ratio variance grows with timestep. Standard GRPO uses constant ε=0.2, so small early ratios get overly clipped, while large late ratios overshoot, causing over-optimization on later timesteps and loss of fine details. SAN normalizes each timestep to unit variance, making updates scale invariant. Thus we expect SAN to produce images with better fine detail and higher HPSv3, with the gap concentrated on prompts requiring multi-stage coherence.

**vs. GRPO-Guard** — In a small batch (e.g., 4 trajectories), GRPO-Guard's percentile-based clipping is noisy and can asymmetrically shrink one side of the ratio distribution, biasing updates. SAN's warmup yields stable variance estimates, and normalization centers the distribution. Consequently, SAN avoids bias and achieves more stable learning curves. We expect SAN to have higher HPSv3 and lower variance across runs, especially in early training.

**vs. PPO with adaptive ε** — Consider a scenario where the ratio distribution is symmetric but has mean 1.2 (shifted due to policy drift). Adaptive ε scales ε to match σ, but without centering, the shift persists, causing asymmetric clipping (e.g., more restriction on high ratios). SAN normalizes to mean 1.0, making clipping symmetric. We expect SAN to maintain better exploration and higher HPSv3, with a more balanced KL divergence from the reference.

### What would falsify this idea

If SAN's improvement over baselines is uniform across all timesteps rather than concentrated on high-variance late timesteps, then the exponential variance modeling is not the key driver. Additionally, if SAN underperforms on tasks with heavy-tailed ratio distributions (e.g., extreme rewards), the assumption of symmetric unimodal ratios is violated, suggesting the design hinges on that assumption.

## References

1. Qwen-Image-2.0-RL Technical Report
2. GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping
3. HPSv3: Towards Wide-Spectrum Human Preference Score
4. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
5. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
6. Learning Multi-Dimensional Human Preference for Text-to-Image Generation
7. Boosting Latent Diffusion with Flow Matching
8. AGIQA-3K: An Open Database for AI-Generated Image Quality Assessment

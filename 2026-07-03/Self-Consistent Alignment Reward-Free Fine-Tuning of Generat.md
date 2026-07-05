# Self-Consistent Alignment: Reward-Free Fine-Tuning of Generative Models via Distribution-Level Invariance

## Motivation

Current RL fine-tuning of generative models relies on external reward models (e.g., VisionReward, DanceGRPO) that require expensive human annotations and may introduce biases. These methods share the structural limitation of requiring per-sample or group-level reward scores from an external source, preventing alignment in open-ended settings where reward specification is infeasible. This reliance stems from the assumption that alignment objectives must be defined at the sample level; we argue that a self-supervised consistency objective computed over the generative model's own output distribution can serve as a universal alignment signal.

## Key Insight

Consistency of the generative model's output distribution under semantic-preserving stochastic transformations is a sufficient structural proxy for alignment with desirable properties such as robustness and coherence, because the transformations enumerate an invariant subspace that human preferences implicitly respect.

## Method

**Self-Consistent Alignment (SCA)**  
(A) **What it is**: SCA is a reward-free fine-tuning method for generative models. It takes a pretrained generative model (e.g., diffusion model) and a set of text prompts, and outputs a fine-tuned model without any external reward model. The training signal is a distribution-level consistency loss between the model's output distribution and its distribution under semantic-preserving augmentations.  

(B) **How it works** (Pseudocode with hyperparameters inline):
```python
# Hyperparameters:
#   ENCODER = pretrained image encoder (e.g., CLIP ViT-B/32)
#   AUG_SET = set of random semantic-preserving augmentations
#       (random crop (scale [0.8,1.0]), color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), horizontal flip with p=0.5)
#   N_REF = 16  # reference set size per prompt
#   K = 4       # number of augmentations per sample (for distribution estimate)
#   BETA = 0.01 # learning rate for generator (Adam, β1=0.9, β2=0.999)
#   SUBSET_SIZE = 4  # number of reference samples to replace per step
#   BATCH_SIZE = 8 prompts per GPU
#   TOTAL_STEPS = 10000  (approximately 4 A100-40GB GPUs for 3 days on 30K prompts)
#   MEMORY: reference sets store 30K * 16 images (approx 8GB for 256x256 RGB)

for each training step:
    1. Sample batch of prompts {p_i} from dataset (BATCH_SIZE=8).
    2. For each prompt p_i:
        a. Retrieve current reference set R_i = {x_1,...,x_{N_REF}} (generated in previous step or initialized with model samples from pretrained model).
        b. For each x in R_i, apply K random augmentations from AUG_SET to get {x'_1,...,x'_K}.
        c. Encode all original samples via ENCODER -> {E(x)} (dim=512).
           Encode all augmented samples via ENCODER -> {E(x')} (pooled over augmentations).
        d. Compute Gaussian approximation:
           - mu_orig, Sigma_orig = mean and covariance of {E(x)} (covariance computed with np.cov, regularized with 1e-6 * I)
           - mu_aug, Sigma_aug = mean and covariance of {E(x')}
        e. Compute consistency loss (symmetric KL):
           L_cons = 0.5 * (KL(N(mu_orig, Sigma_orig) || N(mu_aug, Sigma_aug)) +
                            KL(N(mu_aug, Sigma_aug) || N(mu_orig, Sigma_orig)))
           # KL(N(μ1, Σ1) || N(μ2, Σ2)) = 0.5 * (tr(Σ2^{-1} Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1) - d + log(|Σ2|/|Σ1|))
    3. Total loss = avg(L_cons over batch).
    4. Update generator parameters via gradient descent (Adam, lr=BETA).
    5. For each prompt p_i, replace SUBSET_SIZE randomly selected samples in R_i with new samples from current generator (without gradient).
```

(C) **Why this design**: We chose to compute the consistency loss as the symmetric KL divergence between Gaussian approximations of the output distributions for original and augmented sets, rather than using a contrastive loss (e.g., SimCLR-style), because the distribution-level objective directly captures the invariance of the entire generated set, which aligns with alignment desiderata like diversity and coverage. The Gaussian approximation enables tractable computation even with few samples, accepting the cost that the true distribution may be non-Gaussian, potentially misrepresenting multimodality. A key assumption of this design is that the Gaussian approximation is sufficient to capture the distributional invariance that correlates with alignment; this assumption may be violated when the true output distribution is multimodal, leading to reduced diversity. We use a pretrained encoder (CLIP) rather than learning one jointly to avoid collapse and ensure semantic features transfer; this trades domain-specificity for robustness. The augmentation set is designed to be semantic-preserving (e.g., random crop (scale [0.8,1.0]), color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), horizontal flip with p=0.5) to ensure that the invariance required is meaningful; if augmentations destroy semantics, the loss penalizes irrelevant variance. The subset-replace strategy for the reference set reduces computational overhead compared to regenerating all samples each step, but introduces temporal correlation between successive losses, which may slow convergence. These choices prioritize scalability and simplicity over theoretical optimality.

(D) **Why it measures what we claim**: The consistency loss L_cons measures the model's robustness to semantic-preserving perturbations, which we claim is a proxy for alignment with human preferences. The computational quantities: mu_orig and Sigma_orig capture the central tendency and spread of the original output distribution; mu_aug and Sigma_aug capture the same for the augmented outputs. The symmetric KL divergence measures the mutual information between the two distributions under the Gaussian assumption. The underlying assumption is that the augmentation set spans an invariant subspace that human preferences implicitly respect. This assumption fails when the transformation is too aggressive (e.g., cropping removes the main subject) — then the loss reflects sensitivity to the transformation rather than semantic invariance; or when the encoder's features are not invariant to the transformation (e.g., color jitter shifts the hue, which may change semantics in fine-grained tasks) — then the loss penalizes changes the encoder is sensitive to, which may not align with human judgment. The Gaussian approximation assumes unimodality; if the true distribution is multimodal (e.g., multiple valid interpretations of a prompt), the KL divergence penalizes multimodality, reducing diversity, which is misaligned with the aim of preserving variety. The subset-replace strategy introduces a non-stationary reference set, meaning the loss measures consistency relative to an evolving target, which may cause oscillations.

**Load-bearing assumption explicit**: The consistency loss (symmetric KL divergence between Gaussian approximations of original and augmented output distributions) is a valid proxy for alignment with human preferences. This assumption is load-bearing because if it fails, the method optimizes an objective uncorrelated with alignment. We verify this assumption by comparing to a non-parametric variant (see ablation) and by analyzing failure modes on fine-grained tasks.

## Contribution

(1) A reward-free fine-tuning framework for generative models that replaces external reward models with a self-consistency objective computed over the model's output distribution under semantic-preserving augmentations. (2) A practical algorithm combining subset-replace reference set updating, Gaussian distribution approximation, and a pretrained encoder for scalable distribution-level consistency regularization. (3) Identification of the invariant subspace property as a sufficient alignment proxy, enabling alignment in settings without human annotations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MS COCO validation set (30K prompts) | Standard benchmark for text-to-image. |
| Primary metric | FID-50K | Measures distributional similarity to real images. |
| Baseline 1 | Supervised SFT on high-quality subset | Tests if consistency adds value over fixed data. |
| Baseline 2 | Reward-based RL (PPO + ImageReward) | Tests reward-free vs. reward-driven alignment. |
| Baseline 3 | Pretrained model (no fine-tuning) | Establishes base capability. |
| Ablation of ours | SCA with contrastive loss (SimCLR) | Isolates effect of symmetric KL divergence. |

### Why this setup validates the claim

This experimental design forms a falsifiable test by isolating the mechanism of distribution-level consistency. Comparing SCA to supervised SFT tests whether the invariance objective provides benefits beyond imitating a fixed dataset, which would manifest as better diversity and robustness on ambiguous prompts. The reward-based RL baseline tests whether reward-free alignment can match or exceed the alignment achieved with explicit human rewards, crucially avoiding reward hacking. The pretrained baseline establishes the starting point. The ablation, replacing symmetric KL with contrastive loss, tests the specific design choice for handling multimodality. FID is chosen because it directly measures the distributional similarity we aim to improve, capturing both quality and diversity. If SCA fails to outperform the relevant baseline in the predicted failure regime, the core hypothesis is contradicted.

### Expected outcome and causal chain

**vs. Supervised SFT** — On a prompt like "a bird", where a fixed high-quality dataset contains only a few species, supervised SFT collapses to those species because it learns a narrow conditional distribution. SCA instead encourages the model to produce a diverse set of birds (different poses, colors) by penalizing variance reduction under augmentations. We expect SCA to show a noticeable FID improvement on ambiguous prompts (FID ~5 points lower) while performing similarly on specific prompts.

**vs. Reward-based RL (PPO+ImageReward)** — On a prompt like "a vibrant sunset", the ImageReward model may over-reward high saturation, causing PPO to produce unnaturally vivid colors. SCA, being reward-free, only enforces consistency under color jitter augmentations, which penalizes drastic saturation changes. Thus SCA maintains natural color distributions. We expect SCA to have lower FID by ~3 points on color-sensitive categories, with RL method showing a clear distribution shift in hue statistics.

**vs. SCA with contrastive loss** — On a prompt with multimodal interpretation (e.g., "a dog in a field" could be many compositions), contrastive loss pushes augmented samples apart in embedding space, reducing diversity. Symmetric KL instead matches the full Gaussian distributions, preserving multimodality. We expect SCA (KL) to produce larger intra-prompt diversity (measured by LPIPS variance) by ~10% than the contrastive variant on abstract prompts.

### What would falsify this idea

If SCA fails to outperform the supervised SFT baseline on ambiguous prompts (i.e., no FID gap beyond noise), or if its gain is uniform across all prompt types rather than concentrated in the predicted failure modes (e.g., diversity collapse, color bias), then the central claim that distribution-level consistency provides unique alignment benefits is not supported.

## References

1. Optimizing Visual Generative Models via Distribution-wise Rewards
2. Model Merging in Pre-training of Large Language Models
3. DanceGRPO: Unleashing GRPO on Visual Generation
4. VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation
5. HunyuanVideo: A Systematic Framework For Large Video Generative Models
6. Align Video Diffusion Model with Online Video-Centric Preference Optimization
7. What Matters for Model Merging at Scale?
8. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations

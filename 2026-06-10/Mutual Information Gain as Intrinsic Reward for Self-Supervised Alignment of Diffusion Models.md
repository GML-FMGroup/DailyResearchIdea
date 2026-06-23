# Mutual Information Gain as Intrinsic Reward for Self-Supervised Alignment of Diffusion Models

## Motivation

Existing RL fine-tuning for diffusion models (e.g., ESPO, d2, TraceRL) rely on external reward signals from human preferences or task-specific metrics, which are expensive to obtain and limit scalability. This dependency stems from the assumption that alignment requires an external signal. However, the denoising process inherently provides a rich signal: the stepwise reduction in uncertainty about the final clean image. By exploiting this, we can define a self-supervised intrinsic reward that requires no human annotations.

## Key Insight

Mutual information between consecutive denoising steps quantifies the information gain about the final clean sample; maximizing this gain inherently drives the model toward higher-quality generation because optimal denoising steps are those that most reduce uncertainty.

## Method

### Self-Rewarding Diffusion Alignment (SRDA)

**Load-bearing assumption:** The mutual information between consecutive denoising steps, estimated via the model's own score predictions under a Gaussian assumption, accurately measures the information gain about the final clean image and serves as a valid intrinsic reward for improving generation quality. However, score estimation is inaccurate in low-density regions (early steps), leading to unreliable estimates. We address this by replacing the Gaussian-based estimator with a non-parametric Kraskov nearest-neighbor estimator (k=5) for MI, which does not assume Gaussianity and uses samples from the rollout directly.

**Pseudocode:**
```
Input: Pretrained diffusion model p_θ(x_{t-1}|x_t) with score function s_θ(x_t,t)
Hyperparameters: K rollout steps, γ=0.99 discount, η=0.9 entropy bonus coefficient

For each training iteration:
  1. Sample initial noise x_T ~ N(0,I)
  2. Rollout trajectory (x_T, x_{T-1}, ..., x_0) using current denoising policy (the model)
  3. For each step t from T down to 1:
     a. Estimate mutual information MI_t between x_{t-1} and x_t:
        - Generate N=100 pairs (x_t^{(i)}, x_{t-1}^{(i)}) from rollout (using same x_T? Actually use each step's samples)
        - Use Kraskov estimator (k=5) on these samples to compute MI_t directly, without Gaussian assumption.
     b. Reward r_t = MI_t (normalized by dividing by max over steps in trajectory)
  4. Compute advantages A_t = discounted returns minus baseline (value network with hidden size 256)
  5. Policy gradient update: θ ← θ + α ∇_θ Σ_t log p_θ(x_{t-1}|x_t) * A_t + η * entropy_bonus
  6. Update value network via MSE loss (learning rate 1e-4)
```

**Why this design:** We chose mutual information over simple variance reduction because it directly measures information gain, aligning with the goal of informative denoising. We originally used score-based estimation of MI via Tweedie's formula, but due to inaccuracies in low-density regions, we replaced it with a non-parametric Kraskov estimator to avoid reliance on the Gaussian assumption and model calibration. We applied a discount factor γ to weight early steps less, as they are less critical for final quality; the trade-off is that heavily discounting may ignore important early denoising. We included an entropy bonus to maintain diversity, preventing policy collapse from always taking the most informative steps. This design avoids external rewards by using the model's own scores.

**Why it measures what we claim:** The computational quantity MI_t (estimated via Kraskov) measures the information gain about the final clean image between steps because it directly estimates the mutual information between consecutive latent states without parametric assumptions; this holds provided the rollout samples are representative of the denoising distribution. This assumption fails during early training when the model produces poor samples; in that case, MI_t may reflect model uncertainty rather than true information gain, but the non-parametric estimator is more robust than the Gaussian one.

## Contribution

(1) A self-supervised intrinsic reward for diffusion models derived from mutual information of the denoising process, eliminating the need for external annotations. (2) A practical algorithm for estimating this reward using the model's score predictions, enabling RL fine-tuning without human feedback. (3) Demonstration that the proposed reward improves sample quality across continuous image generation tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | CIFAR-10 (50k training images, 32x32) + ImageNet 64x64 (1.2M training images) | Standard for image generation; larger scale for generality |
| Primary metric | FID (↓) (10k samples), Recall (↑) for diversity | Measures sample quality & diversity |
| Secondary metrics | FLOPs per iteration (M), Spearman correlation between per-step MI and per-step FID improvement | Computational cost; causal link evidence |
| Baseline 1 | Pretrained diffusion (DDPM with cosine noise schedule, 1000 steps) | No fine-tuning reference |
| Baseline 2 | RL with classifier reward (trained on CIFAR-10 labels) | Uses external reward signal |
| Baseline 3 | Diffusion-DPO (preference pairs from classifier scores) | Preference-based RL for diffusion |
| Baseline 4 | ICM intrinsic reward on diffusion (feature space: hidden layer of a pretrained ResNet-18) | Intrinsic reward baseline from RL |
| Baseline 5 | RND intrinsic reward on diffusion (target network: 4-layer MLP, hidden=512) | Another intrinsic reward baseline |
| Ablation-of-ours | SRDA w/o entropy bonus (η=0) | Isolates diversity effect |
| Ablation-of-ours | SRDA with negative MI reward (r_t = -MI_t) | Tests if MI direction is causal |
| Ablation-of-ours | SRDA with variance reward (r_t = Var(x_{t-1}|x_t)) | Alternative information-theoretic reward |

### Why this setup validates the claim

This combination tests the central claim—that mutual information as an intrinsic reward improves sample quality without external supervision—through a controlled ladder: Pretrained diffusion establishes the baseline without any RL. The classifier-reward baseline tests whether a handcrafted external reward outperforms a self-supervised one. Diffusion-DPO tests an alternative self-supervised method based on preferences. Baselines 4 and 5 test whether standard intrinsic rewards from RL (ICM, RND) can be applied to diffusion, isolating the novelty of using mutual information. The ablation without entropy bonus verifies that both MI and diversity maintenance are necessary. The negative MI ablation checks if the reward sign is correct. The variance reward ablation tests whether a simpler uncertainty reduction measure works. FID is chosen because it jointly captures fidelity and diversity, which SRDA aims to improve via informative denoising trajectories. The Spearman correlation analysis provides direct empirical evidence for the causal link: if MI is indeed driving improvements, steps with higher MI should correlate with larger FID drops when that step is modified. If SRDA outperforms all baselines and shows significant positive correlation, the claim is strongly supported.

### Expected outcome and causal chain

**vs. Pretrained diffusion** — On a case where the pretrained model produces blurry samples due to inefficient denoising, the baseline fails to enhance detail because it has no feedback to refine steps. Our method instead rewards steps that maximally reduce uncertainty (high MI), focusing denoising on critical transitions, so we expect a noticeable FID gap (~5-10 points) with SRDA, especially on images with fine textures on CIFAR-10, and similar gains on ImageNet 64x64 with appropriate hyperparameters (same schedule, LR 1e-4, batch size 256, 500 steps).

**vs. RL with classifier reward** — On a case where a classifier reward overfits to class-discriminative features, the baseline may generate exaggerated textures or mode-collapse because it optimizes external scores. Our self-reward, based on information gain, avoids such bias, so we expect SRDA to achieve lower FID (by ~3-5 points) and higher diversity (e.g., higher recall) on CIFAR-10.

**vs. Diffusion-DPO** — On a case where preference pairs are noisy (e.g., ambiguous quality differences), DPO may learn suboptimal rewards, limiting improvement. Our intrinsic MI is directly tied to denoising dynamics and is robust to such noise, so we expect SRDA to achieve a clear FID advantage (~2-4 points) and faster convergence (fewer iterations to reach best FID).

**vs. ICM** — ICM uses prediction error in a learned feature space as reward. Applied to diffusion, it may reward steps that produce diverse but not necessarily informative transitions. We expect SRDA to outperform ICM by ~5-10 FID points because MI directly targets information gain about the final image, whereas ICM focuses on novelty in a separate feature space.

**vs. RND** — RND uses prediction error of a fixed random network. It may encourage exploration but not correlate with denoising improvement. SRDA should show a clearer FID gain (3-7 points) and higher Spearman correlation between reward and FID improvement.

**Ablations**: SRDA without entropy bonus will show lower diversity (lower recall) but similar FID; negative MI reward will likely degrade performance (higher FID), confirming that maximizing MI is beneficial. Variance reward may show some improvement but less than MI because variance does not capture higher-order dependencies.

### What would falsify this idea

If SRDA’s FID improvement is uniform across all sample subsets rather than concentrated on those where early denoising steps have high MI variance, it would indicate the reward is not driving the effect and the claim is false. Also, if the Spearman correlation between per-step MI and per-step FID improvement is insignificant or negative, the causal link is unsupported.

## References

1. Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages
2. Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies
3. Dream 7B: Diffusion Large Language Models
4. d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation
5. Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective
6. Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models
7. DiFFPO: Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning
8. SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models

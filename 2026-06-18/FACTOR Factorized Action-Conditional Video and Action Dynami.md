# FACTOR: Factorized Action-Conditional Video and Action Dynamics with Sparse Coupling for Robot Learning

## Motivation

Existing methods like DiT4DiT require large paired video-action datasets for joint modeling, limiting scalability. The root cause is the assumption that video and action dynamics are tightly coupled, but we hypothesize a structural property: most latent dimensions are independent across modalities, with only a sparse set of interacting dimensions. This motivates a factorized latent representation that can leverage abundant unpaired data by learning modality-specific dynamics and a sparse coupling term, calibrated with a small paired dataset.

## Key Insight

Sparse coupling between video and action latent dynamics enables factorized training on unpaired data while retaining cross-modal consistency through a small set of paired examples, because the independence of most latent dimensions allows marginal likelihood maximization without joint supervision.

## Method

**FACTOR** (Factorized Action-Conditional Video and Action Dynamics with Sparse Coupling) learns a shared latent space split into three components: video-specific ($z_v$), action-specific ($z_a$), and coupled ($z_c$). The framework uses a VAE with structured prior and separate dynamics models. We explicitly assume that the coupling between video and action latents is sparse; if this assumption is violated, the learned mask will saturate and model degrades, which we verify via diagnostic experiments.

**(B) Core Mechanism (pseudocode)**
```pseudocode
# Hyperparameters: beta=0.01 (sparsity weight), lambda=0.1 (coupling weight), mask M learned via Gumbel-sigmoid with initial temperature 1.0
# Encoder/decoder: 3D convolutional for video (ResNet-18 style), 2-layer MLP (hidden=256, ReLU) for actions
# Dynamics: LSTM for video (hidden=512, 2 layers), 2-layer MLP (hidden=256, ReLU) for actions, coupling: z_c_pred = MLP_coupling(z_c_a, M) with 2 hidden layers (256, 128, ReLU)
# Optimizer: Adam, lr=1e-4, cosine annealing schedule, batch size 64 (paired: 16, unpaired video: 32, unpaired action: 16)
# Training on 4x A100 GPUs, ~100 epochs for Maniskill2 (2M frames)
for each batch:
    if unpaired video batch:
        z_v, z_c_v = encoder_video(x_v)  # x_v: 4x64x64 RGB-D, output means and logvars
        loss_vae_v = -ELBO(x_v | z_v, z_c_v) + beta * sparsity_prior(z_c_v)  # sparsity_prior: L1 norm on z_c_v
        update encoder_video, decoder_video, dynamics_v
    if unpaired action batch:
        z_a, z_c_a = encoder_action(x_a)  # x_a: 7-dim joint positions/velocities, output means and logvars
        loss_vae_a = -ELBO(x_a | z_a, z_c_a) + beta * sparsity_prior(z_c_a)
        update encoder_action, decoder_action, dynamics_a
    if paired batch:
        z_v, z_c_v = encoder_video(x_v)
        z_a, z_c_a = encoder_action(x_a)
        # Cross-modal coupling: predict z_c_v from z_c_a using learned mask M
        z_c_pred = dynamics_coupling(z_c_a, M)  # z_c_pred = (M * W) z_c_a + b, where M is binary mask learned via Gumbel-sigmoid
        loss_coupling = MSE(z_c_v, z_c_pred)
        loss_vae_joint = -ELBO(x_v, x_a | z_v, z_a, z_c_v, z_c_a) + beta * sparsity_prior(z_c)  # z_c = [z_c_v, z_c_a]? Actually separate
        update all modules with loss = loss_vae_joint + lambda * loss_coupling
```

**(C) Why this design**
We chose a VAE over diffusion models because VAEs provide explicit latent distributions enabling marginal likelihood estimation for unpaired data; the trade-off is lower sample fidelity, but the structured latent is essential for factorization. We split latents into three components (video-specific, action-specific, coupled) rather than a single shared latent, because enforcing independence on two subspaces while allowing a sparse coupling head reduces the risk of mode collapse and improves interpretability; the cost is increased parameter count and training complexity. We use a learned binary mask for the coupling term rather than a dense linear layer, because sparsity is a core assumption; the mask selects which dimensions interact, imposing a hard constraint that forces the model to use only a few dimensions for cross-modal consistency. This design accepts that if the true coupling is dense, the mask will force remaining information into modality-specific components, reducing coupling accuracy but preserving individual dynamics quality.

**(D) Why it measures what we claim**
The ELBO for unpaired video measures marginal likelihood $p(video)$ under the factorized model — this operationalizes the assumption that video dynamics are mostly independent of actions, so the video-only branch can model them without action input; this assumption fails when video dynamics are strongly action-dependent (e.g., object trajectories determined by robot motion), in which case the ELBO will be low and the model will fail to capture action-driven variations. The sparsity prior's L1 norm on $z_c$ measures the number of active coupling dimensions — this operationalizes the assumption that coupling is sparse; when the true coupling is dense, the L1 norm will be high and the prior will be violated, causing the model to push information into modality-specific components, reducing cross-modal predictive accuracy. The coupling loss MSE between $z_c$ from video and action branches measures cross-modal consistency — this operationalizes the assumption that the coupled latent captures shared content (e.g., task goals); if the encoder fails to align shared content (e.g., due to observation noise), the MSE will be high, indicating poor alignment. The mask M measures which dimensions are coupled — this operationalizes the structural property that only a few dimensions need to interact; if tasks require many coupled dimensions, the mask will saturate, indicating the sparsity assumption's limit. Additionally, we test whether high MSE on the coupling loss correlates with downstream task failures by logging per-episode MSE and success; if a clear threshold (e.g., MSE > 0.1) predicts failure, it validates that alignment in latent space corresponds to shared content.

## Contribution

(1) FACTOR, a novel framework for learning factorized video and action dynamics with sparse coupling, enabling training on unpaired or weakly paired data by separate marginal likelihood maximization and a small paired set for coupling calibration. (2) A design principle that exploiting structural independence between modalities can reduce reliance on large paired datasets in robot learning, demonstrated through the FACTOR framework.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Maniskill2 (2M frames, 500 tasks) | Tests 3D manipulation with varied actions. |
| Primary metric | Task Success Rate (%) | Measures downstream task performance directly. |
| Baseline | DiT4DiT | State-of-the-art video-action joint model. |
| Baseline | Cosmos-Predict1 | Video foundation model for world simulation. |
| Baseline | Multi-modal VAE with Product-of-Experts | Highlights difference from factorized coupling (no explicit sparse coupling). |
| Baseline | Multi-modal VAE with Mixture-of-Experts | Highlights difference from factorized coupling (no explicit sparse coupling). |
| Ablation | FACTOR without sparse mask (dense linear coupling) | Tests sparsity assumption necessity. |
| Ablation | FACTOR with increasing unpaired data (0, 50%, 100% of total) | Demonstrates scaling benefit of factorized training. |
| Diagnostic | Mutual Information (MI) between paired video and action latents (ground truth vs. learned) | Measures true sparsity in data; compare to learned mask density. |
| Diagnostic | Correlation between coupling loss MSE and task success per episode | Verifies that alignment in latent space corresponds to shared content. |

### Why this setup validates the claim

The combination of Maniskill2, which requires understanding action-driven 3D dynamics, with baselines DiT4DiT (joint video-action model) and Cosmos-Predict1 (purely video-based world model) creates a clear contrast: DiT4DiT intertwines video and action features, while Cosmos-Predict1 ignores action conditioning. Our method FACTOR claims that sparse coupling is optimal for cross-modal consistency. The primary metric, task success rate, directly reflects world model quality for control. The ablation with dense coupling isolates the effect of sparsity. If FACTOR outperforms baselines and ablation, it validates that sparse coupling efficiently captures shared dynamics without diluting modality-specific information. Failure patterns—e.g., DiT4DiT failing on action-variable tasks, Cosmos failing on action-critical steps—would confirm our assumptions. This design falsifies if FACTOR does not improve precisely on cases where coupling should be sparse.

### Expected outcome and causal chain

**vs. DiT4DiT** — On a task where the video shows a predictable outcome but the action sequence is atypical (e.g., object moves due to robot but human consistently intervenes), DiT4DiT couples video and action features densely, so it may overfit to typical action-video correlations and fail when actions diverge from video expectations. Our method splits latents: video-specific captures scene dynamics, action-specific captures motor commands, and only a few coupled dimensions link them via a learned mask. Thus, when actions are unusual, the action-specific branch can compensate without disrupting video predictions. We expect FACTOR to outperform DiT4DiT by >10% success rate on tasks requiring novel action sequences, while matching on standard tasks. The product-of-experts and mixture-of-experts baselines, which combine modalities probabilistically, will likely perform worse than FACTOR on unpaired data because they require paired data for training the expert combination.

**vs. Cosmos-Predict1** — On a manipulation task where the next frame depends critically on the applied action (e.g., pushing a block to a goal), Cosmos-Predict1, as a video-only model, predicts the future without action input, leading to high uncertainty or hallucinations of implausible outcomes. Our method includes action-specific latents that explicitly condition future predictions on actions via the coupled component. Sparse coupling ensures that only relevant action dimensions affect video dynamics. Therefore, we expect FACTOR to achieve >20% higher success rate on action-conditional tasks, especially those with high action-variability, while Cosmos-Predict1 may degrade on long horizons due to accumulating error.

### What would falsify this idea

If FACTOR's gain over DiT4DiT is uniform across all task subsets rather than concentrated on cases with weakly coupled video-action dynamics (i.e., where sparsity should help), then the sparse coupling assumption is not the source of improvement. Also, if the ablation with dense mask matches or exceeds FACTOR's performance on all tasks, then sparsity is unnecessary. Additionally, if the mutual information diagnostic shows that the true coupling is dense (high MI across many dimensions) and FACTOR's mask fails to capture it, then the core assumption is violated.

## References

1. DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
2. World Simulation with Video Foundation Models for Physical AI
3. VideoVLA: Video Generators Can Be Generalizable Robot Manipulators
4. π*0.6: a VLA That Learns From Experience
5. Diffusion Policy Policy Optimization
6. π0: A Vision-Language-Action Flow Model for General Robot Control
7. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
8. Open-Sora: Democratizing Efficient Video Production for All

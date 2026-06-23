# PhysLat: Physics-Constrained Latent Space for LLM-Driven Trajectory Generation in World Models

## Motivation

Existing LLM-enhanced world models like DriveDreamer-2 generate trajectories that lack physical constraints, leading to implausible motions in novel scenarios. This is because the LLM samples trajectories in an unconstrained space without any enforcement of physical realizability, relying instead on learned priors that fail for out-of-distribution inputs. A differentiable physics simulator as decoder guarantees plausibility by construction, eliminating the need for post-hoc verification.

## Key Insight

Because a differentiable physics simulator as decoder maps any latent code to a physically valid trajectory, the latent space is inherently a manifold of physically plausible motions, guaranteeing plausibility without post-hoc constraints.

## Method

### (A) What it is
PhysLat is a variational autoencoder (VAE) where the decoder is a differentiable physics simulator (e.g., a simplified kinematic bicycle model). It takes a latent code \(z\) and outputs a physically plausible trajectory (sequence of 2D positions) via forward simulation. Input: ground-truth trajectories. Output: latent codes that encode physically realizable motions.

### (B) How it works
```python
# Training
# Real trajectory dataset D = {(x_i)} where x_i = [p_0, p_1, ..., p_T] (positions, shape T×2)
# Initialize encoder E_θ (GRU with hidden size 128 → mu, sigma → 32-dim latent)
# Initialize decoder D_φ as differentiable bicycle model (params: initial speed, steer angle rate)

for each batch:
    x = sample_batch(D)  # shape (batch, T, 2)
    # Encode
    mu, logvar = E_θ(x)
    sigma = exp(0.5 * logvar)
    z = mu + sigma * N(0,1)  # reparameterization
    # Decode via differentiable physics simulator
    # Simulator takes z (which encodes initial state and control parameters) and runs T steps
    x_hat = D_φ(z)  # shape (batch, T, 2)
    # Reconstruction loss (MSE) + KL divergence
    recon_loss = MSE(x, x_hat)  # scalar
    kl_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    loss = recon_loss + 0.01 * kl_loss
    # Backpropagate through D_φ (differentiable)
    update θ, φ

# After VAE training, fine-tune an LLM (e.g., DriveDreamer-2's LLM) on (text, z) pairs
# LLM input: text; output: latent code z (via MSE loss on z)
```

### (C) Why this design
We choose a differentiable physics simulator as decoder over a standard neural decoder for three reasons. (1) **Guaranteed physical plausibility**: any latent code maps to a valid trajectory under the simulator, eliminating the need for post-hoc constraints or verification. This is structurally sound because the decoder is not a learned approximation but a fixed physical model. (2) **Interpretable latent space**: each latent dimension corresponds to physical parameters (initial velocity, steering angle rate), enabling direct human inspection and control. A standard decoder would entangle these factors. (3) **Trainable via backpropagation**: the differentiable simulator allows end-to-end training without reinforcement learning or additional regularizers. The trade-off is that the simulator must be a simplified model (e.g., kinematic bicycle) that may not capture complex dynamics like tire slip or collisions. However, for highway driving, such simplifications suffice for overall plausibility, and the encoder can compensate for minor inaccuracies by learning to ignore non-essential variations.

### (D) Why it measures what we claim
The reconstruction loss \(||x - x_hat||^2\) measures physical fidelity because the decoder \(D_φ\) is a fixed physics simulator; if a latent code \(z\) cannot be simulated to reproduce the ground truth trajectory \(x\), then \(x\) is not physically realizable under the simulator's dynamics. The assumption this equivalence relies on is that the simulator's dynamics are a valid approximation of real-world physics for the target domain (structured driving). This assumption fails when the true trajectory involves maneuvers outside the simulator's modeling capacity (e.g., aggressive drifting), in which case the reconstruction loss may be high even for physically plausible trajectories—the metric then reflects mismatch to the simulator rather than to reality. The KL divergence term measures latent space regularity (closeness to standard normal), not physical fidelity; its role is to encourage smooth interpolation, which is orthogonal to the plausibility guarantee provided by the decoder.

## Contribution

(1) A novel framework that integrates a differentiable physics simulator as the decoder of a VAE to create a physics-constrained latent space for trajectory generation, ensuring all decoded outputs are physically plausible by construction. (2) A demonstration that a pre-trained LLM (e.g., from DriveDreamer-2) can be fine-tuned to generate trajectories by sampling in this latent space, eliminating the need for post-hoc verification. (3) A design principle that using a structured, physically-grounded decoder imposes a strong inductive bias, reducing the required scale of training data for plausible motion generation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | nuScenes trajectory dataset | Real driving trajectories vary in speed and curvature |
| Primary metric | Physical plausibility rate | Tests whether decoder guarantees plausible motions |
| Baseline 1 | VAE with MLP decoder | No physics constraints; can produce unrealistic paths |
| Baseline 2 | VAE with GRU decoder | Sequential but no physics knowledge |
| Baseline 3 | VAE with non-differentiable physics decoder | Same physics but trained differently; isolates diff. advantage |
| Ablation | PhysLat without built-in physics (regularized) | Tests importance of fixed physics vs. learned penalty |

### Why this setup validates the claim
The chosen dataset of real driving trajectories from nuScenes provides diverse and realistic motions. The primary metric, physical plausibility rate, directly tests the central claim that the differentiable physics decoder guarantees physically realizable outputs. By comparing against baselines that lack physics constraints (MLP, GRU), we isolate the effect of the physics decoder. The third baseline (non-differentiable physics decoder) controls for the type of physics model, testing the advantage of differentiability. Finally, the ablation where we replace the fixed physics decoder with a regularized neural decoder assesses whether the guarantee stems from the built-in physics rather than a learned penalty. This combination forms a falsifiable test: if PhysLat achieves higher plausibility than all baselines and the ablation, the claim is supported; if not, the decoder may not be effective.

### Expected outcome and causal chain
**vs. VAE with MLP decoder** — On a case where the true trajectory has a tight turn, the MLP decoder may output a path with unrealistic acceleration or sharp kinks because it lacks any physics prior. Our method instead enforces smooth turns limited by the bicycle model's steering constraints, so the reconstructed trajectory will be physically plausible. Thus we expect Plausibility Rate near 100% for PhysLat vs. ~70% for MLP VAE on the subset of turning trajectories, while on straight trajectories both may be high.

**vs. VAE with GRU decoder** — On a case with high-speed braking, the GRU decoder can produce deceleration patterns that violate physics (e.g., instantaneous speed change) because it learns temporal correlations but not underlying dynamics. Our method models deceleration via the bicycle model's braking parameter, resulting in smooth deceleration. Therefore we expect PhysLat to have >95% plausibility on braking sequences vs. ~80% for GRU VAE.

**vs. VAE with non-differentiable physics decoder** — On a case requiring precise adaptation to a complex trajectory, the non-differentiable decoder trained via REINFORCE may not align latent codes perfectly due to high variance gradients, leading to higher reconstruction errors. PhysLat with differentiable simulator achieves lower reconstruction MSE because gradients flow directly. We expect PhysLat to have 30% lower reconstruction error on average, while plausibility rates may be similar.

### What would falsify this idea
If PhysLat's plausibility rate is not significantly higher than that of the regularized ablation (which uses a learned physics penalty), then the fixed physics decoder is not the source of the guarantee. Also, if all baselines achieve near-perfect plausibility on the dataset (e.g., because real trajectories are already smooth), then the idea is not falsifiable. The key diagnostic is that PhysLat's advantage is concentrated on trajectories with complex maneuvers (turns, braking, acceleration) where physics constraints matter.

## References

1. DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation
2. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
3. VideoPoet: A Large Language Model for Zero-Shot Video Generation
4. SparseFusion: Distilling View-Conditioned Diffusion for 3D Reconstruction
5. Latent Video Diffusion Models for High-Fidelity Long Video Generation

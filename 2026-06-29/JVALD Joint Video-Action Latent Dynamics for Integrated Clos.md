# JVALD: Joint Video-Action Latent Dynamics for Integrated Closed-Loop Robotics Control

## Motivation

Existing video generation models for robotics, such as PhysicsForcing, produce high-quality videos but require a separate action extraction post-processing module to obtain control signals. This decoupling introduces latency and potential misalignment between the generated video and the actions needed for real-time control. The root cause is that these models are trained solely for visual prediction, without enforcing that their latent representations are informative for action inference. We address the meta-gap of integrated action-conditional closed-loop generation by designing a latent dynamics model that jointly learns to generate video frames and infer actions through shared state transitions, eliminating the additional post-processing step.

## Key Insight

Jointly optimizing forward (latent+action→next latent) and inverse (latent+next latent→action) dynamics in a shared latent space forces the representation to be simultaneously predictive of visual futures and sufficient for control, enabling closed-loop operation without separate action extraction.

## Method

### (A) What it is:
JVALD (Joint Video-Action Latent Dynamics) is a generative model that produces video frames and corresponding actions by evolving a shared latent state via learned forward and inverse dynamics. Its inputs are an initial image and a sequence of actions (during training) or a goal latent (during closed-loop control); its outputs are a video sequence and per-step actions.

### (B) How it works:
```python
# Architecture specifications:
# Encoder: 4-layer CNN (32,64,128,256 channels, 4x4 kernel, stride 2, ReLU) + 2-layer MLP (hidden 512, output latent dim 128).
# Decoder: mirror of encoder (transposed conv).
# Forward dynamics F: 3-layer MLP (hidden 256,128; output latent dim 128; ReLU).
# Inverse dynamics G: 3-layer MLP (hidden 256,128; output action dim; ReLU).
# Latent dimension: 128.
# Optimizer: Adam, lr=1e-4, batch size 32.
# Training steps: 100k (early stop if action MSE on validation plateaus for 10k steps).

for episode (I_0, a_0, I_1, a_1, ..., I_T) in dataset:
    # Encode frames to latent space
    z_t = Encoder(I_t) for all t

    # Forward dynamics: predict next latent from current latent and action
    z_pred_{t+1} = F(z_t, a_t; θ_f)

    # Inverse dynamics: predict action from current and next latent
    a_pred_t = G(z_t, z_{t+1}; θ_g)

    # Decode latents back to frames (for video generation)
    I_decoded_t = Decoder(z_t; θ_d) for all t

    # Losses
    L_recon = Σ_t MSE(I_decoded_t, I_t)  # applied on raw pixel values normalized to [0,1]
    L_forward = Σ_t MSE(z_pred_{t+1}, z_{t+1})
    L_inverse = Σ_t MSE(a_pred_t, a_t)
    L = L_recon + λ_forward * L_forward + λ_inverse * L_inverse
    # Hyperparameters: λ_forward = 0.1, λ_inverse = 1.0 (chosen to balance scales)
    update θ_e, θ_f, θ_g, θ_d

# Closed-loop control at test time (single-step example)
I_curr ← camera_observation
z_curr = Encoder(I_curr)
# Option 1: action inference from desired next latent (e.g., goal image)
z_goal = Encoder(I_goal)
a = G(z_curr, z_goal; θ_g)
execute a
# Option 2: plan via forward dynamics over action candidates (sample 100 random actions, pick action minimizing ||F(z_curr,a) - z_goal||_2)
```

### (C) Why this design:
We chose to jointly train forward and inverse dynamics in a shared latent space rather than using separate modules for video generation and action extraction (as in PhysicsForcing) because the bidirectional consistency forces the latent to capture action-relevant information, enabling direct action inference without post-processing. We used MSE losses for simplicity and scalability, accepting that they may not capture perceptual similarity, but perceptual losses (e.g., LPIPS) could be added if computational budget allows. We encode frames into a compact latent space of dimension 128 rather than operating on pixels to reduce the computational cost of dynamics prediction and to abstract away irrelevant pixel noise, trading off fine-grained visual fidelity for faster rollouts. We applied a higher weight on L_inverse (λ_inverse=1.0) relative to L_forward (λ_forward=0.1) because action accuracy is critical for control, while forward prediction can tolerate slight errors as long as they are not compounding. We did not use a transformer-based diffusion (e.g., DiT) for the dynamics because the recurrent latent formulation is more efficient for iterative closed-loop simulation, though a diffusion prior could be added for stochasticity if needed.

**Load-bearing assumption:** The latent space learned from reconstruction and forward dynamics will naturally encode action-relevant information, enabling accurate inverse dynamics. This assumption is explicit: we rely on the joint training to force the latent to be action-informative, but it may fail if reconstruction dominates and the encoder discards subtle action-related visual features (as in β-VAE).

### (D) Why it measures what we claim:
The forward dynamics error (MSE between predicted z_{t+1} and encoded z_{t+1}) measures the model's ability to predict future latent states given actions, which is necessary for planning; however, this measure assumes that the latent space is Euclidean and that distance correlates with physical plausibility, which fails if the encoder produces non-smooth representations (e.g., sudden jumps from similar images). The inverse dynamics error (MSE between predicted action and true action) measures the controllability of the latent space, i.e., whether actions can be inferred from state transitions; this assumes that latent space is action-identifiable (different actions yield distinguishable latent transitions), which fails if the dynamics are non-injective (multiple actions lead to the same next latent) or if the latent collapses to a low-dimensional manifold. The reconstruction error (MSE between decoded frames and input frames) measures visual fidelity, but assumes pixel-space distance captures semantic correctness, which fails for tasks where small pixel shifts are critical for action inference (e.g., precise grasping). All three errors together ensure that the latent is simultaneously useful for video generation and control only when these assumptions hold; when they break, the model will fail either in visual quality or action accuracy, making it possible to detect misuse. We explicitly name the key assumption: the encoder outputs a smooth homomorphism where Euclidean distance in latent space correlates with physical similarity. This is verified by checking Lipschitz constant of encoder (empirical estimate: sample pairs of frames with small pixel perturbations and check latent distance ratio).

## Contribution

(1) JVALD, a generative model that jointly outputs video frames and actions through a shared latent dynamics, eliminating the need for separate action post-processing in robotic video generation pipelines. (2) The design principle of enforcing forward and inverse consistency in a latent space to learn a representation that is simultaneously predictive of future frames and informative for control, demonstrated through the coupling of reconstruction, forward, and inverse losses. (3) An architecture that integrates an encoder, recurrent latent dynamics, a decoder, and inverse dynamics into a single trainable framework for closed-loop video-based control.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | RoboNet (train/val/test split 80/10/10, 20 object categories, 10 tasks) | Diverse real-world manipulation tasks with varying action complexity |
| Primary metric | Action Prediction MSE (mean over test episodes, 3 seeds) | Directly measures inverse dynamics quality |
| Baseline 1 | PhysicsForcing (using its pretrained model on RoboNet, no action inference module added) | Lacks joint video-action latent training |
| Baseline 2 | Separate Forward+Inverse (same encoder/decoder, F and G trained independently with no shared loss) | No shared latent or bidirectional consistency |
| Ablation-of-ours | JVALD without L_inverse (set λ_inverse=0) | Tests importance of inverse dynamics loss |

Additional details: All models use same latent dim 128, optimizer Adam lr=1e-4, batch size 32, trained for 100k steps on single NVIDIA A100 (≈5 days). Evaluation on 1000 test episodes per seed. FVD (Fréchet Video Distance) computed on 256-frame samples from each method.

### Why this setup validates the claim
This combination tests the core claim that jointly training forward and inverse dynamics in a shared latent space improves action inference without sacrificing video quality. RoboNet provides diverse tasks with varying action complexity. Comparing against PhysicsForcing isolates the effect of joint training versus its separate physics module. Comparing against separate forward+inverse models tests the benefit of a shared latent space. The ablation without L_inverse determines whether the inverse loss is necessary for action-relevant latents. Action Prediction MSE is the direct metric for action inference, and observing lower error for JVALD on action-critical subsets (e.g., pick-and-place of small objects) would confirm the claim, while uniform error would falsify it. Additionally, we measure FVD to ensure video quality is not degraded. The setup includes 3 seeds to assess variance.

### Expected outcome and causal chain

**vs. PhysicsForcing** — On a case where precise action inference is needed (e.g., grasping a small object), PhysicsForcing generates visually plausible videos but its inverse dynamics (if any) are not jointly trained, so it often predicts wrong actions because the latent is not forced to encode action-relevant features. Our method instead jointly trains forward and inverse dynamics, ensuring the latent captures transitions that discriminate between actions. We expect a noticeable gap in Action MSE on fine-grained manipulation tasks (e.g., 0.05 vs 0.12 normalized MSE), but similar video quality (FVD within 10%) on simple motions due to shared reconstruction objective.

**vs. Separate Forward+Inverse** — On a case with long-horizon rollouts (e.g., 50 steps) where latent drift occurs, separate models produce increasingly inconsistent predictions (forward predicts one latent, inverse infers a different action from actual next latent), leading to high action error (e.g., MSE 0.08 at step 1 vs 0.25 at step 50). Our method’s shared latent space enforces consistency, so forward and inverse are aligned. We expect JVALD’s Action MSE to degrade more slowly with horizon length (e.g., 0.05 at step 1 vs 0.10 at step 50), while separate models’ error compounds faster.

**vs. Ablation (without L_inverse)** — On a case where action information is not trivially present in the latent (e.g., similar images with different actions—e.g., pushing left vs right with same start frame), the ablation loses action-relevant features because there is no inverse loss to enforce encoding them. Our full model retains those features. We expect the ablation to have significantly higher Action MSE (e.g., 0.15 vs 0.05), while video reconstruction MSE remains comparable (within 5%).

### What would falsify this idea
If JVALD’s Action MSE is not substantially lower (≥30%) than both baselines on the subset of tasks requiring precise, fine-grained actions (e.g., pick-and-place with small objects), then the central claim that joint training improves action inference fails. Additionally, if FVD of JVALD is worse than PhysicsForcing by more than 20%, the method’s advantage is negated.

## References

1. PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation
2. World Simulation with Video Foundation Models for Physical AI

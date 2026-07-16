# Bisimulation-Invariant Hierarchical Latent Planning for Embodied World Models

## Motivation

Existing embodied world models, such as Unified Embodied Synthesis (Xiaomi-Robotics-U0) and dual-system approaches like Galaxea's G0, assume that sequential autoregressive modeling or disjoint VLM+VLA steps are sufficient for long-horizon tasks. This reliance on step-by-step prediction or separate planning modules lacks hierarchical abstraction and compounding error guarantees, leading to inefficient planning and brittle execution. The root cause is the absence of a latent representation that serves as a sufficient statistic for future outcomes.

## Key Insight

Bisimulation invariance ensures that latent states with identical future reward distributions are mapped to the same representation, enabling parallel, non-autoregressive planning over compressed abstract plans without loss of optimality.

## Method

We propose **Bisimulation-Conditioned Hierarchical Planner (BCHP)**.

(A) **What it is**: BCHP is a hierarchical latent variable model that learns a bisimulation-invariant latent space for planning. Its inputs are sequences of states and rewards from embodied interactions, and its output is a latent plan (a sequence of latent variables) that can be decoded into actions non-autoregressively.

(B) **How it works**:

The model consists of:
- Encoder q_phi(z|s): 2-layer MLP, hidden=256, GeLU activation, output dim=64
- Latent transition model T_psi(z'|z,a): 3-layer MLP, hidden=512, ReLU activation
- Reward predictor R_rho(z,a): 2-layer MLP, hidden=128, output dim=1
- Action decoder pi_theta(a|z): 2-layer MLP, hidden=256, output dim=num_actions (softmax)
- Diffusion planner: 1D U-Net with 4 downsampling layers, 64 base channels, 1000 training steps, 50 DDIM sampling steps

Training procedure (pseudocode):
```python
# Hyperparameters: batch_size=64, latent_dim=64, eps=0.05, delta=0.1, 
# lambda_bisim=0.1, lambda_reward=1.0, lambda_trans=1.0, lambda_action=1.0
# Pairs: sample 128 random pairs from batch, enforcing i!=j and different episodes

for batch in dataloader:  # each batch: (s, a, r, s') from replay buffer
    z = encoder(s)           # current latent
    z_next = encoder(s')     # next latent
    
    # Bisimulation loss (assumption: thresholds correctly identify bisimilar pairs)
    # Verify on calibration set (512 transitions) that latent distance correlates with reward+KL distance (mean error < 0.1)
    mask = (abs(r_i - r_j) < eps) & (KL(q(z'|z_i,a_i) || q(z'|z_j,a_j)) < delta)
    loss_bisim = mean(mask * MSE(z_i, z_j))
    
    # Reconstruction losses
    loss_reward = MSE(R(z, a), r)
    loss_transition = MSE(T(z, a), z_next)
    loss_action = CrossEntropy(pi(z), a)
    
    total = 0.1*loss_bisim + 1.0*loss_reward + 1.0*loss_transition + 1.0*loss_action
    update model parameters via Adam (lr=3e-4)
```

Planning: Given current state s, encode to z_0. Use trained diffusion planner to sample latent plan (z_1,...,z_H) in 50 DDIM steps, conditioned on initial latent z_0. Decode actions a_t = argmax pi(z_t) for each step in parallel.

(C) **Why this design**: We chose a differentiable bisimulation loss over discrete clustering (e.g., VQ-VAE) because it allows gradient-based optimization and maintains smooth latent representations, accepting the cost that bisimulation may be approximate. We train the reward predictor and transition model jointly with the encoder rather than using separate offline models, because joint training aligns the latent space with both reward and dynamics, enabling bisimulation enforcement; the trade-off is increased computational cost. We use a diffusion model for planning instead of an autoregressive prior because diffusion samples the entire latent sequence in parallel, avoiding compounding errors and leveraging the bisimulation property that any latent sequence consistent with the transition model is valid; the cost is higher inference latency. We manually set thresholds eps and delta for bisimilar pair selection to maintain interpretability, at the risk of missing some pairs.

(D) **Why it measures what we claim**: The computational quantity `loss_bisim = MSE(z_i, z_j)` measures bisimulation invariance under the assumption that thresholds correctly identify all bisimilar pairs; failure occurs when thresholds misclassify or encoder is insufficiently expressive, in which case the loss measures proxy similarity rather than true bisimulation. The quantity `loss_reward` measures reward predictability from latent state and action, a necessary condition for bisimulation; the assumption is that R_rho is well-specified for the true reward. This assumption fails under non-stationary rewards, making loss_reward measure approximation error. The quantity `loss_transition` measures latent dynamics consistency; the assumption is that the latent space is Markovian. This assumption fails when the latent space loses temporal information, making loss_transition measure temporal inconsistency instead of dynamics accuracy.

## Contribution

(1) A hierarchical latent variable model for embodied world models that enforces bisimulation invariance via a differentiable loss, enabling non-autoregressive planning. (2) The introduction of a diffusion-based latent plan sampler that exploits the bisimulation property to generate coherent long-horizon plans in parallel. (3) A demonstration that the bisimulation constraint can be integrated into an end-to-end trainable framework without requiring separate alignment steps.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | MetaWorld ML45 | Diverse long-horizon manipulation tasks |
| Primary metric | Average success rate | Direct measure of task completion |
| Baseline 1 | HiRL | Standard hierarchical RL without bisimulation |
| Baseline 2 | DreamerV3 | Model-based RL without explicit bisimulation |
| Baseline 3 | VQ-VAE planner | Discrete latent planner contrasting smooth bisimulation |
| Ablation-of-ours | BCHP w/o bisimulation loss | Isolates effect of bisimulation enforcement |

Additionally, we compare with a contrastive invariance objective (InfoNCE, τ=0.07) as a robustness check.

### Why this setup validates the claim

This experimental design tests whether enforcing bisimulation invariance in the latent space improves planning generalization. The MetaWorld ML45 benchmark provides 45 tasks that share underlying dynamics and reward structures, allowing bisimulation to align representations of equivalent states across tasks. The baselines cover alternatives: HiRL as a hierarchical method without latent bisimulation, DreamerV3 as a model-based approach that learns latent dynamics but not bisimulation, and VQ-VAE planner using discrete clustering which may lose smoothness. The ablation directly isolates the contribution of the bisimulation loss. The primary metric, success rate, directly measures whether each method can achieve task goals, providing a clear falsifiable test: if our method succeeds significantly more than baselines, the claim holds; if not, it is refuted. We also perform sensitivity analysis on eps and delta (grid search: eps∈[0.01,0.05,0.1,0.2], delta∈[0.05,0.1,0.2]) on a subset of 5 tasks to verify robustness. Resource estimates: each run uses ~48 hours on a single NVIDIA V100 (16GB) with batch size 64, totaling 5M environment steps.

### Expected outcome and causal chain

**vs. HiRL** — On a task where two subtasks have similar reward but different state appearances, HiRL overfits to visual features, causing policy degradation when reused. Our method maps both subtasks to the same latent via bisimulation, enabling zero-shot transfer. We expect a noticeable success gap (e.g., 20%) on tasks with perceptual variation but identical rewards.

**vs. DreamerV3** — On a task with irrelevant state details (e.g., object color), DreamerV3’s latent space retains these, leading to reward prediction errors when color changes. Our bisimulation loss collapses such irrelevant variations, making reward prediction robust. We expect similar performance on simple tasks but a 15% gap on tasks with distractor variables.

**vs. VQ-VAE planner** — On a continuous control task requiring fine-grained action sequences, VQ-VAE’s discrete latents cause quantization errors, leading to jerky motions and failure. Our smooth bisimulation latent allows continuous planning via diffusion, yielding smoother trajectories. We expect a 25% success gap on tasks requiring precision (e.g., assembly).

**vs. contrastive ablation** — The contrastive variant may align latents based on similarity but lacks bisimulation's explicit reward-link, leading to weaker transfer when rewards differ slightly; we expect a 10% gap in favor of bisimulation.

### What would falsify this idea

If the performance gain of BCHP over its ablation (without bisimulation loss) is uniform across all task subsets rather than concentrated on tasks where state aliasing or reward-equivalence is predicted to matter, then the central claim that bisimulation invariance drives improvement is false.

## References

1. Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model
2. Galaxea Open-World Dataset and G0 Dual-System VLA Model
3. Qwen3-VL Technical Report
4. Emu3.5: Native Multimodal Models are World Learners
5. TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability
6. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding
7. VTimeLLM: Empower LLM to Grasp Video Moments
8. LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models

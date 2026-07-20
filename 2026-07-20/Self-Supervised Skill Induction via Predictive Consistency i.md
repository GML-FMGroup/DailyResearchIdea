# Self-Supervised Skill Induction via Predictive Consistency in Reward-Free Exploration

## Motivation

Existing methods like RESOURCE2SKILL distill skills from human-created resources, but they cannot bootstrap from scratch; the agent must have access to curated tutorials, code, or demonstrations. Without these resources, skill induction remains arbitrary because there is no automatic criterion to distinguish reusable patterns from noise. We identify that predictive consistency under a learned world model provides a self-supervised signal: trajectories the model can predict accurately are precisely those that are deterministic and repeatable, making them natural candidates for skill extraction.

## Key Insight

Predictive consistency of a world model serves as a self-supervised success criterion because low prediction error implies that the trajectory dynamics are deterministic and reproducible, which is a necessary condition for a skill to be reliably executed.

## Method

(A) **What it is**: PRED-SKILL is a framework that discovers skills from scratch by (1) exploring the environment with random actions, (2) learning a world model, (3) filtering trajectories with high predictive consistency, and (4) distilling them into a latent skill space via a variational autoencoder (VAE). Inputs: raw observations from the environment. Outputs: a continuously-parameterized skill library (latent codes z) that can generate actions given a state.

(B) **How it works**: Pseudocode
```python
# Hyperparameters
THRESHOLD_PERCENTILE = 20  # percentile of prediction errors used as filter threshold
BETA = 0.1                # weight on KL divergence in VAE loss
LATENT_DIM = 16           # dimension of skill code z
K = 1000                  # steps between world model updates

# Initialize
world_model = DynamicsPredictor()  # neural network: (s, a) -> s'
skill_encoder = Encoder()         # maps trajectory tau to z
skill_decoder = Decoder()         # maps (z, current_state) to action
buffer = ReplayBuffer()

# Phase 1: Exploration and world model training
for episode in range(N_EPISODES):
    s = env.reset()
    trajectory = []
    for t in range(T):
        a = random_action()
        s_next, _, done, _ = env.step(a)
        trajectory.append((s, a, s_next))
        buffer.add((s, a, s_next))
        if len(buffer) % K == 0:
            sample batch from buffer
            update world_model via MSE(s_next - world_model(s,a))
        if done: break
    store entire trajectory in buffer_trajectories

# Phase 2: Consistent trajectory filtering
errors = []
for tau in buffer_trajectories:
    err = mean([||world_model(s,a) - s_next|| for each transition])
    errors.append(err)
threshold = percentile(errors, THRESHOLD_PERCENTILE)
filtered_trajectories = [tau for tau in buffer_trajectories if mean_error(tau) < threshold]

# Phase 3: VAE skill abstraction on filtered trajectories
for tau in filtered_trajectories:
    encoder_input = flatten(tau)   # e.g., concatenated (s,a,s') vectors
    z = skill_encoder(encoder_input)  # outputs mu and logvar
    reconstruction = skill_decoder(z, initial_state)  # predict actions over the trajectory
    recon_loss = MSE(actions_true, actions_pred)
    kl_loss = KL_div(z || N(0,I))
    loss = recon_loss + BETA * kl_loss
    update skill_encoder and skill_decoder

# At inference: sample z ~ N(0,I) and use skill_decoder to produce actions from current state.
```

(C) **Why this design**: We chose predictive consistency (phase 2) over intrinsic reward (e.g., novelty) because consistency directly captures whether a pattern is reliably reproducible—a necessary condition for skill transfer; novelty may highlight one-off behaviors. The trade-off is that stochastic but useful trajectories are discarded; we accept this because such trajectories cannot form reliable skills. We used a VAE over hard clustering (e.g., k-means) because the latent space permits interpolation between skills and composition, enabling generalization; the cost is that training is more complex and requires tuning β. We set β=0.1 (vs. β=1) to avoid forcing a factorized prior that would collapse skill diversity; the trade-off is that the latent space may not be perfectly isotropic but still allows smooth arithmetic. We used a percentile threshold rather than a fixed value to adapt as the world model improves; the trade-off is that early in training, the threshold may be too strict or lenient, but we accept this because the threshold converges as the model stabilizes.

(D) **Why it measures what we claim**: The predictive consistency score C(τ) measures "skill-worthiness" because we assume that low prediction error indicates deterministic, repeatable dynamics; this assumption fails when the world model is undertrained (epistemic uncertainty), in which case C reflects model capacity rather than trajectory regularity. The VAE reconstruction loss measures "skill abstraction" because we assume that a latent code z decompressing to the trajectory captures the essential decision pattern; this assumption fails when trajectories are long and high-dimensional, causing the decoder to memorize low-level details rather than abstract skill structure, in which case the loss reflects compression efficiency. The KL divergence term measures "skill prior coverage" because we assume a standard normal prior encourages a smooth manifold; this assumption fails when true skill distribution is multi-modal, in which case the KL term forces unrealistic unimodal coverage.

## Contribution

(1) A self-supervised skill induction framework (PRED-SKILL) that uses predictive consistency of a learned world model as an automatic success criterion for trajectory filtering, eliminating dependence on human-created resources. (2) The identification that predictive consistency serves as a training signal correlating with reproducibility, enabling autonomous skill discovery without explicit reward. (3) An analysis of the operationalization assumptions linking prediction error to skill-worthiness and VAE reconstruction to skill abstraction, clarifying the method's scope.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Meta-World (visual + proprioceptive) | Multimodal, skill-rich benchmark |
| Primary metric | Task success rate on unseen variants | Measures skill generalization |
| Baseline 1 | Random action baseline | Lower bound with no skill |
| Baseline 2 | Curiosity-driven exploration (RND) | Explores via novelty, not consistency |
| Baseline 3 | K-means trajectory clustering | Hard assignment baseline |
| Ablation | Ours w/o consistency filter | Tests necessity of filtering |

### Why this setup validates the claim

This combination directly tests the central claim that predictive consistency identifies reusable skills. The random baseline establishes a floor performance. Curiosity-driven exploration (RND) tests whether novelty-based intrinsic rewards can yield comparable skill discovery; if our method outperforms RND, it supports consistency over novelty. K-means trajectory clustering tests whether hard assignment suffices for skill abstraction; superior performance from our VAE would justify the continuous latent space. The ablation removes the consistency filter to isolate its contribution. Success rate on unseen task variants is the right metric because it captures skill transfer, which is the ultimate goal. Any observed advantage of our method must be attributable to the filtering or VAE mechanisms, and the baselines partition the failure modes.

### Expected outcome and causal chain

**vs. Random action baseline** — On a task requiring sequenced coordination (e.g., opening a drawer with a specific sequence), random actions rarely succeed because they lack structure. Our method discovers the drawer-opening skill from consistent trajectories, enabling reliable reproduction. We expect our success rate to be ~60% while random remains below 5%.

**vs. Curiosity-driven exploration (RND)** — Curiosity often fixates on novel but non-reproducible events (e.g., a one-time object collision), wasting exploration. On a task with high consistency but low novelty (e.g., always picking up the same cup), our consistency filter captures the skill while curiosity is distracted. We expect our method to achieve ~60% success vs. ~25% for RND.

**vs. K-means trajectory clustering** — K-means assigns each trajectory to a discrete cluster, failing on tasks requiring interpolation (e.g., reaching targets at varying positions). Our VAE latent space smoothly connects similar skills, enabling generalization to unseen targets. We expect our method to achieve ~70% success vs. ~40% for K-means.

**Ablation: Ours w/o consistency filter** — Without filtering, noisy and unrepeatable trajectories contaminate the VAE training, degrading skill quality. On a task where consistency is high (e.g., precise peg insertion), the filter removes distracting failures. We expect our full method to outperform the ablation by ~10 percentage points (e.g., 60% vs. 50%).

### What would falsify this idea

If our method's gain over baselines is uniform across all trajectory consistency levels (i.e., not concentrated on subsets with high predictive consistency), then the fundamental assumption that consistency indicates skill-worthiness is unsupported, and some other component (e.g., VAE structure) may be responsible.

## References

1. RESOURCE2SKILL: Distilling Executable Agent Skills from Human-Created Multimodal Resources

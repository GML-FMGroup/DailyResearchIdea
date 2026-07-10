# Differentiable Kinematic Consistency Loss for Causal World Models

## Motivation

The diagnostic iKCE (from 'Imagined Rollouts are Kinematic, Not Dynamic') reveals that world models like DreamerV3 produce latent rollouts that are action-independent kinematic extrapolations, particularly under dynamics shift. However, iKCE is purely diagnostic and does not modify training. The root cause is that world model training minimizes reconstruction error only, neglecting dynamic faithfulness. This structural failure limits generalization to changing dynamics. We need a differentiable loss that directly penalizes kinematic shortcuts during training.

## Key Insight

The squared deviation between a world model's predicted latent state and a kinematic null (constant velocity extrapolation) is a differentiable function of model parameters, enabling gradient descent to enforce action-conditional causality.

## Method

We propose **DKC (Differentiable Kinematic Consistency)** loss integrated into world model training.

### (A) What it is
DKC is an auxiliary loss term added to the world model objective that penalizes latent state transitions that are close to a kinematic extrapolation, thereby encouraging the model to rely on actions rather than temporal inertia.

### (B) How it works
```python
def dkc_loss(world_model, batch, policy=None, λ=0.1, null_mode='constant_velocity', 
            t_warmup=2, t_max=25, action_norm_thresh=0.05):
    """Compute differentiable iKCE loss on imagined rollouts."""
    # Step 1: Generate imagination rollout using world model and policy
    # For simplicity, we use the batch's initial states and a fixed random action sequence
    actions = sample_random_actions(batch_size=len(batch), horizon=50)  # H=50
    latent_states = []
    s = world_model.encode_initial(batch['observations'][:, 0])  # [B, D]
    for t in range(50):
        a = actions[:, t]
        s_next = world_model.transition(s, a)  # RSSM or diffusion denoised latent
        latent_states.append(s_next)
        s = s_next
    latent_states = torch.stack(latent_states, dim=1)  # [B, 50, D]

    # Step 2: Compute kinematic null predictions
    if null_mode == 'constant_velocity':
        # Null: s_null[t] = s[t-1] + (s[t-1] - s[t-2]) for t>=2
        # Compute over timesteps t in [t_warmup, t_max)
        velocities = latent_states[:, 1:t_max-1] - latent_states[:, :t_max-2]  # [B, T-2, D]
        s_null = latent_states[:, 1:t_max-1] + velocities  # [B, T-2, D]
        s_actual = latent_states[:, 2:t_max]  # [B, T-2, D]
        # Downweight if action magnitude is small (environment may be truly kinematic)
        action_norms = torch.norm(actions[:, :t_max-1], dim=-1)[:, 1:]  # [B, T-2]
        weight = (action_norms >= action_norm_thresh).float()  # 0 or 1
        loss = (weight * (s_actual - s_null).pow(2)).mean()
    else:
        raise ValueError(f"Unknown null_mode {null_mode}")

    # Step 3: Combine with original world model loss
    return λ * loss
```

### (C) Why this design
We chose **constant-velocity null** over constant-acceleration because it penalizes only the simplest kinematic shortcut; a higher-order null would risk penalizing legitimate dynamic accelerations, increasing false positives. The trade-off is that constant-velocity may fail to catch higher-order kinematic patterns (e.g., jerk), but those are rarer in practice. We compute iKCE on **imagined rollouts** with random actions rather than on replay data because the diagnostic is strongest when the action sequence is varied (actions that should cause deviation), and random actions maximize coverage of action-conditional responses; the cost is that random actions may not match the policy distribution, but the loss only needs the model to respond to actions in general, not to a specific policy. We use **L2 loss** instead of Huber to keep gradients smooth for all deviation magnitudes, accepting that outliers (e.g., initial transient) might dominate the loss; we mitigate by only applying loss after a warm-up timestep (t>=2) and averaging over a range (t=2..25). The hyperparameter λ=0.1 is chosen as a small weight to avoid disrupting reconstruction convergence; a larger λ could enforce causality too aggressively, harming representation learning. The action norm threshold ε=0.05 downweights loss when actions are negligible, preventing false positives in near-kinematic environments.

### (D) Why it measures what we claim
**Computational quantity Q = MSE(s_hat_t, s_null_t)** where s_null_t = s_{t-1} + (s_{t-1} - s_{t-2}) measures **kinematic faithfulness** because the kinematic null acts as a counterfactual of 'what would happen if the action had no effect on this transition'; this proxy equates to kinematicity under the assumption that the true dynamics are **not** exactly constant-velocity for the relevant timesteps. This assumption fails when the environment is truly constant-velocity (e.g., a coasting ball), in which case low Q reflects correct dynamics, not a shortcut. To address this, we bypass the loss when the action magnitude is small (actions with norm < ε=0.05) via a binary weight, and we validate this threshold in ablations. The **choice of random actions** ensures that the actions cover diverse directions, so that the constant-velocity null is unlikely to coincide with the true dynamics except by chance; the assumption is that random actions are uniformly distributed and unlikely to produce exactly the same latent state as the kinematic null. This fails if the latent space is low-dimensional or the world model is stuck, but we mitigate by using full trajectory loss (average over multiple timesteps) and validating with iKCE on held-out actions. We explicitly assume: true dynamics are rarely exactly constant-velocity over consecutive timesteps; when they are, the loss is downweighted by the action magnitude threshold.

## Contribution

(1) A differentiable kinematic consistency loss (DKC) that directly penalizes action-independent latent transitions during world model training, derived from the iKCE diagnostic. (2) The design principle that dynamic faithfulness can be optimized via a contrast between predicted and kinematic-null latents, with explicit assumptions and failure modes. (3) Integration of DKC into a standard world model training loop (e.g., DreamerV3) to improve generalization to dynamics shifts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------|
| Primary dataset | DMC walker-walk | Exhibits kinematic shortcut failure |
| Secondary dataset | Atari Pong | Tests generalization to high-dimensional visuals |
| Primary metric | iKCE + task reward | Measures kinematicity & performance |
| Baseline | DreamerV3 (no DKC) | Standard world model baseline |
| Baseline | DreamerV3 + L2 latent reg | Regularization baseline |
| Baseline | IRIS (transformer WM) | Alternative architecture baseline |
| Baseline | DreamerV3 + action-conditional MI penalty | Alternative causal consistency measure (InfoNCE) |
| Ablation | DKC with replay actions | Tests necessity of random actions |
| Ablation | DKC with varying kinematic null threshold ε | Tests sensitivity to action magnitude downweighting |

### Why this setup validates the claim

On DMC walker-walk, world models often produce kinematic rollouts that ignore action conditioning, leading to long-horizon prediction failure and poor policy performance. By comparing iKCE between imagined and real rollouts, and task reward, we directly measure whether DKC reduces the kinematic shortcut. The baselines isolate key challenges: DreamerV3 (no DKC) shows the default failure; DreamerV3 + L2 latent reg tests if generic regularization suffices; IRIS tests a different architecture's propensity for the shortcut. The additional baseline with action-conditional mutual information (MI) penalty — implemented as an InfoNCE loss that encourages latent transitions to be informative of the action — isolates the benefit of the kinematic null specifically. The ablation with replay actions verifies that random actions are crucial for DKC's diagnostic signal. The ablation varying threshold ε validates the assumption that downweighting near-kinematic regimes prevents false positives. Each experiment runs on a single NVIDIA A100 GPU for approximately 5 days total (2 days for DMC, 3 days for Atari). Computational details: world model has ~10M parameters, rollouts of horizon 50, training for 10M environment steps.

### Expected outcome and causal chain

**vs. DreamerV3 (no DKC)** — On a case where the walker needs to change direction, the baseline imagines smooth kinematic continuation because its latent dynamics ignore action conditioning, leading to high iKCE and eventual policy failure. Our method penalizes such kinematic transitions, forcing the model to rely on actions, so we expect iKCE drop by ~30% and reward increase by ~20% on DMC walker-walk, and similar improvements on Atari Pong.

**vs. DreamerV3 + L2 latent reg** — On the same case, L2 reg indiscriminately penalizes all latent changes, reducing dynamics richness and causing blurry rollouts. DKC specifically penalizes kinematic patterns, preserving true dynamics, so we expect L2 reg to hurt both iKCE and reward (unfavorable comparison).

**vs. IRIS** — On a case involving high-frequency actions, IRIS's discrete latent may be less prone to kinematic smoothing due to quantization, but still suffers from ignoring action causality in some regimes. DKC's continuous latent+kinematic penalty should be more targeted, so we expect comparable iKCE but better reward due to finer action conditioning.

**vs. DreamerV3 + action-conditional MI penalty** — On a case with redundant actions (e.g., no effect on some dimensions), the MI penalty may still encourage action-conditioning but does not specifically target kinematic shortcuts. DKC directly penalizes the null, so we expect lower iKCE with DKC, and comparable reward if MI also improves dynamics.

### What would falsify this idea
If DKC fails to reduce iKCE on walker-walk or Atari Pong (i.e., iKCE similar to or higher than baseline without DKC) or if reward is worse despite lower iKCE, then the central claim is wrong. Specifically, if improvement is observed equally across all subsets (e.g., also on non-kinematic environments) instead of concentrated on kinematic cases, the causal mechanism would be invalid.

## References

1. Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure
2. Mastering Diverse Domains through World Models
3. Transformers are Sample Efficient World Models
4. Deep Hierarchical Planning from Pixels

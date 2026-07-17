# LATENT-ACT: Jointly Training World Models and Policies via Closed-Loop Prediction Consistency

## Motivation

The survey 'From Pixels to States' categorizes interactive world models but reveals that none jointly optimize perception, prediction, and action under a common signal that measures interaction fidelity. Existing methods train the dynamics predictor offline on fixed data, causing distribution shift when the policy explores, and the encoder is not incentivized to produce states that are both predictable and controllable. This structural separation leads to compounding errors in closed-loop interaction.

## Key Insight

Jointly minimizing the latent prediction error on the agent's online rollouts enforces mutual consistency: the encoder must produce states that are accurately predictable by the dynamics given the policy's actions, and the policy must select actions that keep the system in predictable regions, aligning all components with interaction fidelity.

## Method

### LATENT-ACT: Jointly Training World Models and Policies via Closed-Loop Prediction Consistency

**A) What it is**: LATENT-ACT is a unified closed-loop world model that takes raw observations and actions as inputs and outputs a policy action and a latent state prediction. It consists of three differentiable components—latent encoder f_enc, dynamics predictor f_dyn, and policy π—trained end-to-end via a single PPO objective that combines task reward with a latent prediction consistency reward.

**B) How it works** (pseudocode):
```python
# Hyperparameters: lambda_tradeoff=0.5, rollout_len=10, gamma=0.99, PPO_clip=0.2, latent_dim=64
# Encoder: f_enc: 4-layer CNN (channels 32,64,64,64; kernel sizes 8,4,3,3; strides 4,2,1,1) + 2-layer MLP (hidden 512) -> latent z
# Dynamics: f_dyn: 3-layer MLP (hidden 256, ReLU) -> z_pred (latent_dim)
# Policy: pi: 2-layer MLP (hidden 256, tanh) -> Gaussian mean, fixed std=0.1
# Value: V: 2-layer MLP (hidden 256) -> scalar

Initialize networks randomly.
for episode in range(N):
    obs = env.reset()
    buffer = []
    for t in range(T):
        z = f_enc(obs)
        a = pi(z)
        obs_next, env_reward, done = env.step(a)
        z_next_true = f_enc(obs_next)                # target latent
        z_next_pred = f_dyn(z, a)                    # predicted latent
        pred_reward = -||z_next_pred - z_next_true||^2  # consistency reward (dimensionless)
        total_reward = lambda_tradeoff * env_reward + (1 - lambda_tradeoff) * pred_reward
        buffer.append((z, a, total_reward, z_next_true, z_next_pred, done))
        obs = obs_next
        if done: break
    # Compute advantages via GAE (lambda_gae=0.95) using V(z) from buffer
    # Update all networks jointly with PPO objective (K=4 epochs, mini-batch size 256, entropy coefficient 0.01)
    # Use Adam optimizer with learning rate 3e-4
```

**C) Why this design**: We chose to train all components jointly with a shared PPO objective rather than alternating between model learning and policy learning as in Dreamer. This ensures the encoder adapts to the policy's latent space needs, preventing exploitation of stale representations. However, joint training increases instability, which we mitigate with a single critic that evaluates both task and prediction rewards. We combine task reward and prediction error linearly with λ=0.5 as default to balance performance and accuracy; a learned weighting could be used but adds complexity. We use one-step prediction error instead of multi-step to avoid long-horizon credit assignment, trading long-term consistency for simpler optimization. Unlike Dreamer, which trains the world model via MSE on observations independently, our prediction error directly on latent states ties all modules to a common objective, reducing distribution shift during policy rollouts.

**D) Why it measures what we claim**: The quantity ||z_pred - z_true||^2 measures interaction fidelity under the assumption that z is a sufficient statistic of the observation for both task and dynamics. If the encoder discards task-relevant information (e.g., due to limited capacity), prediction error may reflect representation quality rather than dynamics error. The total reward r_total combines task reward with prediction error; the task reward prevents the policy from trivially minimizing error by staying in low-variability areas. The encoder's training objective ensures that latent states are both predictable (by minimizing prediction error) and useful for task (via policy gradients), operationalizing the concept of mutual consistency. The trade-off λ controls the relative importance; the assumption of linear combination holds only when the two objectives are aligned, which may fail in stochastic environments where optimal actions are inherently unpredictable. We test the sufficiency assumption by varying latent dimension (32, 64, 128) and by including an observation reconstruction loss as an ablation (see experiment).

## Contribution

(1) A unified architecture, LATENT-ACT, that jointly trains a latent state encoder, dynamics predictor, and action policy under a shared reward signal combining task reward and prediction consistency. (2) Demonstration that online prediction error minimization acts as a regularization that aligns the learned latent space with the policy's actions, reducing distribution shift compared to offline model learning. (3) An analysis of the trade-off between task reward and prediction fidelity in closed-loop world models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Atari 2600 games (Pong, Breakout) | Popular benchmark for world models. |
| Primary metric | Average Return | Directly measures task performance. |
| Baseline 1 | Dreamer | World model trained with observation reconstruction. |
| Baseline 2 | TD-MPC | Model-based RL with latent planning. |
| Baseline 3 | PPO (model-free) | No world model; tests modeling necessity. |
| Ablation-of-ours | LATENT-ACT w/o pred. consistency | Removes consistency reward; tests its benefit. |

### Why this setup validates the claim

Using Atari as a standard video-based RL benchmark allows direct comparison to established world models like Dreamer and TD-MPC, which separate model and policy learning. Joint training of encoder, dynamics, and policy under a single PPO objective should yield better latent consistency and task performance. Including a model-free PPO baseline tests whether any world model improves over direct pixel-to-policy mapping. The ablation of the prediction consistency reward isolates whether this term is responsible for gains. Average Return is the ultimate measure of closed-loop task success; if the method's central claim holds, it should achieve higher returns than baselines, with the ablation performing worse. This design creates a falsifiable test: if our method does not outperform Dreamer on tasks requiring long-horizon consistency (e.g., Montezuma's Revenge), the claim is refuted.

### Expected outcome and causal chain

**vs. Dreamer** — On a situation where the learned world model's latent states are inaccurate due to distribution shift from policy rollouts (e.g., after a novel action), Dreamer's separate training fails to adapt the encoder, leading to compounding errors in its latent and reduced task performance. Our method jointly trains encoder with policy, so the encoder sees policy-induced states and maintains consistency, leading to higher average return on Atari games, especially in later steps of episodes.

**vs. TD-MPC** — On a case requiring long-horizon planning under noisy observations, TD-MPC's dual encoder for planning and policy can cause inconsistency between planning latent and policy latent, degrading performance. Our unified latent space ensures the same representation is used for both planning (value) and policy, reducing mismatch, so we expect our method to achieve higher returns on tasks with high observation noise or stochastic transitions.

**vs. PPO** — On a task where video frames are redundant but dynamics are complex, PPO without a world model may require many samples to learn from raw pixels due to lack of temporal abstraction. Our method's latent dynamics provide a compact representation that accelerates learning, so we expect faster convergence and higher asymptotic return.

**vs. LATENT-ACT w/o pred. consistency** — On a scenario where the policy must maintain accurate internal predictions to avoid dead-ends (e.g., in Pitfall), the ablation that lacks consistency reward will have less pressure to keep latent states predictable, leading to higher variance and occasional catastrophic failures, while full method maintains stable performance.

### What would falsify this idea

If LATENT-ACT does not outperform Dreamer on Atari games, or if the performance gain is uniform across all games rather than concentrated on those requiring long-horizon consistency (e.g., Montezuma's Revenge), then the central claim of joint training improving closed-loop performance is false.

## References

1. From Pixels to States: Rethinking Interactive World Models as Game Engines

# Advantage-Weighted World Model Updates for Closed-Loop Policy Refinement

## Motivation

Existing world model-based policy refinement, such as World4RL, assumes a static pre-trained diffusion world model and does not correct errors that arise from distribution shift during policy rollouts. This leads to error accumulation, especially when the policy explores regions poorly covered by the training data. The fundamental problem is the absence of a feedback loop that adjusts the world model based on policy behavior, resulting in growing simulation inaccuracies that degrade policy performance.

## Key Insight

Policy advantage estimates provide a direct signal indicating which world model transitions are most critical to correct, because high-advantage transitions correspond to states where model inaccuracies have the largest impact on policy improvement.

## Method

### (A) What it is
We propose **Advantage-Weighted World Model Update (AWMU)**, a closed-loop framework that refines a pre-trained diffusion world model (DDPM with U-Net backbone, 1M parameters, 100 diffusion steps) by selectively updating it on transitions where the policy's advantage estimate is high. The input is the pre-trained diffusion world model and an RL policy (2-layer MLP, hidden=256, tanh activation, plus value network of same size); the output is an updated world model that simulates more accurate dynamics for high-impact transitions.

### (B) How it works (pseudocode)
```python
def awmu_training_loop(initial_world_model, policy, num_iterations=500):
    world_model = initial_world_model  # pretrained on 5M transitions from Robomimic dataset
    replay_buffer = []  # capacity 1e6 transitions
    
    for iteration in range(num_iterations):
        # 1. Rollout policy in current world model for T=1000 steps
        trajectory = []
        obs = world_model.initial_observation()
        for t in range(1000):
            action = policy(obs)
            next_obs = world_model.step(obs, action)  # diffusion forward pass with 100 denoising steps
            reward = compute_reward(obs, action, next_obs)  # known reward function
            done = is_terminal(next_obs)
            trajectory.append((obs, action, next_obs, reward, done))
            obs = next_obs
            if done: break
        
        # 2. Compute advantages via GAE (gamma=0.99, lambda=0.95)
        advantages = compute_gae(trajectory, policy.value_function, gamma=0.99, lambda_=0.95)
        
        # 3. Weight each transition by absolute advantage: w_i = |adv_i| / sum(|adv_i|)
        weights = [abs(adv) for adv in advantages]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # 4. Sample a minibatch of transitions proportional to weights (batch_size=32)
        sampled_indices = np.random.choice(len(trajectory), size=32, p=weights)
        sampled_transitions = [trajectory[i] for i in sampled_indices]
        
        # 5. Update diffusion world model on sampled transitions (denoising objective: MSE on noise prediction)
        loss = diffusion_loss(world_model, sampled_transitions)  # standard DDPM loss
        optimizer.step(loss)  # Adam, lr=3e-4
        
        # 6. Append trajectory to replay_buffer
        replay_buffer.extend(trajectory)
        if len(replay_buffer) > 1e6:
            replay_buffer = replay_buffer[-1e6:]
        
        # 7. Update policy via PPO (clip epsilon=0.2, value coefficient=0.5, entropy coefficient=0.01, mini-batch size=64, 10 epochs)
        if iteration % 10 == 0:
            policy = ppo_update(policy, world_model, replay_buffer)
```

### (C) Why this design
We chose **advantage-weighted sampling** over uniform sampling for world model updates because advantages directly indicate which transitions are most consequential for policy improvement; high absolute advantage means the model's prediction error could lead to a poor policy update. The trade-off is that low-advantage transitions, which may be rare but still critical for safety, are updated less frequently, but we accept that since they do not strongly affect expected returns. We use a **diffusion world model** (DDPM with U-Net, 1M params, 100 steps) rather than a deterministic model because diffusion captures multimodal future distributions, providing richer signal for where errors occur. The cost is higher inference time per step (approx. 0.1s on a V100). We update the world model **every rollout** (rather than after a batch of rollouts) to react quickly to policy distribution shift, at the expense of gradient variance; we mitigate this by using Adam optimizer (lr=3e-4, beta1=0.9, beta2=0.999). This specific combination of advantage-weighting, diffusion, and frequent updates is not a simple composition of existing ideas; unlike World4RL (which never updates the world model) or typical DAgger (which uses expert corrections), our update signal comes purely from the policy's own advantage values, creating a self-correcting loop without external supervision.

### (D) Why it measures what we claim
The computational quantity **advantage magnitude** |adv_i| measures the **model error importance** under the assumption that the value function is accurate and advantages are primarily due to model prediction errors. This assumption fails when policy stochasticity or value function bias dominates the advantage, causing high advantage even when the model is accurate. In that case, our method may incorrectly focus on transitions where updating the model does not reduce error. To verify the assumption's validity, we compute the Spearman correlation between advantage magnitude and prediction error (MSE between predicted and true next observation) on a held-out set of 1024 transitions from the offline dataset; we expect rho > 0.3 to justify the weighting. If the correlation is below this threshold, the assumption is likely violated and the weighting scheme should be revised. This calibration step ensures that our method is only applied when the linking assumption holds.

## Contribution

(1) A closed-loop world model update framework that uses policy advantage estimates to weight transitions for model correction, enabling self-correcting policy refinement without real-world data. (2) Demonstration that advantage-weighted sampling reduces world model error accumulation compared to uniform or no updates, verified by improved policy transfer to real environments. (3) Design principle that the advantage signal can serve as an importance measure for model updates, bridging model-based RL and online adaptation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Robomimic dataset for tasks: can, lift, door (5M transitions) plus a long-horizon assembly task from Robosuite: two-arm peg insertion (50-step average) | Realistic offline data for pretraining; long-horizon tests error accumulation. |
| Primary metric | Task success rate averaged over 100 evaluation episodes per seed | Direct measure of policy performance. |
| Baseline 1 | World4RL (no world model update) | Isolate effect of updating world model. |
| Baseline 2 | Uniform weight update | Isolate effect of advantage weighting. |
| Baseline 3 | Prediction error weighted update (weight = MSE between predicted and true next observation from pre-trained ensemble) | Test if advantage is uniquely beneficial vs. model uncertainty. |
| Ablation | AWMU with uniform sampling | Controls for update distribution. |

### Why this setup validates the claim
This setup tests whether advantage-weighted world model updates improve policy performance by targeting high-impact transitions. The Robomimic dataset (can, lift, door) provides a realistic, diverse set of transitions for pretraining the diffusion world model, ensuring that any performance gains come from the update mechanism rather than data quality. The long-horizon assembly task (two-arm peg insertion) is chosen because it requires precise multi-step control where model errors compound, making the advantage-weighting especially beneficial. Comparing against World4RL (no update) isolates the benefit of updating the world model at all, while the uniform weight baseline and prediction error baseline directly test whether advantage weighting outperforms random or uncertainty-based sampling. Task success rate is the definitive metric for robotics tasks, capturing the ultimate goal. Together, these choices create a controlled environment where the central claim—that focusing updates on high-advantage transitions yields better world model accuracy and downstream policy—can be falsified: if our method does not outperform all baselines on success rate, the claim is unsupported.

### Expected outcome and causal chain

**vs. World4RL** — On a case where the policy encounters a novel state not well captured by the pretrained world model, World4RL cannot correct its predictions, causing the policy to hallucinate unrealistic transitions and learn poor behavior. Our method detects high-advantage transitions (where prediction error would mislead policy updates) and refines the model on those, so the model becomes locally accurate. We expect a noticeable gap in success rate on tasks requiring precise control (e.g., peg insertion), but parity on simple reaching tasks where model errors are negligible.

**vs. Uniform weight update** — On a case where most transitions are low-variance (e.g., approach phase) but a few key transitions (e.g., grasping) determine success, uniform sampling dilutes scarce updates, leaving the critical region inaccurate. Our method concentrates updates on those high-advantage transitions, yielding better model accuracy there. We expect our method to outperform uniform on tasks with a narrow success bottleneck, with a larger gap on the final critical step success rate than on overall trajectory accuracy.

**vs. Prediction error weighted update** — Prediction error weighting uses the same sampling distribution but based on model uncertainty (ensemble disagreement). If advantage weighting outperforms prediction error weighting, it demonstrates that advantage captures not just where the model is uncertain, but where uncertainty matters most for policy performance. We expect advantage weighting to match or exceed prediction error weighting on tasks with sparse rewards (where advantage is zero until near goal) and show clear advantage on tasks with dense rewards (where advantages are more informative).

### What would falsify this idea
If the advantage-weighted update does not outperform uniform sampling or prediction error weighting on success rate, or if the gain is spread uniformly across all task phases rather than concentrated on high-advantage transitions, then the central claim that advantage-weighting targets impactful errors is false.

## References

1. World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation
2. Diffusion for World Modeling: Visual Details Matter in Atari
3. Navigation World Models
4. STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning
5. GAIA-1: A Generative World Model for Autonomous Driving
6. Transformer-based World Models Are Happy With 100k Interactions
7. Learning Interactive Real-World Simulators
8. NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration

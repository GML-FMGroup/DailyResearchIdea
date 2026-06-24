# Self-Supervised Safety Capacity Functions for Generalizable Action Perturbation Bounds

## Motivation

Existing safe RL frameworks, such as SkillHarness, define safety boundaries using fixed harm categories or supervised examples, which fail to generalize to novel harmful actions because they cannot capture harm that falls outside observed categories. This structural reliance on predefined supervision means the agent is blind to unseen failure modes, leading to unsafe behaviors in open-ended environments. We need a mechanism that learns a dynamic, state-dependent safety boundary without requiring explicit harm labels, enabling generalization to novel actions.

## Key Insight

The maximum safe action perturbation before irrevocable harm is inversely related to the agent's own recovery ability—the capacity to return to a safe state—which can be self-supervised by contrasting trajectories that lead to recovery versus those that lead to irreversible failure.

## Method

### (A) What it is
We propose **Self-Supervised Safety Capacity (S3C)**, a method that learns a state-action-dependent scalar function \( C_\theta(s,a) \) estimating the maximum \( L_\infty \) perturbation magnitude an action can tolerate before the agent enters an irrecoverable state. Input: state \( s \), action \( a \). Output: bound \( \epsilon_{\max}(s,a) \). Training uses only the agent's own interactions without harm labels.

### (B) How it works
```python
# Training loop
Initialize recovery policy π_rec (SAC with default hyperparameters from Haarnoja et al. 2018) 
Initialize recovery value function V_rec(s) (2-layer MLP, hidden=256, ReLU activation)
Initialize safety capacity network C_θ(s,a) -> epsilon_max (same architecture as V_rec)
Initialize replay buffer D (capacity 1e6) and safe terminal set S_safe (states with obstacle distance > 0.5m and agent velocity < 0.1 m/s)

# Load-bearing assumption: V_rec(s') > τ (τ=0.5) accurately predicts recoverability within H=10 steps.
# Repair: Use ensemble of 5 V_rec networks; if disagreement (std) > 0.1, label as uncertain and skip transition.

for episode in range(N):
    s = env.reset()
    while not done:
        a = π_explore(s)  # e.g., current policy with Gaussian noise std=0.1
        δ = random_perturbation_magnitude()  # sampled uniformly from [0, Δ_max=0.5]
        random_direction = sample_hyper_sphere(dim=action_dim)  # uniform on hypersphere
        a_pert = clip(a + δ * random_direction, action_space)
        s_next, reward, done = env.step(a_pert)
        
        # Determine recoverability: can agent return to S_safe within H steps?
        # Use ensemble: mean(V_rec) > τ and std(V_rec) ≤ 0.1
        if not done and mean(V_rec(s_next)) > τ:
            recoverable = True
        else:
            recoverable = False
        
        # Compute target for C_θ
        if recoverable:
            # The perturbation was safe; push C_θ(s,a) to be at least δ
            target = δ + margin  # margin = 0.1
        else:
            # The perturbation caused irrecoverable state; push C_θ(s,a) below δ
            target = δ - margin
        
        # Update C_θ with hinge loss
        if recoverable:
            loss = max(0, C_θ(s,a) - target)
        else:
            loss = max(0, target - C_θ(s,a))
        gradient_step(loss, θ)  # Adam, lr=3e-4, batch=256
        
        # Store transition and update recovery components
        if std(V_rec(s_next)) ≤ 0.1:  # only store if confident
            D.push((s, a, s_next, recoverable))
        update_recovery_policy_and_value(D, π_rec, V_rec)  # SAC update: 1 gradient step per env step, batch=256, discount=0.99
        
        s = s_next
```
**Hyperparameters**: \( \tau = 0.5 \), \( H = 10 \), \( \Delta_{\max} = 0.5 \), margin = 0.1, learning rate = 3e-4, ensemble size = 5, uncertainty threshold = 0.1.

### (C) Why this design
We chose **three key design decisions** with specific trade-offs:
1. **Separate recovery policy and value function** instead of an integrated model: This allows the recovery module to be reused for multiple tasks and trained independently, but introduces error propagation when the recovery value is inaccurate due to approximation error or distribution shift. 2. **Hinge loss with a margin** rather than MSE regression: The hinge loss creates a buffer around the safety boundary, making the estimator more robust to noisy recoverability labels, at the cost of slower convergence when the margin is too large. 3. **State-action dependent capacity** instead of state-only: Action-specific bounds are necessary because different actions may have different inherent risk (e.g., large joint torque vs. fine motion), but this increases sample complexity and may overfit to frequently seen actions. These design choices prioritize generalization over sample efficiency, accepting that fine-tuning may be needed for new environments.

### (D) Why it measures what we claim
- **Recovery value \( V_{rec}(s') \)** measures **recovery ability** because we assume that a value above threshold \( \tau \) predicts the agent can return to the safe set under the current recovery policy; this assumption fails when the recovery policy is suboptimal or dynamics are stochastic, in which case the metric may underestimate or overestimate actual recoverability. To mitigate, we use ensemble uncertainty to filter unreliable predictions. 
- **Perturbation magnitude \( \delta \)** measures **actual action deviation** because it is directly applied; we assume it approximates the true boundary when sampled densely; this assumption fails when the perturbation direction is not aligned with the most dangerous direction, and the metric then reflects a conservative lower bound. 
- **Hinge loss target \( \delta \pm \text{margin} \)** operationalizes **maximum allowable perturbation** because it forces \( C_\theta \) to lie between safe and unsafe regions; this assumes the safety boundary is monotonic along the perturbation axis; if the boundary is non-monotonic (e.g., safe islands), the metric may incorrectly treat some safe perturbations as unsafe or vice versa.
- **Safe set \( S_{safe} \)** measures **avoidability of harm** because it defines states from which the agent can recover; this assumes the safe set can be pre-specified without harm labels; if the task or dynamics change, the predefined set may become invalid, making recoverability labels unreliable.

## Contribution

(1) A novel self-supervised method (S3C) for learning a state-dependent safety capacity function without requiring predefined harm categories or supervised examples. (2) The insight that recovery ability serves as a self-supervised signal for safe perturbation bounds, enabling generalization to novel actions unseen during training. (3) A practical algorithm combining recovery policy learning with a hinge-loss objective that can be integrated into existing safe RL pipelines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Safety Gym PointGoal1, CarGoal1, simulated 7-DOF robot arm (Sawyer Reach with obstacles) | Covers diverse locomotion and manipulation tasks. |
| Primary metric | Cumulative safety cost; recoverability prediction ROC-AUC | Directly measures safety; validates assumption. |
| Baseline 1 | PPO-Lagrangian | Classical constrained RL method. |
| Baseline 2 | Random perturbation baseline (fixed max magnitude 0.5) | No safety mechanism. |
| Baseline 3 | Reachability-based safety bound (Hamilton-Jacobi with grid of 10^4 cells) | Provable but conservative bound. |
| Ablation-of-ours | S3C state-only (no action dependence) | Isolates benefit of action-specific capacity. |

### Why this setup validates the claim
This combination tests the core hypothesis: that learning a state-action-dependent safety capacity via self-supervised recoverability labels improves safety over methods that ignore action-specific risk. By comparing against PPO-Lagrangian (which treats all actions equally under a constraint) and a random perturbation baseline (which has no safety mechanism), we can attribute any safety improvement to the learned capacity. The reachability baseline provides an upper bound on safety (provable but impractical), highlighting the trade-off between conservatism and permissiveness. The ablation removes action dependence, testing whether the added complexity is justified. Cumulative safety cost directly measures the method's ability to prevent irrecoverable states, making it a falsifiable metric: if our method truly captures action-specific tolerance, it should yield lower costs than baselines, especially in states where actions differ in risk. Additionally, we measure ROC-AUC of recoverability predictions against ground-truth (simulator check) to verify the load-bearing assumption.

### Expected outcome and causal chain

**vs. PPO-Lagrangian** — On a state where a large perturbation is safe (e.g., open field) but a small one is risky (e.g., near cliff), PPO-Lagrangian applies a uniform constraint, causing it to be overly conservative in safe regions or insufficiently cautious in risky ones. Our method uses the recovery value to per-action adjust the allowed perturbation, allowing safe large actions while restricting dangerous ones. We expect our method to achieve significantly lower cumulative safety cost (e.g., 30–50% reduction) while maintaining similar task success.

**vs. Random perturbation baseline** — This baseline blindly applies random perturbations up to a fixed magnitude, frequently causing irrecoverable states (e.g., collisions). Our method learns to cap perturbations below the recoverability threshold, so it avoids most catastrophic failures. We expect our method to have near-zero safety violations, while the random baseline incurs high cost (e.g., >5 violations per episode).

**vs. Reachability-based bound** — The reachability bound is provably safe but overly conservative, leading to frequent false positives and thus lower task success. Our method learns a tighter per-action bound, maintaining safety while allowing more exploration. We expect our method to have 10–20% lower safety cost and 20–30% higher task success than the reachability baseline.

**vs. S3C state-only ablation** — On a task where two actions have different inherent risk (e.g., high-torque vs. low-torque movement), the state-only model predicts the same capacity for both, leading to overestimation for risky actions. Our action-dependent model gives distinct bounds, preventing overconfident unsafe actions. We expect our method to outperform the ablation specifically in regions where action choice matters (e.g., near narrow passages), with a 15–25% lower cost in those states. Additionally, varying τ shows that performance peaks at τ=0.5; too low (τ=0.2) yields unsafe actions, too high (τ=0.8) yields overconservatism.

### What would falsify this idea
If the safety cost improvement of our full method over the state-only ablation is uniform across all state-action pairs rather than concentrated in action-sensitive states, then the action-dependence is not driving the gain, undermining the central claim. Also, if recoverability prediction ROC-AUC is below 0.7 on any domain, the load-bearing assumption is not satisfied, indicating need for better calibration.

## References

1. SkillHarness: Harnessing Safe Skills for Computer-Use Agents
2. WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks
3. OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents
4. ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents
5. Agent-as-a-Judge: Evaluate Agents with Agents
6. Poisoning Retrieval Corpora by Injecting Adversarial Passages
7. Assessing Prompt Injection Risks in 200+ Custom GPTs
8. FinMem: A Performance-Enhanced LLM Trading Agent With Layered Memory and Character Design

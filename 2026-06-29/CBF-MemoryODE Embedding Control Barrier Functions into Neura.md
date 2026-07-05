# CBF-MemoryODE: Embedding Control Barrier Functions into Neural ODE Latent Dynamics for Unified Memory and Safety in Visual Navigation

## Motivation

The frontier model Qwen-RobotNav discards explicit memory gating (from TrackVLA++) and post-hoc safety reflection (from Discrete Diffusion for Reflective VLA) without providing an alternative mechanism for spatiotemporal consistency or safe recovery. This leaves the model vulnerable to inconsistency over long horizons and unsafe failures in rare scenarios. The root cause is the architectural separation of memory and safety into dedicated modules; their removal creates a structural void that a unified latent dynamics mechanism can fill.

## Key Insight

By encoding a control barrier function into the latent ODE dynamics, we simultaneously enforce safe state constraints and regulate memory retention through a safety-informed gating term, ensuring that the latent state evolves continuously while remaining within a safe set.

## Method

**(A) What it is** — CBF-MemoryODE is a neural-ODE-based visual navigation policy that unifies memory gating and safety scaling in a single latent dynamics. Input: a sequence of visual observations and a navigation command; output: a continuous-time latent state trajectory that directly decodes to safe velocity commands.

**(B) How it works** — Pseudocode for forward pass:
```python
def forward(obs_seq, cmd):
    z0 = encoder(obs_seq[0], cmd)  # encoder: ResNet-18 (pretrained then fine-tuned) -> 128-dim; cmd embedded via learned 32-dim and concatenated
    def dynamics(t, z):
        obs_t = interpolate_obs(obs_seq, t)  # linear interpolation between timesteps
        u = control_head(z, obs_t, cmd)  # control_head: 2-layer MLP hidden=256, ReLU
        h = safety_critic(z)  # safety_critic: 3-layer MLP hidden=512, tanh
        g = 1 - sigmoid(alpha * h)  # alpha=1.0, memory gating factor
        if h < safe_threshold:  # safe_threshold=0.2
            recovery = -beta * grad_h(z)  # beta=0.5
        else:
            recovery = 0
        dz = (model_dynamics(z, u) * g + recovery) * dt  # model_dynamics: 3-layer MLP hidden=512, tanh
        return dz
    dt = 0.2  # T=10, steps=50
    z_traj = odeint(dynamics, z0, t_span=(0, 10), method='euler')
    action = action_head(z_traj[-1])  # action_head: 2-layer MLP hidden=256, linear output
    return action
```
Training: end-to-end with behavior cloning loss on expert trajectories and a safety constraint loss. We enforce the CBF condition ḣ ≥ -α h on sampled states by adding a penalty term L_CBF = max(0, -ḣ - α h) to the loss, with α=1.0. The overall loss is L = L_BC + λ L_CBF, λ=0.1.

**Load-bearing assumption:** We assume that the learned safety critic h(z) constitutes a valid control barrier function for the latent dynamics, meaning that its value and gradient correctly indicate proximity to unsafe states and direction to safety.

**(C) Why this design** — We chose a neural ODE over a discrete RNN because continuous latent dynamics allow the CBF to be defined in a differentiable manner and guarantee smooth state evolution, at the cost of increased computation from numerical integration. We chose to learn the CBF h(z) via a safety critic rather than hand-design it, because safety conditions in navigation are environment-dependent; the trade-off is that the critic may provide only approximate guarantees. We chose to implement memory gating as a multiplicative factor on the dynamics rather than as a separate gating mechanism (e.g., LSTM gates) because this directly couples memory retention with safety, preventing the model from ignoring safety when memory is active; the downside is that safety and memory are now entangled, which could lead to catastrophic forgetting if the CBF is wrong. Compared to prior work like Coconut (continuous latent space reasoning), our method differs in that the latent dynamics are explicitly constrained by safety, not free; our CBF provides a structured prior that the Coconut-style free-form latent does not offer.

**(D) Why it measures what we claim** — The computational quantity g = 1 - sigmoid(alpha * h(z)) measures the degree of memory gating because it scales down the latent velocity when h(z) is low (unsafe), assuming that low h(z) correlates with high risk of entering an unsafe region; this assumption fails when h(z) is miscalibrated (e.g., overconfident in safe states), in which case g remains near 1 and memory is not suppressed. The recovery term -beta * grad_h(z) measures the urgency of returning to safety, assuming that the gradient of h points toward the safe set; this assumption fails when the safety landscape has spurious local minima, in which case the recovery may steer the latent state to a nominally safe but functionally unsafe region (e.g., a dead end). The ODE solver output z_traj as a continuous trajectory measures spatiotemporal consistency because the neural ODE enforces smooth evolution between time steps; this assumption holds if the ODE is Lipschitz continuous, but fails when the dynamics are numerically stiff, in which case the solver may introduce discontinuities. The combined operation of multiplicative gating and recovery gradient measures the coupled safety-memory assumption that safety-aware memory modulation does not destructively interfere with recovery; this assumption fails if the gating term over-suppresses dynamics even when recovery is needed, in which case the latent state may stall and fail to escape unsafe regions.

## Contribution

(1) A unified mechanism for memory gating and safety scaling via control barrier functions embedded in neural ODE latent dynamics, removing the need for separate modules. (2) A training objective that jointly learns the ODE dynamics and the safety critic from demonstration data, ensuring end-to-end trainability. (3) Empirical evidence that the approach provides post-recovery guarantees in navigation scenarios where the frontier Qwen-RobotNav model fails due to inconsistency or unsafe states.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GibsonEnv | Realistic indoor scenes; standard benchmark. |
| Primary metric | SPL | Measures both success and efficiency. |
| Baseline 1 | NeuralODE-Nav (no CBF) | Same ODE dynamics but without CBF terms; isolates effect of safety constraint. |
| Baseline 2 | LSTM-Nav (discrete, no CBF) | LSTM hidden size 256, same encoder/decoder; tests advantage of continuous dynamics. |
| Ablation-of-ours | CBF-MemoryODE w/o recovery | Removes recovery; tests memory gating alone. |

### Why this setup validates the claim

The combination of GibsonEnv, SPL, and the chosen baselines forms a falsifiable test of the central claim that coupling memory gating with safety in continuous latent dynamics improves navigation robustness. Comparing against NeuralODE-Nav isolates the contribution of the learned CBF: if SPL improves, it confirms that safety-aware latent dynamics prevent risky behaviors. Comparing against LSTM-Nav tests whether continuous latent evolution provides smoother, more reliable control than discrete steps. The ablation w/o recovery tests the necessity of the recovery term for regaining safety. SPL is appropriate because it penalizes both failures and inefficient paths, directly reflecting the method's ability to balance safety and progress. If our method only helps in safe corridors but fails in cluttered scenes, the expected patterns would not appear, falsifying the claim.

### Expected outcome and causal chain

**vs. NeuralODE-Nav (no CBF)** — On a case where the robot drifts toward an obstacle, the baseline's latent dynamics ignore safety, leading to a collision or a sudden erratic path because it lacks risk awareness. Our method instead scales down memory via gating and applies a recovery gradient, steering the latent state away from the unsafe region, so we expect a noticeable SPL gain in cluttered scenarios but similar performance in open spaces.

**vs. LSTM-Nav (discrete, no CBF)** — On a case requiring fine-grained obstacle avoidance (e.g., a narrow doorway), the baseline's discrete latent updates cause jerky trajectories or oscillation because it cannot interpolate smoothly. Our method's continuous ODE produces a smooth, safe path through the opening, so we expect a clear SPL advantage on narrow passages and comparable performance on wide corridors.

**vs. CBF-MemoryODE w/o recovery** — On a case where the robot enters a dead end, the ablation's gating slows dynamics but provides no directional correction, so the latent state stalls near the unsafe region, failing to recover. Our method adds a recovery gradient that drives the state back toward safety, so we expect a higher success rate and SPL on scenarios requiring retreat, while both perform similarly on straightforward paths.

### What would falsify this idea

If our method shows no SPL degradation when the recovery term is removed, or if its advantage over NeuralODE-Nav is uniform across all scene types rather than concentrated in cluttered regions, then the central claim that safety-constrained memory gating is responsible for the improvement would be falsified.

## References

1. Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System
2. ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving
3. TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking
4. DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving
5. EMMA: End-to-End Multimodal Model for Autonomous Driving
6. DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
7. Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving
8. Training Large Language Models to Reason in a Continuous Latent Space

# FARA: Fast Adaptation via Residual Online Adjustment for Frozen Foresight Modules

## Motivation

InternVLA-A1.5 uses a frozen foresight module pretrained on diverse data, but it cannot adapt to robot-specific dynamics at deployment time because its weights are frozen. Prior online adaptation methods (e.g., in Motus) can adapt via fine-tuning, but they require modifying the entire model or have high sample complexity. The structural gap is that the foresight module's latent space is rich yet inaccessible to adaptation, necessitating a lightweight correction that preserves pretrained knowledge.

## Key Insight

The frozen foresight module's latent space is a sufficient representation of task-relevant dynamics, so a learned residual correction in that space can bridge distribution shift without forgetting, because the correction network is trained online via RL and only adjusts the latent, not the underlying parameters.

## Method

**FARA: Fast Adaptation via Residual Online Adjustment**

**(A) What it is** — FARA is a lightweight online correction network that takes the frozen foresight module's latent prediction and the observed history, and outputs a residual to the latent, enabling the model to adapt to robot-specific dynamics without modifying frozen weights. It is trained via online reinforcement learning using task rewards.

**(B) How it works** — Pseudocode:
```pseudocode
Initialize correction network C_θ with small random weights.
Initialize value baseline V_φ (optional).
Initialize replay buffer D of size N=100 episodes.
Frozen foresight module F (from InternVLA-A1.5) and action expert A.
For each episode:
  Receive initial observation o_0.
  For t = 0 to T-1:
    # Foresight module predicts latent
    l_t = F(o_t, instruction)   # latent representation of future
    # Correction network predicts residual
    r_t = C_θ(l_t, h_t)         # h_t = history summary (e.g., last 3 latents and actions)
    # Correct latent
    l'_t = l_t + r_t
    # Action expert outputs action
    a_t = A(l'_t)
    Execute a_t, get next observation o_{t+1} and reward r_t.
    Store transition (l_t, h_t, r_t, a_t, reward, done) in D.
  # After episode, compute discounted returns G_t.
  # Update C_θ using REINFORCE with baseline:
  #   ∇_θ = E[(G_t - V_φ(h_t,l_t)) ∇_θ log π(a_t | l'_t)]
  #   where π is implicitly the deterministic action expert A, but we treat correction output r_t as the action.
  #   Actually, the correction network's output r_t determines the latent, so log π = -||r_t||^2? No, since correction is deterministic, we use a stochastic version: add Gaussian noise N(0,σ) to r_t during training, then log π = -||(r_t - μ)||^2 / (2σ^2). σ=0.1 fixed.
  # Update baseline V_φ via MSE on G_t.
  # Optionally, sample replay buffer to stabilize.
  # Hyperparameters: learning rate=1e-3, discount factor=0.99, σ=0.1, buffer size=100, batch size=16.
```

**(C) Why this design** — We chose to apply the correction in latent space rather than action space because the action expert is pretrained to map latents to actions; adjusting actions directly would require retraining the expert, whereas latent residuals preserve the expert's understanding. We selected REINFORCE with a baseline over PPO because the correction network is small (2-layer MLP, 128 units) and the online adaptation runs per-episode; PPO's multiple surrogate optimization epochs could overfit to a single episode and cause forgetting. We added a small replay buffer of recent episodes (size 100) to decorrelate updates and stabilize training, accepting the memory cost of storing history summaries. The stochastic policy (adding Gaussian noise to the correction) enables exploration without a separate exploration policy, trading off lower sample efficiency for simplicity. We did not use a separate exploration strategy (e.g., epsilon-greedy) because the correction noise is centered on the current estimate and naturally diminishes as the policy converges. These choices collectively ensure the correction network learns a residual that reduces distribution shift without catastrophic forgetting of the frozen module.

**(D) Why it measures what we claim** — The residual magnitude ∥r_t∥ measures the distribution shift between the foresight module's pretraining distribution and the deployment dynamics, because the correction network is trained to minimize task error via RL; it will learn to output zero only when the foresight latent already leads to optimal actions. The assumption that the correction is sufficient to compensate for shift relies on the latent space being a complete feature space for dynamics; this assumption fails when the foresight module fails to encode some critical aspect of the task (e.g., a new object not in pretraining), in which case the correction may need to compensate but the latent is fundamentally missing information, and the metric would reflect a residual that cannot fully bridge the gap. The RL reward signal G_t directly measures task success, so the correction network's output is causally linked to the motivation-level concept of adaptation to robot-specific dynamics. The value baseline V_φ estimates expected return from given latent and history, ensuring the policy gradient update is not biased by reward variance; the assumption that V_φ converges to the true expected return is asymptotically valid for a linear or small neural net, but in practice may be biased, leading to less stable adaptation.

## Contribution

(1) A lightweight online correction network that adapts frozen foresight modules to robot-specific dynamics via latent-space residual adjustments, preserving all pretrained weights. (2) An online RL training procedure (REINFORCE with baseline) that learns the correction from episodic task rewards, achieving sample-efficient adaptation without model fine-tuning. (3) Design principle that latent-space residuals are effective for domain adaptation in visuomotor policies with frozen pretrained modules.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SimplerEnv Google Robot | Tests distribution shift from pretraining |
| Primary metric | Task success rate | Directly measures adaptation quality |
| Baseline 1 | Frozen foresight (no correction) | Isolates effect of online adaptation |
| Baseline 2 | Action-space correction (FARA variant) | Compare latent vs action modification |
| Ablation-of-ours | FARA without replay buffer | Assesses buffer contribution to stability |

### Why this setup validates the claim
This design tests whether residual latent correction reduces distribution shift in robot dynamics. The SimplerEnv Google Robot dataset is chosen because its tasks require compositional generalization, a known failure point for frozen models (as seen in InternVLA-A1.5). Frozen foresight baseline shows the gap from pretraining only; action-space correction tests whether latent correction is superior. The ablation removes replay buffer to verify if memory decorrelation is necessary. Task success rate is the unambiguous metric that directly reflects the motivation-level concept of adaptation, as it measures whether the robot accomplishes its goal. A falsifiable claim requires that if correction works, success rate on novel dynamics tasks should improve over frozen model, and latent correction should outperform action correction on tasks where action distribution shift is severe.

### Expected outcome and causal chain

**vs. Frozen foresight (no correction)** — On a case where dynamic object properties differ from pretraining (e.g., lighter object requiring slower grasp), the frozen model produces failed grasps because its latent representation is statically biased. Our method instead uses the correction network to adjust the latent online via RL, so we expect a noticeable gap (e.g., >20% success rate improvement) on such dynamics-shifted tasks, but near parity on tasks matching pretraining distribution.

**vs. Action-space correction (FARA variant)** — On a case where the action expert is pretrained to map specific latent patterns to actions, modifying actions directly changes the mapping, causing unstable behavior because the expert's internal assumptions are violated. Our method adjusts the latent, preserving the expert's learned mapping, so we expect higher stability and success rate (e.g., 10-15% improvement) in high-frequency tasks, but similar performance on simple tasks where action correction also works.

**vs. FARA without replay buffer** — On a case with temporally correlated episodes (e.g., repeated attempts of same task), the no-replay variant suffers from policy oscillation due to on-policy update variance. Our method with buffer decorrelates updates, so we expect smoother learning curves and faster convergence (e.g., reaching 90% success in half the episodes).

### What would falsify this idea
If the success rate improvement over frozen foresight is uniform across all dynamics shifts rather than concentrated on tasks with distribution mismatch (e.g., unexpected objects), then the correction is not truly compensating for shift but merely improving overall—this would falsify the claim that the residual targets distribution shift.

## References

1. InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization
2. InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy
3. UniUGP: Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving
4. Motus: A Unified Latent Action World Model
5. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model
6. DrivingGPT: Unifying Driving World Modeling and Planning with Multi-Modal Autoregressive Transformers
7. DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT
8. Flow Matching Guide and Code

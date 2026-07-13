# Co-Adaptive Memory-Action Optimization via Lyapunov-Guided Coupling

## Motivation

Prior work like 'Remember When It Matters' treats the action agent as a fixed LLM and optimizes only the memory agent, assuming the action agent's policy is invariant to memory interventions. This assumption is structurally flawed because memory injections alter the action agent's hidden state, and ignoring this coupling leads to suboptimal long-horizon performance. The separate optimization prevents the action agent from adapting its internal representations to leverage memory, causing instability when memory is injected.

## Key Insight

The joint system's long-horizon coherence can be quantified by a Lyapunov function whose gradient descent yields coupled updates for both agents, guaranteeing that memory interventions stabilize rather than perturb the action policy.

## Method

(A) **What it is**: We propose CAMOA, a training framework for memory-augmented agents where both the memory agent (M) and the action agent (A) are co-optimized using a Lyapunov function that measures joint trajectory coherence. Inputs: pre-trained GPT-2 Small (124M) base models for A and M. Outputs: fine-tuned A and M with coupled parameters. **Load-bearing assumption**: The learned Lyapunov network L_ψ accurately represents a valid Lyapunov function for the joint dynamics, such that its gradients guarantee monotonic coherence decrease and stable coupling. This assumption is verified during training (see verification step in algorithm).

(B) **How it works**:
```pseudocode
Algorithm: CAMOA Training

Input: pre-trained action agent A_θ (GPT-2 Small), memory agent M_φ (GPT-2 Small), tasks T, Lyapunov network L_ψ (2-layer MLP, hidden=256, GeLU)
Hyperparameters: λ (coupling strength)=0.1, η_A=η_M=1e-5 (learning rates), η_ψ=1e-4, K=100 (memory bank size), δ=0.01 (margin)

1. Initialize θ, φ from pre-trained checkpoints, ψ randomly.
2. For each task episode:
   a. Reset environment, initialize memory bank B = []
   b. For time step t = 1..T:
      i. M_φ observes current state s_t and B; decides to inject reminder r_t or remain silent (via softmax over two outputs).
      ii. A_θ receives input: current observation o_t and reminder r_t (if any, else [PAD]).
      iii. A_θ produces action a_t (via argmax over token logits); environment returns next state s_{t+1} and reward.
      iv. Update memory bank B with new trajectory step (s_t, a_t, r_t) — stored as (state_embedding, action_embedding, reminder_embedding).
   c. After episode, compute Lyapunov loss:
      i. Let τ = (s_1..s_T, a_1..a_T, r_1..r_T) be the trajectory.
      ii. For each t, compute local coherence: C_t = L_ψ(h_t^A, h_t^M) where h_t^A, h_t^M are hidden states (last layer, 768-dim) of A and M at time t.
      iii. Lyapunov loss L_lyap = Σ_t max(0, C_{t+1} - C_t + δ)   (δ > 0 ensures strict decrease)
   d. Update parameters:
      i. ψ ← ψ - η_ψ ∇_ψ L_lyap
      ii. θ ← θ - η_A ∇_θ L_lyap
      iii. φ ← φ - η_M ∇_φ L_lyap
   e. Verification step: check Lyapunov conditions across episode — if C_t < C_{t-1} fails for >20% of timesteps, reset ψ to random initialization and skip parameter updates for this episode (continue to next episode).
   f. Optionally, distill Lyapunov dynamics into policy via auxiliary RL loss (PPO with shared episode return as reward, not shown for brevity).
```

(C) **Why this design**: We chose a Lyapunov function to measure coherence instead of a simple reward sum because Lyapunov stability captures monotonic convergence of the joint state, which is necessary for long-horizon tasks where rewards are sparse. This requires a learned Lyapunov network L_ψ, which introduces an additional training component but provides a dense, non-sparse signal that can guide both agents simultaneously. We couple the updates via gradients of the same Lyapunov loss rather than optimizing separate objectives, accepting that the agents' gradients may conflict; however, the Lyapunov loss ensures that any conflict is resolved in favor of reducing joint instability. We use a hinge-like loss with margin δ to enforce strict decrease rather than a simple MSE, accepting a hyperparameter that controls the strictness of monotonic decrease. We also choose to use hidden states h_t^A and h_t^M as inputs to L_ψ rather than final outputs, because hidden states capture the internal representations that are affected by memory injection. This design trades off interpretability for expressiveness.

(D) **Why it measures what we claim**: The Lyapunov loss L_lyap measures joint system coherence because decreasing L_lyap ensures that the difference between consecutive local coherence values is non-increasing by at least δ. The computational quantity C_t = L_ψ(h_t^A, h_t^M) measures the 'coupling quality' between the memory and action agents at step t under the assumption that L_ψ accurately represents the energy-like potential of the joint system; this assumption fails when L_ψ is poorly optimized (early training) or when the hidden state spaces are misaligned, in which case C_t may capture spurious correlations. The margin δ measures the required monotonicity guarantee, under the assumption that the joint dynamics are contractive; this assumption fails when the dynamics are naturally expanding (e.g., chaotic tasks), in which case the loss forces an unnatural constraint and training may diverge. The shared gradients from L_lyap operationalize the co-adaptive coupling because each agent's update reduces the same scalar, ensuring that memory injections do not increase instability as measured by C_t. Additionally, we assume that monotonic decrease in C_t implies policy stability and thus increased task success rate. This assumption fails when tasks require temporary instability for exploration (e.g., when the agent must deviate from coherent behavior to discover new strategies), in which case the Lyapunov loss may discourage beneficial exploration.

## Contribution

(1) A novel co-optimization framework for memory and action agents based on a shared Lyapunov function, eliminating the independence assumption in prior work. (2) A design principle: Lyapunov-based coupling guarantees monotonic coherence in the joint state space, providing a principled way to measure and optimize joint system stability. (3) A concrete instantiation using a learned Lyapunov network and hinge-loss objective that operationalizes the coupling.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Terminal-Bench 2.0 (200 tasks, avg. episode length 35 steps, sparse rewards only at termination) | Long-horizon agentic tasks with sparse rewards, including subtasks requiring time-sensitive memory and distraction sensitivity. |
| Primary metric | Pass@1 (task success rate over 10 runs per task) | Measures overall task success rate. |
| Baseline 1 | Passive Bank Exposure (full memory bank appended to context at each step, no proactive selection) | No proactive memory injection. |
| Baseline 2 | Always-On Injection (random memory from bank injected at each step) | Constant injection without selectivity. |
| Ablation-of-ours | CAMOA w/o Lyapunov (both agents co-trained via PPO with shared episode return as reward, no Lyapunov loss) | Separate training without Lyapunov coupling. |

### Why this setup validates the claim
Terminal-Bench 2.0 features long-horizon tasks where memory-agent coupling is critical for success. The baselines – passive bank exposure and always-on injection – directly test the benefit of selective, proactive memory modulation. The ablation removes the Lyapunov loss while retaining co-training, isolating the coupling mechanism. Pass@1 as the primary metric captures the end-to-end task completion, which should improve if the Lyapunov-derived coherence reduces cumulative errors. If our method outperforms baselines on tasks requiring timely memory retrieval and sensitivity to distractions, while the ablation shows degraded performance on long tasks, it would confirm that the Lyapunov coupling is responsible for the gains.

### Expected outcome and causal chain

**vs. Passive Bank Exposure** — On a case where a critical reminder must be injected at a specific turn to avoid forgetting, the passive baseline fails because it only exposes the full memory bank without proactive selection, causing the agent to miss the cue. Our method instead injects a tailored reminder at the right moment because the Lyapunov loss penalizes coherence drops that would occur if the reminder were absent, so we expect a significant gap (e.g., Pass@1 >0.8 for ours vs. <0.4 for baseline) on tasks with time-sensitive memory dependencies but parity on tasks where passive access suffices.

**vs. Always-On Injection** — On a case where an irrelevant memory distracts the agent from current goals, always-on injection floods the context, leading to action errors due to information overload. Our method suppresses such injections because the Lyapunov loss would increase if distraction caused a coherence drop, so we expect our method to outperform (Pass@1 >0.7 vs. <0.3) on tasks sensitive to distractions, with the baseline showing higher variance (std >0.2) and lower pass rates on noisy tasks.

**vs. CAMOA w/o Lyapunov** — On a long task where the memory and action agents’ gradients conflict (e.g., memory injects a reminder that momentarily misaligns the action policy), the ablation fails because it optimizes a shared reward, allowing conflicting updates that destabilize the joint policy. Our method uses the Lyapunov loss to enforce monotonic coherence, resolving conflicts in favor of stability, so we expect our full method to achieve higher pass rates (Pass@1 >0.6 vs. <0.4) on tasks with horizons ≥20 steps while the ablation degrades beyond a certain length (e.g., Pass@1 drops by >0.2 from 10-step to 20-step tasks).

### What would falsify this idea
If our method’s gains are uniform across all task subsets (i.e., within 5% of each other in Pass@1) rather than concentrated on tasks with time-sensitive memory or distraction sensitivity, or if the ablation performs within 5% of the full method on tasks with horizons >20 steps, then the central claim that Lyapunov coupling is essential would be falsified.

## References

1. Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents
2. Augmenting Language Models with Long-Term Memory
3. Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks
4. How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models
5. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
6. Training Language Models with Memory Augmentation
7. YaRN: Efficient Context Window Extension of Large Language Models
8. MemGPT: Towards LLMs as Operating Systems

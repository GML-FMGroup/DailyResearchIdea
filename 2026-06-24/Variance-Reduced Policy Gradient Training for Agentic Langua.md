# Variance-Reduced Policy Gradient Training for Agentic Language Models via Curated Multi-Turn Trajectories

## Motivation

Existing agentic model training pipelines, such as the data recipe approach in OpenThoughts-Agent, focus on curating diverse task trajectories but rely on supervised fine-tuning, which ignores the high variance in multi-turn agentic tasks due to varying trajectory lengths and action spaces. This variance leads to unstable learning and suboptimal policy improvement. We identify that the curated data, while diverse, can be repurposed to provide a rich set of baselines for variance reduction.

## Key Insight

The curated multi-turn trajectory dataset, by construction, contains trajectories with varying returns from diverse tasks, which can serve as a natural source for learning a state-value function that generalizes across agentic tasks, thereby reducing gradient variance during policy gradient training.

## Method

We propose VRP-Agent (Variance-Reduced Policy Gradient for Agentic Models).

(A) **What it is**: VRP-Agent is a training algorithm that combines Generalized Advantage Estimation (GAE) with trust region constraints (PPO) to fine-tune agentic language models on a curated dataset of multi-turn trajectories. The inputs are a set of trajectories with per-step rewards, and the output is an updated policy. **Assumption: The value function V_φ trained on the static curated dataset provides unbiased estimates of expected returns under the current policy π_θ for all states in D. Failure mode: Off-policy distribution shift can cause V_φ to be biased. Mitigation: We add a KL divergence penalty to keep π_θ close to the data-generating policy.**

(B) **How it works**:
```pseudocode
Input: Curated trajectories D = {τ_i = (s_t, a_t, r_t)} from OpenThoughts-Agent pipeline.
Initialize: Policy π_θ (base model: Llama-2-7B), Value function V_φ (2-layer MLP, hidden=256, GeLU).
Hyperparameters: γ=0.99, λ=0.95, ε=0.2, learning rate 1e-5 for both policy and value, β=0.01 (KL penalty coefficient).
For each iteration:
  1. Sample batch of trajectories from D.
  2. For each trajectory, compute advantages using GAE:
     δ_t = r_t + γ V_φ(s_{t+1}) - V_φ(s_t)
     A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
  3. Optimize policy using clipped PPO objective with KL penalty:
     L_CLIP(θ) = E_t[ min(ratio_t A_t, clip(ratio_t, 1-ε, 1+ε) A_t) ] - β * D_KL(π_θ || π_data)
     where ratio_t = π_θ(a_t|s_t) / π_old(a_t|s_t), π_data is the initial policy (behavior policy).
  4. Update value function by minimizing MSE between V_φ(s_t) and empirical returns R_t = Σ_{l=0}^{T-t} γ^l r_{t+l}.
  5. Optionally apply trust region via KL penalty if clipping insufficient.
```
Training is performed on 8×A100-80GB GPUs with a total batch size of 256 trajectories.

(C) **Why this design**: We chose GAE over Monte Carlo returns because GAE provides a tunable bias-variance trade-off via λ, which is crucial for multi-turn tasks where rewards are sparse and trajectory lengths vary (bias from short rollouts can be mitigated by lower λ, but at increased variance). We opted for PPO's clipped objective over TRPO's explicit KL constraint because clipping is computationally cheaper and integrates naturally with stochastic gradient descent, accepting the risk of occasional large policy updates that violate the trust region if clipping is insufficient. We train a separate value network V_φ instead of using a single network with both policy and value heads to avoid interference between the two objectives, though this increases parameter count and memory. We use the curated trajectories as a static dataset (off-policy) rather than on-policy rollouts because the data is already diverse and collecting new rollouts per iteration is expensive; however, this introduces off-policy bias that we mitigate by importance sampling via the ratio in the PPO objective and an additional KL penalty to keep the policy near the behavior policy.

(D) **Why it measures what we claim**: The GAE advantage A_t measures the relative benefit of action a_t over the average action at state s_t, operationalizing the concept of credit assignment for multi-turn decisions; the assumption is that the value function V_φ(s_t) accurately estimates the expected return from s_t under the current policy, which fails when the value function is poorly approximated for states unseen in the curated data, in which case A_t reflects next-token likelihood bias rather than true advantage. The PPO clipping mechanism measures the deviation of the new policy from the old policy, operationalizing trust region constraints to ensure stable updates; the assumption is that the clipped ratio prevents destructive policy changes, which fails when the reward signal is noisy such that clipping leads to insufficient progress, in which case the constraint becomes a bottleneck. The value function MSE loss measures the accuracy of return prediction; the assumption is that empirical returns R_t are valid targets for all states, which fails when trajectory returns are influenced by stochastic environment transitions not captured in the dataset, in which case the value function learns a confounded estimate. **Missing equivalence: GAE advantage (A_t) measures true advantage (M). Assumption (A): V_φ(s_t) is an unbiased estimate of expected return under π_θ for all states in D. Failure (F): If V_φ is biased due to off-policy distribution shift, A_t reflects next-token likelihood bias instead.**

## Contribution

(1) Introduction of VRP-Agent, the first training algorithm that integrates variance-reduced policy gradient (GAE+PPO) with a curated data recipe pipeline for agentic language models, directly addressing the high-variance gradient issue in multi-turn tasks. (2) Empirical demonstration that leveraging the diversity of curated trajectories for value function learning reduces gradient variance and yields more stable policy improvement compared to supervised fine-tuning or naive REINFORCE.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OpenThoughts-Agent curated 100K trajectories | Covers diverse multi-turn agentic tasks |
| Base model | Llama-2-7B | 7B parameter model for tractable fine-tuning |
| Hardware | 8×A100-80GB | Standard GPU setup for training |
| Primary metric | Average success rate across 7 agentic benchmarks (including WebArena) | Directly measures task completion quality |
| Baseline 1 | SFT on same data | Tests whether RL adds value over imitation |
| Baseline 2 | REINFORCE with Monte Carlo returns | Isolates effect of GAE variance reduction |
| Baseline 3 | PPO with shared policy/value network | Tests if separate networks avoid interference |
| Baseline 4 | TD(λ) with GAE | Tests if GAE is better than simpler TD(λ) |
| Ablation-of-ours | VRP-Agent without GAE (MC returns) | Quantifies benefit of GAE bias-variance trade-off |
| Ablation-of-ours (on-policy) | VRP-Agent with on-policy rollouts | Isolates effect of off-policy bias |
| Additional benchmark | WebArena | Real-world web tasks for practical impact |

### Why this setup validates the claim

The dataset is the exact same curated trajectories used during development, ensuring fairness. Comparing against SFT isolates the contribution of RL (advantage + trust region). REINFORCE with MC returns tests if GAE's bias-variance trade-off is crucial for sparse/reward settings. Shared network PPO tests if separate value networks reduce objective interference. TD(λ) baseline tests whether GAE's multi-step advantage is superior to a simpler bootstrapping method. The ablation (our method without GAE) directly measures GAE's impact. The on-policy ablation quantifies the effect of off-policy bias. Including WebArena ensures practical relevance. Together, these baselines form a clean ladder: supervised → simple RL → RL with GAE → our full method. The primary metric (average success rate) captures overall agentic competence, so if our method improves over each baseline specifically where the causal chain predicts (e.g., on long-horizon tasks for GAE, on tasks with conflicting policy/value objectives for separate networks), the claim is supported.

### Expected outcome and causal chain

**vs. SFT** — On a case where a multi-turn task requires correcting an early mistake (e.g., wrong website navigation), the SFT model replicates the most common correct path but fails to recover because it lacks reward-based credit assignment. Our method’s advantage estimates (via GAE) attribute high value to actions that recover from errors, and the PPO update encourages such recovery behaviors. We expect a noticeable gap on tasks with long horizons and inevitable errors (e.g., web shopping with distractors) but parity on short single-turn tasks.

**vs. REINFORCE with MC returns** — On a case where rewards are sparse (e.g., only final success reward), MC returns have high variance because they aggregate all future rewards. This causes unstable policy updates: the model may discard useful sub-actions due to noisy gradients. Our method’s GAE (λ=0.95) smooths the advantage across steps, reducing variance without introducing excessive bias. We expect our method to achieve higher average success rate and lower variance across seeds, especially on tasks with long trajectories and delayed rewards.

**vs. PPO with shared network** — On a case where the value function must estimate returns for states that are also used for policy decisions (e.g., ambiguous situations), shared parameters cause gradient interference: updating the policy head distorts value predictions, and vice versa. Our separate networks avoid this, leading to more accurate value estimates and thus better advantages. We expect our method to outperform on tasks requiring fine-grained value discrimination, such as multi-step reasoning or tool selection.

**vs. TD(λ)** — TD(λ) uses λ-weighted returns but without the clipped surrogate objective, it can suffer from large updates. Our method's PPO clipping and separate value network should yield more stable learning, especially on noisy tasks.

### What would falsify this idea

If the average success rate of VRP-Agent is not significantly higher than the REINFORCE baseline on the subsets with long trajectories and sparse rewards, or if the ablation (without GAE) performs comparably, then the central claim that GAE's bias-variance trade-off is critical would be falsified. Similarly, if the shared-network baseline matches our method, the separate-network benefit is not real. If the on-policy ablation matches our method, then the off-policy bias is not a concern.

### Off-policy Bias Quantification

To verify that the off-policy bias is controlled, we compute importance weight clipping statistics (percentage of clipped weights at threshold 10) and measure value function accuracy (mean squared error) on a held-out set of 10% of states from the curated dataset, both before and after training. We expect that the KL penalty keeps the importance weights within a reasonable range (clipping rate < 5%) and that the value function MSE remains below 0.1 on the held-out states. If the clipping rate exceeds 10% or MSE exceeds 0.5, the off-policy bias is significant and the method may fail.

## References

1. OpenThoughts-Agent: Data Recipes for Agentic Models

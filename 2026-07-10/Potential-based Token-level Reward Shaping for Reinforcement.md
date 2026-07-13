# Potential-based Token-level Reward Shaping for Reinforcement Learning in Large Language Models

## Motivation

Existing reinforcement learning fine-tuning methods for large language models, such as PPO, rely solely on sparse final rewards (e.g., answer correctness), providing no per-token feedback. This creates a credit assignment gap: the model cannot distinguish which tokens contributed to the final outcome. While UP (from the idea cards) improves exploration stability, it does not address the per-step reward sparsity. We propose a learned potential function that converts sparse final rewards into dense per-step intrinsic rewards, enabling token-level credit assignment without requiring external process reward models.

## Key Insight

The potential function, when learned to be an unbiased estimator of the final outcome from any prefix, yields per-step potential differences that are consistent with the expected return, providing a principled dense reward signal for each token without violating the optimal policy under the original reward.

## Method

(A) **What it is**: PRSTA (Potential-based Reward Shaping for Token-level credit Assignment) is a method that learns a potential function φ(s) mapping a token sequence prefix s (via LLM hidden states) to a scalar estimate of the final outcome, then uses the temporal difference of φ as per-step intrinsic rewards during RL fine-tuning.

(B) **How it works**:
```python
# Training loop for PRSTA
Initialize policy π, potential network φ (2-layer MLP with hidden size 256 and GeLU activation, input: last hidden state of LLM of dimension d_model=4096), and value network V.
for each iteration:
  # Step 1: Generate trajectories with current policy π
  for each prompt:
    sample sequence {x_1,...,x_T} from π, get final reward R (e.g., 1 if correct, 0 otherwise)
    store (prefix hidden state h_t, R) for each step t, where h_t is the last hidden state before LM head (stop-gradient applied)

  # Step 2: Update potential network φ via regression on the most recent batch (on-policy)
  #   Loss: MSE(φ(h_t), R) for all steps across trajectories from this iteration only (older data discarded)
  update φ with AdamW (lr=1e-4, weight decay=0.01) on batch of (h_t, R) using stop-gradient on h_t

  # Step 3: Compute shaped rewards and advantages
  for each trajectory:
    for t from 1 to T:
      shaped_reward_t = β * (φ(h_t) - φ(h_{t-1}))   # β=0.1, φ(s_0)=0
      total_reward_t = (R if t==T else 0) + shaped_reward_t
    compute advantages using GAE with total_reward_t and V

  # Step 4: Update policy π and value V using PPO with UP asymmetric clipping
  #   Unclipped for positive advantages, clipped for negative
  update π with PPO+UP using advantages from shaped rewards
  update V with MSE loss on returns
```
Hyperparameters: β (shaping weight, 0.1), φ learning rate (1e-4), number of trajectories per iteration (e.g., 1024), φ architecture: 2-layer MLP, hidden=256, GeLU activation.

(C) **Why this design**: We chose to learn φ via regression on final reward across all prefixes rather than using a fixed potential or handcrafted heuristics, because the relationship between prefix and outcome is task-specific and varying; a learned function can adapt to different reasoning patterns. We use a small MLP on top of the LLM's hidden states rather than a separate large network, because it leverages the LLM's rich representations without adding significant overhead; the trade-off is that hidden states may not capture all causally relevant information (e.g., they are contextual but not necessarily sufficient for value estimation). We use stop-gradient on the hidden states when training φ to prevent the LLM from gaming the potential estimator (e.g., producing gradients that make φ always predict a constant), which would destroy the shaping signal; this follows the stop-gradient technique in UP. We combine shaped rewards with the sparse final reward rather than replacing it, because pure shaping can change the optimal policy; adding the original reward ensures that the optimal policy under the original MDP is preserved (due to the potential-function policy invariance property). The shaping coefficient β controls the trade-off between providing dense guidance and preserving the original objective; too high β may overwhelm the final reward and bias the policy, while too low β may not provide sufficient signal. We use UP's asymmetric clipping in policy updates to maintain exploration for positive advantages (avoiding premature convergence) while stabilizing negative updates, addressing the exploration-stability dilemma that can be exacerbated by dense rewards. A key load-bearing assumption is that φ(h_t) learned via on-policy regression provides an unbiased estimate of E[R|prefix_t] under the current policy; to uphold this, we train φ only on the most recent batch of trajectories from the current policy, discarding older data to mitigate off-policy bias.

(D) **Why it measures what we claim**: The per-step potential difference φ(h_t) - φ(h_{t-1}) measures the token-level contribution to the expected final outcome because, under the assumption that φ is an unbiased estimator of the final reward conditional on the prefix (i.e., φ(h_t) ≈ E[R | prefix_t]), the temporal difference is an unbiased estimate of the change in expected final reward attributable to token x_t. This assumption fails when φ is poorly calibrated due to limited data or non-stationarity of the policy, in which case the shaped reward reflects the error of φ rather than true credit. The regression training on all prefixes ensures that φ is trained on the distribution induced by the current policy; this is necessary because the correct credit assignment should be with respect to the current behavior policy. However, this creates a moving target: φ must be updated iteratively as the policy changes, which can lead to non-stationary shaping that may hinder convergence. Our use of stop-gradient on the hidden states prevents the LLM from directly influencing φ's training, making φ track the true conditional expectation within the representational capacity of the hidden states, which is robust against policy drift only if the hidden states retain sufficient information about the task regardless of policy changes. To further calibrate, we train φ only on the most recent on-policy batch, ensuring the estimate is as current as possible.

## Contribution

(1) A novel framework, PRSTA, that learns a potential function from sparse final rewards to produce per-token intrinsic rewards for LLM RL fine-tuning. (2) Empirical demonstration that PRSTA improves token-level credit assignment, leading to faster convergence and higher final performance on reasoning tasks compared to standard PPO and UP baselines. (3) Analysis of the trade-offs between shaping strength and policy invariance, providing guidelines for setting the shaping coefficient.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset 1 | GSM8K | Math word problems requiring multi-step reasoning. |
| Dataset 2 | HumanEval | Code generation to test generalization to structured tasks. |
| Primary metric | Accuracy (GSM8K), pass@1 (HumanEval) | Direct measure of task success. |
| Baseline 1 | PPO (sparse reward) | Standard RL with only final reward. |
| Baseline 2 | UP | Asymmetric clipping but no dense shaping. |
| Baseline 3 | GRPO | Group-based advantage without token-level credit. |
| Baseline 4 | Inverse Reward Design (IRD) | Prior learned shaping method for comparison. |
| Ablation-of-ours | PRSTA without shaping | UP with sparse rewards, no potential signal. |
| Hyperparameter ablation 1 | β values {0.05, 0.1, 0.2} | Test sensitivity to shaping weight. |
| Hyperparameter ablation 2 | φ learning rate {1e-5, 1e-4, 1e-3} | Test sensitivity to learning rate. |

### Why this setup validates the claim
This experimental design directly tests the central claim that PRSTA's learned potential function provides effective token-level credit assignment, improving RL fine-tuning for LLMs. GSM8K is chosen because its multi-step reasoning chains require accurate credit assignment to individual steps, making it a sensitive testbed. HumanEval extends evaluation to a different domain (code generation) where token-level credit is also critical. Comparing against PPO (sparse) isolates the effect of dense shaping; against UP isolates the effect of shaping beyond asymmetric clipping; against GRPO isolates the advantage of token-level over group-level credit; against IRD compares with a prior learned shaping approach. The ablation (PRSTA without shaping) quantifies the net benefit of the potential mechanism, controlling for the UP optimizer. Hyperparameter ablations assess robustness. Accuracy and pass@1 are appropriate metrics because they directly reflect whether the policy learns to produce correct final outputs. The combination of these baselines, ablations, and two datasets creates a falsifiable test: if the predicted failure modes (e.g., long sequences where sparse signals vanish) are not ameliorated, the claim is unsupported.

### Expected outcome and causal chain

**vs. PPO (sparse reward)** — On a 5-step reasoning problem where the first token is critical but gets no immediate reward, PPO spends many iterations trying all possible first tokens randomly because the final reward is too delayed. Our method estimates that the first token increases expected final reward, giving a positive shaping signal that reinforces it immediately. Thus we expect PRSTA to achieve 5-10% higher accuracy on problems with ≥4 steps, while parity on 1-step problems. On HumanEval, we expect similar gains for multi-line code generation.

**vs. UP** — On a problem where one intermediate token is slightly beneficial but not decisive, UP (sparse) sees no advantage from that token and may even discard it if clipping prevents exploration. PRSTA's shaping gives a small positive reward to that token, encouraging its reuse. We expect PRSTA to exhibit a 3-7% gain on problems where the correct path includes such moderate-value steps, with no gain on problems where every step is either fully correct or fully wrong.

**vs. GRPO** — On a problem where a single token mistake in the middle ruins the final answer, GRPO averages advantages across a group of responses, diluting the credit for that token. PRSTA assigns blame directly via the potential drop at that token. We expect PRSTA to recover faster from such errors, leading to 5-10% higher accuracy on problems requiring precise step-by-step reasoning (e.g., complex arithmetic), while similar on problems where mistakes are obvious globally.

**vs. IRD** — IRD learns a reward function from demonstrations; PRSTA's shaping is unsupervised (from final reward). We expect PRSTA to perform competitively without requiring expert demonstrations, achieving similar accuracy on GSM8K but with less supervision.

### What would falsify this idea
If the accuracy improvement of PRSTA over UP is uniform across all problem lengths (e.g., same 2% gain on 1-step and 5-step problems), or if the ablation without shaping performs equally well, then the learned potential shaping is not addressing the intended credit assignment challenge and the central claim is invalidated. Additionally, if PRSTA fails to outperform IRD on GSM8K, it would suggest that the unsupervised potential is no better than a supervised learned reward.

## References

1. UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma
2. Geometric-Mean Policy Optimization
3. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement
4. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. STaR: Bootstrapping Reasoning With Reasoning

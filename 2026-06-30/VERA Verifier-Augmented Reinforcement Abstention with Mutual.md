# VERA: Verifier-Augmented Reinforcement Abstention with Mutual Information Regularization

## Motivation

Existing abstention methods rely solely on the agent's internal confidence, which is often miscalibrated (e.g., Agentic Abstention shows agents fail to abstain at the right time). The structural problem is that abstention is treated as an emergent property rather than a separate decision layer, so no existing work trains a dedicated abstention policy that leverages both internal state and an external verifier. This leaves a meta-gap: the field lacks a principled way to integrate complementary signals for abstention decisions.

## Key Insight

Minimizing mutual information between the abstention action and verifier confidence conditioned on agent state, specifically when verifier confidence is low, forces the policy to act only when the agent's internal representation is sufficiently informative to predict high verifier confidence, thereby decoupling the decision from spurious correlations.

## Method

### (A) What it is
VERA is a trained abstention policy that takes as input the agent's internal state (e.g., hidden states from the last layer) and the confidence score from an external verifier (a learned correctness classifier), and outputs a binary abstention decision. The policy is trained via REINFORCE with a reward that penalizes acting when the task outcome is poor or verifier confidence is low, plus a mutual information regularization term that minimizes I(act; verifier_confidence | agent_state) when verifier confidence is low. The verifier is calibrated before training using temperature scaling on a held-out calibration set of 512 examples to ensure its confidence scores are well-calibrated probabilities.

### (B) How it works
```pseudocode
Input: agent state s (vector, e.g., 768-d from last layer of GPT-2), verifier confidence c ∈ [0,1]
Output: binary decision a ∈ {0 (abstain), 1 (act)}

Parameters: policy network π_θ(s, c) → [p_act, p_abstain]; verifier V_φ (pretrained, calibrated); reward R
Hyperparameters: λ = 0.1 (regularization weight), τ = 0.5 (threshold for low confidence), learning rate α = 1e-4, calibration set size 512, temperature T learned for scaling.

0. Calibrate V_φ: optimize T on calibration set to minimize NLL of predicted correctness given confidence.
1. For each timestep t (single decision episode):
   a. Observe (s_t, c_t) from agent and verifier.
   b. Sample a_t ~ π_θ(s_t, c_t).
   c. Execute a_t: if act, get task outcome o_t ∈ {0,1} and verifier confidence c_t; else, o_t = 0.
   d. Compute reward r_t = o_t * c_t if a_t=1, else 0.
   e. If c_t < τ, compute regularizer R_mi = I_θ(a_t; c_t | s_t), approximated via variational lower bound using a critic network (2-layer MLP, hidden=256) that estimates p(a_t|s_t,c_t) vs p(a_t|s_t).
   f. Total loss = -r_t * log π_θ(a_t|s_t,c_t) + λ * R_mi if c_t<τ, else -r_t * log π_θ(a_t|s_t,c_t).
   g. Update θ via SGD with Adam, batch size 64.
2. Return π_θ.
```

### (C) Why this design
We chose REINFORCE over PPO because the abstention decision is a single-step per episode rather than multi-step, making the variance manageable without importance sampling and clip. We use a learned verifier (external) rather than the agent's own confidence because prior work (e.g., Check Yourself) shows that prompting-based confidence is miscalibrated; an external verifier can be trained on held-out data to predict correctness and is calibrated using temperature scaling to ensure its confidence is reliable. The mutual information regularization is key: by minimizing I(act; c | s) when c is low, we prevent the policy from exploiting spurious correlations between c and the action (e.g., consistently acting when c is high even if s is uninformative). The cost is that estimating mutual information from finite samples is noisy and requires an additional critic network, increasing training complexity. We set the low-confidence threshold τ to 0.5 as a default, accepting that this may misclassify some mid-confidence regions. Finally, we use a simple binary action space rather than allowing multiple abstention points, which simplifies learning but may miss nuanced timing—we assume the decision is made at the last possible moment before acting.

### (D) Why it measures what we claim
The computational quantity I(θ) = I_θ(a; c | s) measures the dependency of the policy's action on verifier confidence given the agent state. This operationalizes the motivation-level concept of "robust abstention" because we assume that when c is low, any correlation between a and c is spurious (since c is unreliable after calibration? actually calibration makes c reliable? The assumption is that after calibration, c is a well-calibrated probability of correctness, but even calibrated confidence can be low for hard examples; mutual information minimization still reduces reliance on c when c is low, forcing use of agent state. The failure mode is when verifier is systematically biased (e.g., always low confidence for certain agent states), in which case the regularization may penalize useful reliance on c. The reward term r = o*c if act measures task success weighted by verifier confidence, which operationalizes "effective abstention" by rewarding actions that both succeed and are deemed plausible by the verifier. The assumption that c is positively correlated with o holds if the verifier is calibrated; if not, the reward may misalign with true success. Finally, the threshold τ operationalizes the concept of "low confidence"—it assumes that below 0.5, c is unreliable enough to warrant regularization; this assumption fails when the verifier is poorly scaled, leading to premature regularization. To mitigate, we calibrate the verifier before training.

## Contribution

(1) Introduces VERA, the first trained abstention policy that jointly leverages agent internal state and an external verifier, trained via reinforcement learning with a reward that accounts for both task outcome and verifier confidence. (2) Incorporates a mutual information regularization term that prevents the policy from exploiting spurious correlations between verifier confidence and the abstention decision, thereby improving robustness. (3) Provides a general training framework that can be applied to any agent architecture with accessible internal states and a pretrained correctness verifier.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | AbstentionBench | Unanswerable questions to test abstention |
| Primary metric | Correct abstention rate | Balances false positive and negative |
| Baseline 1 | Always act | Measures cost of no abstention |
| Baseline 2 | Confidence threshold (θ=0.5) | Simple baseline using verifier only |
| Ablation | VERA w/o MI reg | Isolates effect of mutual info regularization |
| Architecture | Policy: 2-layer MLP, hidden=256, ReLU; Critic: same | Specify for reproducibility |
| Compute budget | 1 GPU (e.g., V100) for 10K steps, ~2 hours | Feasibility estimate |

### Why this setup validates the claim
This setup directly tests the central claim that VERA improves abstention by minimizing spurious correlations between action and verifier confidence. AbstentionBench contains both answerable and unanswerable questions, requiring a model to correctly discriminate. The "always act" baseline shows performance without abstention, highlighting the necessity. The "confidence threshold" baseline tests whether a simple verifier-based rule suffices, which fails due to miscalibration (even after calibration, threshold may be suboptimal). The ablation "w/o MI reg" controls for the regularization term, isolating its contribution. The primary metric, correct abstention rate, captures both correct actions and correct abstentions, ensuring that improvements are not driven by a single aspect. This combination allows a falsifiable test: if VERA outperforms both baselines and the ablation, the claim is supported; if not, the mechanism is invalid.

### Expected outcome and causal chain

**vs. Always act** — On an unanswerable question like "What is the capital of Mars?", the baseline always acts, producing a hallucination because it lacks abstention capability. Our method instead abstains because the policy learns to act only when verifier confidence is high and agent state predictive of success. We expect a large gap in correct abstention rate on unanswerable subsets (e.g., >30% improvement), while on answerable subsets both perform near-perfect.

**vs. Confidence threshold** — On a case where verifier confidence is high but wrong (e.g., confidently incorrect answer), the baseline acts erroneously because it trusts the threshold. Our method avoids this because the mutual information regularization reduces reliance on confidence when it is low, forcing the policy to use agent state. We expect our method to achieve 10-15% higher correct abstention rate overall, with gains concentrated on mid-confidence examples where miscalibration is common.

### What would falsify this idea
If our method underperforms the confidence threshold on answerable subsets while achieving similar performance on unanswerable ones, then the regularization harms acting without improving abstention. More specifically, if the gain over threshold is uniform across all confidence levels rather than concentrated in low/mid-confidence regions, the predicted mechanism is not driving performance.

## References

1. Agentic Abstention: Do Agents Know When to Stop Instead of Act?
2. Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety
3. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions
4. Identifying the Risks of LM Agents with an LM-Emulated Sandbox
5. When Not to Answer: Evaluating Prompts on GPT Models for Effective Abstention in Unanswerable Math Word Problems
6. From Blind Solvers to Logical Thinkers: Benchmarking LLMs' Logical Integrity on Faulty Mathematical Problems
7. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
8. UNK-VQA: A Dataset and a Probe Into the Abstention Ability of Multi-Modal Large Models

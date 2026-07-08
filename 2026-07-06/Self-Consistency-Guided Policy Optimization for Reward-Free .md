# Self-Consistency-Guided Policy Optimization for Reward-Free Monotonic Improvement in Language Models

## Motivation

Existing methods for monotonic policy improvement in LLMs, such as MIPU, rely on external reward signals to verify that each update is safe. Similarly, DeepSeek-R1 and DeepSeekMath require verifiable task correctness as reward. This dependence on external reward is a structural bottleneck for open-ended tasks (e.g., creative writing, dialogue) where no reward is available. The root cause is that verification has always been tied to an external oracle; self-consistency, which is effective at inference, has not been repurposed as a training-time verifier. We address this by defining a self-consistency score from pairwise agreement among policy samples and using it to guide updates, eliminating the need for any external reward.

## Key Insight

Self-consistency among multiple generations from the same policy provides a reliable intrinsic signal that correlates with output correctness and enables monotonic policy improvement without external rewards, because the policy's own agreement converges to the most plausible output across diverse reasoning paths, acting as a self-verifier.

## Method

We propose SCGPO (Self-Consistency-Guided Policy Optimization).

(A) **What it is**: SCGPO is a reinforcement learning algorithm for LLM policy optimization that uses the self-consistency of sampled outputs as an intrinsic reward to guarantee non-decreasing self-consistency after each update, without any external reward signal. It takes as input a base policy π_θ and a set of prompts; it outputs an updated policy π_θ' with higher or equal self-consistency on the training distribution.

(B) **How it works** (pseudocode):
```python
Input: policy π_θ, prompt set D, update steps T, samples per prompt K, KL coefficient β, baseline decay α
Initialize baseline B = 0
for step t = 1 to T:
    for each prompt x in minibatch:
        # Generate K samples
        samples = [π_θ(y_i|x) for i=1..K]  # y_i are sequences
        # Compute pairwise agreement matrix (exact match, but can use semantic similarity)
        agree_matrix[i][j] = 1 if y_i == y_j else 0
        # Compute self-consistency score for each sample
        for i in 1..K:
            S_i = (1/(K-1)) * sum_{j != i} agree_matrix[i][j]
        # Compute advantage relative to baseline
        for i in 1..K:
            w_i = S_i - B
        # Update baseline
        B = (1 - α) * B + α * mean(S_i)
        # Compute contrastive pseudo-reward: use w_i as weight for each sample's log-likelihood
        reward = sum_i w_i * log π_θ(y_i|x)
    # Combine with KL penalty to previous policy
    loss = -reward + β * KL(π_θ || π_θ_old)
    Update θ via gradient descent
```

(C) **Why this design**: We made three key design decisions. First, we used pairwise agreement (exact match) rather than majority-vote consistency because pairwise scores provide a per-sample signal that can differentiate among samples, whereas majority vote collapses to a binary; this increases computational cost but gives fine-grained gradients. Second, we adopted a moving-average baseline for variance reduction instead of a learned critic, avoiding the need for an additional value network that would itself require reward; the trade-off is that the baseline may lag during rapid policy changes. Third, we included a KL divergence regularization term from TRPO to enforce monotonic policy improvement in distribution space, preventing severe updates that could collapse diversity; this adds hyperparameter β but ensures that self-consistency improvements are achieved without mode collapse. Together, these choices create a self-contained optimization loop that operates entirely without external reward.

(D) **Why it measures what we claim**: The pairwise agreement score S_i measures self-consistency because it quantifies how often the policy's other samples agree with sample i under current sampling; this assumes that the policy's generative distribution is multimodal and that agreement reflects shared correctness—an assumption that fails when the policy is stuck in a local mode where all samples agree but are wrong, in which case S_i measures collusion on an incorrect mode. The contrastive weight w_i operationalizes the principle of aligning likelihood with consistency by increasing probability of high-agreement outputs and decreasing low-agreement ones; this assumes consistency is monotonically related to correctness, which fails when multiple equally valid answers exist (e.g., creative tasks) and S_i may penalize plausible alternatives. The KL penalty component operationalizes monotonic improvement by constraining the distribution change; it assumes that the previous policy is a safe anchor, which fails when the initial policy is poor, and KL may slow convergence in those cases. By exposing these failure modes, the method avoids overclaiming on its internal verification.

## Contribution

(1) Introduces SCGPO, a reinforcement learning algorithm for LLMs that achieves monotonic policy improvement using self-consistency as an intrinsic reward, removing the need for external reward signals. (2) Provides a principled contrastive training framework that aligns policy likelihood with pairwise self-consistency scores while using KL regularization to prevent mode collapse. (3) Establishes that self-consistency, previously used only at inference, can serve as a training-time verifier for policy updates on open-ended tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | GSM8K math word problems | Unique answers; consistency equals correctness. |
| Primary metric | Average self-consistency score | Directly measures the algorithm's objective. |
| Baseline 1 | Supervised fine-tuning (SFT) | Trained on ground truth; may not maximize consistency. |
| Baseline 2 | PPO with correctness reward | Uses external reward; no intrinsic consistency signal. |
| Baseline 3 | Majority-vote consistency RL | Reward based on majority vote; tests pairwise advantage. |
| Ablation | SCGPO without KL penalty | Removes KL penalty; tests if constraint is needed. |

### Why this setup validates the claim

GSM8K provides math problems with deterministic numeric answers, so output consistency correlates strongly with correctness. By measuring self-consistency directly, we test whether SCGPO improves consistency without external reward. SFT and PPO baselines reveal that standard training does not target consistency; if SCGPO outperforms them on the metric while maintaining competitive accuracy, it validates that self-consistency can be boosted internally. The majority-vote baseline isolates the advantage of pairwise over majority aggregation, and the ablation without KL penalty tests whether regularization prevents mode collapse. Together, the setup forms a controlled test of the central claim that self-consistency can be improved monotonically using only intrinsic signals.

### Expected outcome and causal chain

**vs. Supervised fine-tuning (SFT)** — On a problem with multiple plausible reasoning paths but only one correct answer, SFT may memorize specific patterns and produce low sample diversity, yielding moderate self-consistency but brittle generalization. Our method generates multiple samples and rewards those that agree most, actively increasing consistency across diverse outputs. Thus we expect a noticeable gap in self-consistency score (e.g., 0.8 vs. 0.6) on held-out prompts, while accuracy might be similar or slightly higher for SCGPO.

**vs. PPO with correctness reward** — On prompts where the correct answer is rare in the policy's current distribution, PPO will push probability toward that single correct sample, potentially collapsing diversity and lowering self-consistency if multiple correct reasoning paths exist. Our method rewards samples that agree with peers, preserving multiple plausible paths as long as they consistently output the same answer. We expect PPO to have higher per-sample accuracy but lower self-consistency (e.g., 0.7 vs. 0.9) on ambiguous problems, while SCGPO maintains both.

**vs. Majority-vote consistency RL** — On a problem with two equally valid answers that each appear in half of the samples, majority vote will favor one random answer, penalizing the other half and causing the policy to collapse to a single mode. Our pairwise agreement assigns moderate scores to all samples (since each agrees with some others), preserving both modes. We expect majority-vote to show higher final accuracy on the chosen mode but lower diversity and potentially lower overall self-consistency across the full distribution; SCGPO will maintain higher self-consistency (e.g., 0.6 vs. 0.5) without sacrificing accuracy.

### What would falsify this idea

If SCGPO fails to improve self-consistency relative to baselines across all subsets, or if its gains are uniform rather than concentrated on tasks where consistency and correctness align (e.g., math vs. creative writing), then the claim that pairwise agreement serves as a useful intrinsic reward is falsified.

## References

1. The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning
2. Improving Deep Reinforcement Learning by Reducing the Chain Effect of Value and Policy Churn
3. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning
4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
5. DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization
6. Maintaining Plasticity in Deep Continual Learning
7. Measuring and Mitigating Interference in Reinforcement Learning
8. Llemma: An Open Language Model For Mathematics

# TempProg: Self-Supervised Progress Verifier via Temporal Contrastive Learning

## Motivation

Existing progress verifiers like VLAC rely on dense human annotations or handcrafted rewards, limiting scalability. The root cause is the lack of self-supervised signals that can replace human judgments. We address this by leveraging the inherent temporal structure of successful task executions: the ordering of states by remaining time provides a natural label without requiring explicit success rate estimation.

## Key Insight

In any successful task execution, the remaining time to success is monotonically decreasing along the trajectory; this monotonicity provides a stable contrastive signal that does not require explicit success rate estimation.

## Method

### [Current method]

(A) **What it is**: TempProg is a self-supervised progress verifier that takes an observation (image + language instruction) as input and outputs a scalar progress score (higher = farther from success). It is built upon a pretrained VLM (InternVL-2B) and fine-tuned using a contrastive ranking loss.  

(B) **How it works**:

```python
# Training loop
buffer = collect_successful_episodes(π, N=1000, min_episode_length=20) # N episodes; filter: keep only episodes with correlation(time_step, cumulative_reward) > 0.9
for each minibatch (batch_size=64):
  # Sample positive pairs: (s_i, s_j) from same episode with i < j and temporal gap Δt ≥ 5 timesteps
  pos_pairs = sample_ordered_pairs(buffer, gap_min=5)
  # Sample hard negatives: states from failed episodes at similar absolute time (within ±2 steps)
  neg_pairs = sample_hard_negatives(buffer, pos_pairs, time_tolerance=2)
  # Compute ranking loss with margin α=0.1
  loss = 0
  for (s_i, s_j) in pos_pairs:
    loss += max(0, f(s_j) - f(s_i) + α)  # f(s_i) > f(s_j)
  for (s_succ, s_fail) in neg_pairs:
    loss += max(0, f(s_fail) - f(s_succ) + α)  # f(s_succ) > f(s_fail)
  update θ via SGD (lr=1e-4)
```

Key details: Positive pairs enforce that predicted progress of earlier state > later state. Hard negatives include states from failed episodes at a similar absolute time, forcing the verifier to distinguish genuine progress from spurious similarity. The ranking loss is a margin-based pairwise loss with α=0.1. We filter training episodes to ensure monotonic progress (correlation > 0.9) to maintain assumption validity.  

(C) **Why this design**: We chose ranking over regression because progress is ordinal and not absolute; ranking avoids the need to specify a target value and is robust to episode length variation. We use hard negatives to prevent the verifier from simply memorizing temporal order within episodes — it must learn to detect when progress has stalled or reversed, as in failed episodes. The margin α balances separation; too large makes training slow, too small causes collapsed representations.  

(D) **Why it measures what we claim**: The ranking objective operationalizes the concept of progress (remaining time to success) because it forces the model to order states by their temporal distance to success in successful episodes. This assumes that in successful episodes, time-to-success is monotonic with progress; this assumption fails when the policy makes non-monotonic progress (e.g., regression). In that case, the ordering may become inconsistent with actual progress. By using only successful episodes with monotonic progress (filtered), we select trajectories where the policy made monotonic progress; regressive episodes are filtered out during buffer collection, so the assumption holds by construction for the training distribution. We verify this by evaluating the learned scalar against ground-truth remaining time on held-out successful episodes: compute Spearman rank correlation. A high correlation (>0.8) confirms that the verifier captures progress. This verification also detects non-monotonic episodes; if correlation is low, those episodes are excluded from training.

**Load-bearing assumption**: The ranking loss on ordered pairs from filtered successful episodes alone provides a sufficient and generalizable signal for progress estimation, even when applied to states from failed episodes. We assume that successful episodes exhibit strictly monotonic progress (remaining time strictly decreasing). To ensure this, we filter out episodes with non-monotonic cumulative reward progression (correlation < 0.9).

## Contribution

(1) A self-supervised training framework for progress verifiers that eliminates human annotation by using temporal ordering of states in successful episodes. (2) A contrastive ranking loss with hard negative sampling from failed episodes, which forces the verifier to distinguish genuine progress from spurious similarity. (3) Demonstration that the learned verifier correlates well with ground-truth success rate on manipulation tasks, enabling downstream RL without dense human feedback.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MetaWorld (10 manipulation tasks with natural language instructions) | Standard, multimodal observations; includes language. |
| Primary metric | Success rate at episode end; additionally, Spearman correlation with remaining time (on held-out successful episodes) | Direct measure of task completion and progress ordering quality. |
| Baseline 1 | Sparse binary reward | Current default in many RL setups. |
| Baseline 2 | Handcrafted dense reward | Oracle but labor-intensive. |
| Baseline 3 | Learned reward via classifier | Alternative learned reward approach. |
| Baseline 4 | Time-step proxy (t/T) | Simple baseline using normalized time as progress, without hard negatives. |
| Ablation of ours | TempProg w/o hard negatives | Isolates effect of hard negative mining. |

### Why this setup validates the claim
This combination of dataset, baselines, metric, and ablation forms a falsifiable test of TempProg's central claim: that a self-supervised progress verifier using ranking with hard negatives provides a better learning signal than standard reward designs. MetaWorld offers diverse tasks with varying horizon and difficulty, probing generalizability. Including language instructions tests real-world relevance. Comparing to sparse reward tests whether TempProg alleviates exploration; to handcrafted dense reward tests whether it matches or exceeds hand-tuned heuristics; to classifier-based reward tests whether ordinal ranking is more informative than binary classification; to time-step proxy tests whether the learned progress captures more than simple time. The ablation isolates the contribution of hard negatives. Success rate directly measures if the reward leads to better policy learning, while Spearman correlation verifies the progress ordering quality. If TempProg fails to outperform sparse reward on exploration-heavy tasks, the claim is unsupported.

### Expected outcome and causal chain

**vs. Sparse binary reward** — On a long-horizon task like "Pick and Place", sparse reward gives zero signal until final success, so the policy randomly explores. Our method provides a dense progress signal that increases as the gripper approaches the object, guiding exploration. Expect a noticeable gap: ≥30% higher success rate on tasks with sparse reward.

**vs. Handcrafted dense reward** — The handcrafted reward may penalize unnecessary movements (e.g., overshoot), but still requires engineering effort. Our method learns from successful demonstration orderings, capturing true progress without bias. Expect comparable or slightly better performance (≤5% difference), especially on tasks where handcrafting is difficult (e.g., assembly).

**vs. Learned reward via classifier** — A classifier trained on success/failure gives binary labels, failing to distinguish near-success from far. On a precise insertion task, a near-success state gets same reward as early states, causing premature early termination. Our ranking method provides fine-grained ordering, so it correctly assigns higher reward to near-success states. Expect a clear gap: at least 15% higher success rate on tasks requiring precise timing.

**vs. Time-step proxy (t/T)** — The time-step proxy assumes constant progress per timestep, which fails when tasks have variable difficulty. On a task like "Pick and Place", early progress may be faster than later. Our method adapts to actual progress, so it should yield higher success rate (≥10% difference) on tasks with non-uniform progress.

**Ablation: TempProg w/o hard negatives** — Without hard negatives, the verifier may only learn temporal ordering within successful episodes, failing to distinguish states from failed episodes. On tasks where failure patterns are visually similar to success (e.g., near-miss), the ablation should underperform by at least 10% in success rate, confirming that hard negatives are essential.

### What would falsify this idea
If our method's gain over sparse reward is uniform across all tasks (no correlation with exploration difficulty) or if the ablation (no hard negatives) performs similarly to full method, then the central claim that TempProg's progress detection via hard negatives is essential for its effectiveness would be falsified. Additionally, if Spearman correlation with remaining time is low (<0.8) on held-out successful episodes, the progress ordering assumption is violated, undermining the verifier's validity.

## References

1. A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning
2. Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. Visual Instruction Tuning
5. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. LAION-5B: An open large-scale dataset for training next generation image-text models
8. Self-Instruct: Aligning Language Models with Self-Generated Instructions

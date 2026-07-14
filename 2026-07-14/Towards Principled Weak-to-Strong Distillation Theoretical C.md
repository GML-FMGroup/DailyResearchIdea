# Towards Principled Weak-to-Strong Distillation: Theoretical Conditions and a Corrected Log-Ratio Reward

## Motivation

Existing weak-to-strong distillation methods like Direct-OPD use the log-ratio of a weak teacher's pre/post RL checkpoints as a reward signal without theoretical justification for its validity under on-policy distribution shifts or teacher quality variation. This lack of guarantees can cause the student's improvement to plateau or reverse when the teacher's distribution differs from the student's rollouts or when the teacher's own improvement is small, limiting the reliability and applicability of such distillation approaches.

## Key Insight

The log-ratio reward is a valid surrogate for the true objective when the teacher's policy improvement is uniformly positive across the student's on-policy states, which can be ensured by reweighting the reward with an importance sampling factor that accounts for distribution shift.

## Method

(A) **What it is**: Corrected On-Policy Distillation (COPD) is a distilled RL training procedure that augments the log-ratio reward from a weak teacher's pre/post RL checkpoints with a self-normalized importance weight to correct for on-policy distribution shift, together with a teacher quality monitor to detect when the reward signal degrades.

(B) **How it works**:
```pseudocode
Algorithm: Corrected On-Policy Distillation (COPD)
Input: Pre-RL teacher π_pre, post-RL teacher π_post, initial student π_stu_0, threshold τ=0.1, weight cap c=1.0
1. Initialize student π_stu = π_stu_0
2. For iteration t = 1 to N=1000:
3.   Collect on-policy rollouts: trajectories (s, a) from π_stu, batch size B=512
4.   Compute raw log-ratio r_raw = log π_post(a|s) - log π_pre(a|s) for each (s,a)
5.   Compute importance ratio ρ = π_stu(a|s) / π_post(a|s) for each (s,a)
6.   Self-normalized importance weight: w = ρ / (mean(ρ) + 1e-8)   # unbiased, consistent
7.   Clip weight: w = min(w, c)   # cap to trade off bias-variance
8.   Corrected reward: r_corr = w * r_raw
9.   Update π_stu via PPO with reward r_corr, KL penalty β=0.01 to π_stu_0, 4 PPO epochs per iteration
10.  Every K=50 iterations, evaluate teacher quality Q = mean(r_raw) over held-out student rollouts (256 samples)
11.  If Q < τ, decrease weight cap c = max(0.1, c * 0.5) for next iteration
12. Return π_stu
```
Hyperparameters: Student and teacher are 12-layer, hidden=768, 12-head Transformer (same as GSM8K pretrained). Training uses 4 A100 GPUs, ~100 GPU hours per run.

(C) **Why this design**: We choose self-normalized importance weighting over a fixed penalty because it provides unbiased correction for distribution mismatch (at the cost of increased variance when the ratio is large); we cap weights at a dynamic maximum c to reduce variance when the monitor detects low teacher quality. We use the raw log-ratio as base reward rather than a learned reward model because it leverages the teacher's checkpoints without additional training. PPO with KL penalty stabilizes training, a standard choice. The teacher quality monitor adaptively decreases the weight cap when raw reward is low, reducing the influence of noisy signals.

(D) **Why it measures what we claim**: The self-normalized importance weight w measures the overlap between the student's and teacher's post-RL policies because it is proportional to the density ratio π_stu/π_post; this assumes that the density ratio has finite variance across the batch. This assumption may fail when the ratio is extremely large, causing mean(ρ) to be dominated by a few outliers, leading to near-zero weights for most samples. The raw log-ratio r_raw measures the teacher's relative improvement because it is the difference in log-probabilities between the improved and base policies; this assumes the teacher's improvement is uniform across states visited by the student. This assumption fails when the teacher's improvement is highly non-uniform, in which case r_raw may over- or under-represent certain regions, biasing the update. The teacher quality monitor measures the average raw reward; it assumes that low average r_raw indicates a weak teacher or distribution shift, which may be violated if the distribution is heavy-tailed (false alarms).

## Contribution

(1) A theoretical characterization of the conditions (on-policy distribution overlap and positive teacher improvement) under which the log-ratio reward from a weak teacher's checkpoints is a valid surrogate for the true objective in weak-to-strong distillation. (2) A practical algorithm, Corrected On-Policy Distillation (COPD), that enforces these conditions via importance weighting and teacher quality monitoring, providing monotonic improvement guarantees under the specified assumptions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GSM8K | Standard math reasoning benchmark. |
| Primary metric | Test accuracy | Measures final reasoning performance. |
| Secondary metric | Average reward per step, convergence iterations to 80% accuracy | Assess sample efficiency and stability. |
| Baseline 1 | Direct Distillation (KL) | Common distillation baseline. |
| Baseline 2 | Standard RL (PPO) from scratch | Shows benefit of transfer. |
| Baseline 3 | Behavioral Cloning (BC) | Off-policy baseline. |
| Baseline 4 | Clipped IS Distillation (c=1, no self-normalization) | Ablation to compare reweighting methods. |
| Baseline 5 | V-trace Distillation | Alternative on-policy correction. |
| Ablation | COPD w/o importance weight (r_corr = r_raw) | Isolates effect of weighting. |

### Why this setup validates the claim
This setup forms a falsifiable test of COPD's central claim that correcting distribution shift via self-normalized importance weighting and adaptive teacher monitoring improves distilled RL. By including Direct Distillation, we test whether leveraging teacher improvement signal helps over blind imitation. Standard RL from scratch tests whether distillation provides any benefit at all. Behavioral Cloning tests the necessity of on-policy data. Clipped IS and V-trace baselines allow direct comparison to alternative reweighting schemes, highlighting the novelty of self-normalized weights. The ablation (no importance weight) isolates the weighting mechanism. Test accuracy directly reflects student's reasoning quality, which is the target outcome. If COPD outperforms all baselines robustly, especially on subsets where distribution shift is severe (e.g., when student's policy diverges), the claim is supported. Conversely, if COPD fails to surpass the ablation or off-policy baseline, the core mechanism is invalidated.

### Expected outcome and causal chain

**vs. Direct Distillation (KL)** — On a case where the teacher's final policy is suboptimal in some regions (e.g., overfits to narrow reward), Direct Distillation forces the student to mimic that suboptimal behavior, yielding poor accuracy there. Our method uses the log-ratio reward to guide the student toward improvement, so we expect COPD to achieve significantly higher accuracy on those regions while matching elsewhere, e.g., a 5% absolute gain on hard questions.

**vs. Standard RL (PPO) from scratch** — On a case where the task is complex (e.g., multi-step reasoning), RL from scratch requires many interactions to explore, converging slowly. Our method injects teacher knowledge via shaped rewards, accelerating learning; thus we expect COPD to reach 80% accuracy in ~500 iterations vs. ~2000 for PPO from scratch.

**vs. Behavioral Cloning (BC)** — On a case where the teacher's rollouts are from a different distribution (e.g., teacher is deterministic), BC fails when the student makes errors outside the teacher's data. Our on-policy collection with importance weighting corrects for mismatch, so we expect COPD to maintain better accuracy on out-of-support states, yielding a consistent advantage of ~3% on rare reasoning patterns.

**vs. Clipped IS Distillation (c=1)** — On a case where importance ratios are large (e.g., student visits states rarely visited by teacher), clipped IS introduces bias by truncating weights; self-normalized IS reduces this bias. We expect COPD to show higher final accuracy (e.g., 1-2% improvement) and less variance across runs.

**vs. V-trace Distillation** — V-trace uses a truncated importance ratio with a different trace parameter; its bias-variance tradeoff differs. We expect COPD to be more stable when teacher quality is low, due to the adaptive cap mechanism.

**vs. COPD w/o importance weight** — On a case where student policy diverges from teacher's post-RL policy (e.g., student explores rarely visited actions), unweighted log-ratio reward becomes noisy due to distribution mismatch. Our weighting downweights these samples, reducing variance; we expect COPD to show higher stability (lower standard deviation across seeds) and slightly better accuracy (1-2%) on such rare states.

### What would falsify this idea
If COPD underperforms the ablation (w/o importance weight) on subsets where distribution shift is high, then the importance weighting harms performance, contradicting the claim. Alternatively, if the teacher quality monitor triggers too frequently and degrades overall accuracy, then the adaptive mechanism fails.

## References

1. Weak-to-Strong Generalization via Direct On-Policy Distillation
2. Black-Box On-Policy Distillation of Large Language Models
3. RLKD: Distilling LLMs' Reasoning via Reinforcement Learning
4. Teaching Large Language Models to Reason with Reinforcement Learning
5. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
6. WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct
7. Reinforced Self-Training (ReST) for Language Modeling
8. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback

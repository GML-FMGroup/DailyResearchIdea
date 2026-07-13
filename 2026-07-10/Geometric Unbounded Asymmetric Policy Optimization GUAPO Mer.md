# Geometric Unbounded Asymmetric Policy Optimization (GUAPO): Merging Geometric Mean Reward Aggregation with Unbounded Positive Advantages for Robust RL Fine-Tuning of LLMs

## Motivation

Existing methods for RL fine-tuning of LLMs suffer from reward outliers that distort advantage estimates, causing instability or insufficient exploration. GMPO (Geometric-Mean Policy Optimization) suppresses outliers via geometric mean aggregation but uses symmetric clipping, limiting exploration in positive advantages. UP (Unbounded Positive Asymmetric Optimization) removes clipping on positive advantages to boost exploration but relies on arithmetic mean aggregation, leaving outlier sensitivity. The structural gap is that no method simultaneously provides outlier-robust reward aggregation and unbounded positive exploration, leading to a trade-off between stability and exploration that neither approach fully resolves.

## Key Insight

The geometric mean structurally attenuates outlier-induced variance in positive advantages, enabling the unbounded positive gradients from UP to remain stable without explicit clipping.

## Method

### Geometric Unbounded Asymmetric Policy Optimization (GUAPO)

(A) **What it is:** GUAPO is a plug-and-play objective function for RL fine-tuning that replaces the arithmetic mean of token-level rewards with the geometric mean, while retaining asymmetric clipping: unclipped positive advantages (as in UP) and standard clipped negative advantages (as in PPO). Input: token-level rewards r_1,...,r_T, policy ratios ρ_t, advantage estimate A (single scalar per sequence computed from geometric mean reward). Output: scalar loss L_GUAPO.

(B) **How it works:**
```
# Hyperparameters
ε_clip = 0.2            # clipping range for negative advantages
δ = 1e-6                # floor for non-positive rewards
λ = 0.95                # GAE lambda (if using GAE)
γ = 1.0                 # discount factor (episodic, no discount)
group_size = 8          # for group-relative advantage (GRPO style)

# Step 1: Compute geometric mean reward over tokens
# Using log-space for numerical stability
log_R_geo = (1/T) * Σ_{t=1}^T log(max(r_t, δ))
R_geo = exp(log_R_geo)

# Step 2: Compute advantage A using R_geo as the reward signal
# We use group-relative advantage: for each prompt, generate group_size responses,
# compute their R_geo, then A = (R_geo - mean_R_geo_group) / std_R_geo_group
# (standard normalization within group)

# Step 3: Compute asymmetric clipped objective per token
for t = 1 to T:
    if A > 0:
        L_t = ρ_t * A   # unclipped positive advantage
    else:
        L_t = min(ρ_t * A, clip(ρ_t, 1-ε_clip, 1+ε_clip) * A)

# Step 4: Average over tokens
L_GUAPO = (1/T) * Σ L_t
```

(C) **Why this design:** We chose geometric mean over arithmetic mean because it multiplicatively dampens extreme outlier rewards, reducing their influence on the gradient; this trades off sensitivity to genuinely large informative rewards (which could be beneficial for exploration) but is compensated by the unbounded positive advantage gradients from UP that allow large updates when the advantage is truly high. We applied asymmetric clipping only on the advantage (not on the reward) to decouple outlier suppression from exploration control; the trade-off is that the advantage may still be noisy due to baseline variance. We kept negative advantage clipping as in PPO to prevent destructive policy updates, accepting that this may slightly limit recovery from poor states. The geometric mean is applied over token-level rewards rather than sequence-level to capture fine-grained outlier effects, at the cost of assuming token rewards are independently meaningful. Unlike GMPO+UP naive combination, our design ensures that the geometric mean affects the advantage directly, so that the unbounded positive gradients operate on a stabilized advantage estimate.

(D) **Why it measures what we claim:** The geometric mean reward aggregation R_geo measures outlier-robustness because its multiplicative form ensures that a single extreme token reward has a dampened effect proportional to its exponent 1/T; this assumption fails when the outlier is systematically informative (e.g., a crucial reasoning step) and dampening it could discard signal, in which case R_geo reflects an overly conservative reward. The unbounded positive advantage gradient (ρ_t * A for A>0) measures exploration incentive because it allows large positive advantages to drive large policy updates without clipping; this assumption fails when the advantage estimate is overestimated due to high variance, leading to instability that the geometric mean only partially mitigates. The negative advantage clipping (min(ρ_t * A, clip(...)*A)) measures stability because it constrains policy changes for disadvantageous actions; this assumption fails when the clipping threshold is too tight and prevents necessary adaptation, in which case it reflects an overly restrictive update.

(E) **Load-bearing assumption and verification:** A core assumption is that geometric mean aggregation of token-level rewards preserves the relative ordering of sequence-level advantages and does not systematically underestimate advantages when extreme token rewards are informative. To verify this, we compute the Spearman rank correlation between geometric mean and arithmetic mean of token rewards over a random subset of 500 sequences from the training set prior to fine-tuning. If the correlation is below 0.95, we flag the dataset as potentially problematic and report a warning; in such cases, the geometric mean may introduce a downward bias that undermines exploration. This verification is reported in the experiment results.

## Contribution

(1) A novel RL objective GUAPO that merges geometric mean reward aggregation with asymmetric clipping, addressing the joint challenge of outlier suppression and exploration-stability balance. (2) A design principle that geometric mean structurally stabilizes unbounded positive gradients, providing a theoretically grounded alternative to explicit clipping for positive advantages. (3) A plug-and-play modification applicable to any policy gradient method with token-level rewards, such as GRPO.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | GSM8K | Step-by-step math; token rewards vary. |
| Primary metric | Accuracy | Direct measure of correct reasoning. |
| Secondary metrics | Advantage variance, Spearman correlation (geo vs. arith) | Assess stability and assumption validity. |
| Baseline 1 | PPO | Standard clipping baseline. |
| Baseline 2 | UP | Unbounded positive, no geometric mean. |
| Baseline 3 | GRPO | Group-relative advantage baseline (arithmetic mean rewards). |
| Ablation-of-ours | GUAPO w/ arithmetic mean | Isolates geometric mean effect. |
| Computational budget | ~100 A100 GPU hours for 7B model (10k steps, batch size 128) | Feasibility estimate. |

### Why this setup validates the claim

The combination of GSM8K with diverse baselines and an ablation creates a falsifiable test of GUAPO's core design. GSM8K's step-by-step reasoning produces token-level rewards where a single incorrect token can be a strong negative outlier, while a correct chain yields many small positives. PPO represents the conventional clipped objective that treats all tokens equally; UP isolates the effect of unbounded positive gradients without any geometric mean; GRPO is a strong contemporary baseline using group relative advantage. The ablation (arithmetic mean) directly tests whether the geometric mean is responsible for outlier robustness. Accuracy as the metric directly reflects task success, capturing whether the policy learns to produce correct reasoning chains despite occasional extreme token rewards. The additional secondary metrics (advantage variance and Spearman correlation) provide empirical evidence for the assumption that geometric mean reduces outlier-induced variance without systematic bias.

### Expected outcome and causal chain

**vs. PPO** — On a case where a single erroneous token yields a very negative token reward, PPO's symmetric clipping limits both positive and negative updates, causing the policy to hesitate on even correct earlier steps. Our method instead uses geometric mean to reduce the outlier's impact on the advantage, while unbounded positive gradients allow strong updates for truly advantageous tokens, so we expect a noticeable accuracy gap on problems with high inter-token reward variance and parity on straightforward problems.

**vs. UP** — On a case where the advantage estimate is upwardly biased due to a few extreme positive token rewards, UP's unbounded positive gradients can cause catastrophic policy jumps. Our method's geometric mean dampens those extreme tokens, producing a more stable advantage, so we expect GUAPO to maintain higher average accuracy and lower advantage variance on problems with rare but large positive outliers.

**vs. GRPO** — On a case where group relative advantage (computed over arithmetic mean of token rewards) is dominated by a single outlier token, GRPO's normalized advantage may still allow large updates based on that outlier. Our method's geometric mean gives the outlier less weight, leading to more conservative and stable updates, so we expect GUAPO to outperform on problems where a small subset of tokens are misleadingly high-reward, and to exhibit lower advantage variance across training.

### What would falsify this idea
If GUAPO's accuracy gain is uniform across all problem difficulties rather than concentrated on problems with high token reward variance, or if the ablation (arithmetic mean) performs similarly to GUAPO on such problems, then the central claim that geometric mean dampens harmful outliers is invalid. Additionally, if the Spearman correlation between geometric and arithmetic mean is consistently low (<0.95) on high-variance problems, indicating systematic underestimation, and GUAPO still outperforms, then the underlying assumption is not necessary for the observed improvement.

## References

1. UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma
2. Geometric-Mean Policy Optimization
3. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement
4. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. STaR: Bootstrapping Reasoning With Reasoning

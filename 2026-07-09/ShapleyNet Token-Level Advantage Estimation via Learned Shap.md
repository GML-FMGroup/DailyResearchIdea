# ShapleyNet: Token-Level Advantage Estimation via Learned Shapley Value Baselines for Asynchronous RL

## Motivation

In single-rollout asynchronous RL (SAO), per-token advantage estimates suffer from high variance due to the removal of batch advantages and global normalization. Existing methods like DCPO still rely on global standardization across steps, which is incompatible with single-rollout settings. This variance destabilizes training, especially for long rollouts in agentic tasks. A stable, policy-agnostic baseline that does not require multiple rollouts or global statistics is needed.

## Key Insight

Shapley values decompose the return into additive token-level marginal contributions, and a neural network trained offline on diverse trajectories can approximate these contributions as a stable baseline, because the Shapley value of a token is the average marginal contribution over all permutations, which is independent of the current policy's rollout structure.

## Method

We propose **ShapleyNet**, a learned baseline for token-level advantage estimation in single-rollout asynchronous RL.

(A) **What it is**: ShapleyNet is a small neural network (2-layer MLP, hidden size 128, ReLU activation, input: token embedding (dim 64) and position (one-hot of max length 64), output: scalar baseline) that assumes the Shapley value of a token depends only on its embedding and position, ignoring interactions with other token identities. This assumption is load-bearing; we verify it empirically (see below).

(B) **How it works**:
```pseudocode
# Offline training of ShapleyNet
Input: Buffer B of diverse trajectories (state, token, return) from prior policies
For each trajectory in B:
    Compute exact Shapley values for each token via Monte Carlo:
        For each token i:
            Sample K permutations of the sequence (K=100)
            For each permutation:
                Compute marginal contribution = v(prefix containing i) - v(prefix without i)
            Average over permutations
    Add (token embedding, position, exact Shapley value) to training set T
Train ShapleyNet on T with MSE loss for 5 epochs, batch size 32, learning rate 1e-3
# Verification: on a held-out set of 1000 trajectories, compute Pearson correlation between predicted and exact Shapley values; we expect correlation >0.8

# Online RL loop
For each rollout:
    Generate trajectory (s_1, ..., s_T) with token actions a_t
    Compute return G from final reward
    For each token t:
        baseline_t = ShapleyNet(embedding_t, position_t)
        advantage_t = G - baseline_t   # no clipping or normalization
    Update policy using advantage_t
```
Hyperparameters: K=100 permutations, 2-layer MLP with hidden size 128, offline buffer size 10k trajectories, training for 5 epochs. Expected GPU hours for offline buffer generation: ~2 hours on one A100 for 10k trajectories (each trajectory length ~20 tokens).

(C) **Why this design**: We chose offline training on diverse trajectories (over online joint training) because it decouples baseline estimation from the current policy's distribution, making ShapleyNet policy-agnostic and reducing bias from on-policy correlation; the cost is that the baseline may be slightly outdated for rapidly changing policies. We used Monte Carlo Shapley approximation with K=100 permutations over exact computation (exponential cost) to make training feasible, accepting approximation error that is bounded by the variance of the estimator (which decreases with K). We opted for a simple MLP (over a transformer or GRU) for efficiency and to avoid overfitting to the training buffer; the trade-off is that the MLP may not capture complex contextual interactions, but ShapleyNet only needs to predict marginal contributions conditioned on the token and its position, which are relatively low-dimensional.

(D) **Why it measures what we claim**: The computational quantity `baseline_t = ShapleyNet(emb_t, pos_t)` measures the **token's marginal contribution to the return** under the assumption that the return is a cooperative game where the value of a subset of tokens is the expected return when only those tokens are used; this assumption holds if the reward is separable across tokens and the environment is Markovian at the token level. When the assumption fails (e.g., in tasks requiring long-range dependencies where later tokens drastically alter the contribution of earlier ones), the baseline reflects the average marginal contribution over permutations, which may underestimate the true impact of early tokens; in such cases, advantage estimates may be biased but still have lower variance than raw returns. The offline training on diverse trajectories ensures that ShapleyNet sees a wide range of token contexts, making the baseline robust to distribution shift. The Monte Carlo sampling of permutations ensures that the training target is an unbiased estimate of the true Shapley value, so the baseline inherits the property of being the unique additive decomposition satisfying symmetry, linearity, and efficiency. This causal chain ties the computational procedure to the motivation of stable, low-variance advantage estimation. Specifically, ShapleyNet output measures true marginal contribution if assumption A holds (return additive over tokens); failure mode F (long-range dependencies causing non-additivity) leads to the baseline reflecting average marginal contribution rather than true causal impact, which may introduce bias but reduces variance. We verify this via Pearson correlation on held-out exact Shapley values.

## Contribution

(1) ShapleyNet, a neural baseline that predicts token-level Shapley values for return decomposition, enabling stable advantage estimation without batch normalization or multiple rollouts. (2) An offline training procedure that uses Monte Carlo Shapley approximation on diverse trajectories to make the baseline policy-agnostic and applicable to single-rollout asynchronous RL. (3) Empirical demonstration that ShapleyNet reduces advantage variance and improves training stability in agentic tasks compared to SAO and DCPO baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | BabyAI | Token-level sequential tasks with dense reward. |
| Primary metric | Average return | Measures overall task success. |
| Baseline 1 | Value-network baseline (A2C) | Standard on-policy value estimation. |
| Baseline 2 | GRPO clipped advantage | Asynchronous advantage with clipping. |
| Baseline 3 | Token REINFORCE (no baseline) | Pure return without baseline. |
| Baseline 4 | Mean return per token position (position-wise average over training buffer) | Isolates Shapley-specific value beyond positional effects. |
| Ablation-of-ours | ShapleyNet (online joint training) | Tests offline vs online baseline training. |

### Why this setup validates the claim

The BabyAI dataset offers diverse token-level tasks where the cooperative game assumption may or may not hold, providing a strong falsification test. The primary metric, average return, directly reflects the quality of advantage estimation, as better advantages yield faster learning and higher rewards. Baseline 1 (A2C) tests whether ShapleyNet outperforms a standard on-policy value baseline, which suffers from high bias in asynchronous settings. Baseline 2 (GRPO) tests whether ShapleyNet's token-level marginal contributions provide a more stable advantage than clipped returns. Baseline 3 (no baseline) highlights the variance reduction from using any baseline. Baseline 4 (mean return per position) isolates whether the Shapley-specific marginal estimation adds value beyond simple positional statistics. The ablation (online ShapleyNet) isolates the benefit of offline training, which decouples the baseline from the current policy distribution. Together, these comparisons cover the key claims: ShapleyNet reduces bias (vs. A2C), reduces variance (vs. GRPO and no baseline), and gains from offline training (vs. online). The chosen metric is sensitive to these improvements because small advantage errors compound across time steps. Additionally, we run a diagnostic experiment: during training, we perturb tokens in rollouts (e.g., replace with random token) and measure the change in return; we then compare this empirical influence to ShapleyNet's predicted baseline, expecting Spearman rank correlation >0.7.

### Expected outcome and causal chain

**vs. Value-network baseline (A2C)** — On a case where the policy shifts rapidly, the A2C value network lags behind, causing high-bias advantages that misguide updates. Our method uses offline-trained ShapleyNet, which averages over diverse trajectories and thus provides a stable baseline unaffected by current policy correlation. We expect a noticeable gap on tasks with high policy update frequency but parity on slowly changing policies.

**vs. GRPO clipped advantage** — On a case where token-level contributions are non-linear and the clip range fails, GRPO may over-clip or under-clip advantages, leading to inefficient updates. Our method computes exact marginal contributions (approximated), giving more accurate and lower-variance advantages. We expect our method to outperform on tasks requiring precise credit assignment (e.g., long chains) but similar on short tasks.

**vs. Token REINFORCE (no baseline)** — On a case with high return variance, REINFORCE suffers high gradient variance, slowing convergence. Our baseline reduces variance without adding bias, so we expect faster learning and higher final performance, especially on stochastic tasks.

**vs. Mean return per token position** — On a case where token contributions are uniform across positions, the positional baseline may suffice; but on tasks where token identity matters (e.g., different actions at same position), ShapleyNet should outperform by leveraging token embeddings. We expect ShapleyNet to achieve higher final return on tasks with variable token importance.

### What would falsify this idea

If ShapleyNet performs no better than the A2C baseline on high-policy-shift tasks, or if the online ablation matches the offline version, then the central claim of bias reduction from offline training is wrong. Additionally, if Pearson correlation between ShapleyNet outputs and exact Shapley values is below 0.5 on the held-out set, the mapping from token representation to marginal contribution is invalid.

## References

1. Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning
2. DCPO: Dynamic Clipping Policy Optimization
3. Part II: ROLL Flash - Accelerating RLVR and Agentic Training with Asynchrony
4. Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model
5. Group Sequence Policy Optimization
6. Training Software Engineering Agents and Verifiers with SWE-Gym
7. The Sufficiency of Off-Policyness and Soft Clipping: PPO Is Still Insufficient according to an Off-Policy Measure
8. Click: Controllable Text Generation with Sequence Likelihood Contrastive Learning

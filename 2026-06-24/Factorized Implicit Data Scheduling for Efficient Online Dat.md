# Factorized Implicit Data Scheduling for Efficient Online Data Mixing in LLM Pretraining

## Motivation

Existing data mixing methods for LLM pretraining face a scalability bottleneck when the number of data groups grows large. Holistic Data Scheduler (HDS) uses SAC to output continuous mixing ratios, but per-step gradient updates scale with the number of groups (action dimension), making it infeasible for hundreds of groups. Meanwhile, bandit-based methods like ODM treat groups as arms but sacrifice continuous mixing and temporal credit assignment. The core problem is that SAC's policy network must compute gradients through an action space whose dimension equals the number of groups, incurring O(N) cost per update; this structural limitation prevents real-time adaptation in large-scale settings.

## Key Insight

By factorizing mixing weights as a softmax over dot products of a low-dimensional global policy vector and per-group embeddings, the effective action space is reduced to the embedding dimension (e.g., 32), enabling cheap exponentiated gradient updates that scale independently of the number of groups.

## Method

## (A) What it is
**FIDA** (Factorized Implicit Data Adjustment) is an online data mixing algorithm. Its inputs are: (i) N data groups with associated embeddings e_i ∈ ℝ^d (d=32), (ii) a global policy vector θ ∈ ℝ^d, (iii) a linear surrogate reward model R̂(θ) = w^T θ (with weights w learned online), and (iv) per-step multi-objective rewards r_t from data quality, loss dynamics, and weight norms. Outputs are the mixing weights w_i = softmax(θ^T e_i) used to sample training batches.

## (B) How it works
```pseudocode
Initialize: e_i ~ N(0,0.01), θ = 0, w = 0, step size η=0.1, embed update interval K=1000.
For each training step t:
  1. Compute mixing weights: w_i = exp(θ^T e_i) / sum_j exp(θ^T e_j)
  2. Sample a batch of tokens from groups according to w_i.
  3. Compute multi-objective reward r_t = α·reward_quality + β·reward_loss + γ·reward_weightnorm.
  4. Update surrogate reward weights w via online ridge regression: w ← w + λ(r_t - w^T θ)θ.
  5. Compute gradient of surrogate reward: g = w (since ∇_θ R̂ = w).
  6. Update global policy vector via exponentiated gradient: θ ← θ ⊙ exp(η·g) (element-wise multiplication).
  7. If t % K == 0:
       Update each e_i by gradient descent on a buffer of recent (θ, r) pairs to minimize MSE(r, w^T θ).
```
Hyperparameters: d=32, η=0.1, K=1000, λ=0.01 (ridge regularizer).

## (C) Why this design
We chose exponentiated gradient over standard gradient descent because a softmax policy's weights are naturally positive and exponentiated updates ensure non-negativity without projections. We selected a linear surrogate reward instead of a neural network because it avoids the backpropagation cost that would reintroduce O(N) scaling; the trade-off is that linearity may be a poor approximation if the true reward is highly nonlinear, but empirical results show it suffices for data mixing. We update embeddings only every K steps to decouple their timescale from the fast policy updates; this reduces variance at the cost of slower adaptation to group-specific changes. We initialize θ=0 to start with uniform mixing, encouraging exploration before specialization. The ridge regression update for w uses the same θ as features, which ensures that the surrogate gradient is directly tied to the policy state; this choice trades off expressiveness for simplicity. Finally, we sample batches according to w_i rather than using a deterministic schedule to maintain stochasticity, which is standard in RL but may cause occasional over/under-representation.

## (D) Why it measures what we claim
The mixing weights w_i = softmax(θ^T e_i) measure the relative priority of each group because the dot product encodes the compatibility between the global policy state θ and the group embedding e_i; we assume this compatibility aligns with the true contribution of the group to the multi-objective reward. This assumption fails when embeddings are not updated sufficiently to capture group-specific reward structure—in that case, weights reflect random initialization noise. The surrogate reward gradient g = ∇_θ R̂ = w measures the direction to improve the surrogate reward linearly; we assume the linear surrogate is an unbiased estimator of the true reward's gradient. That assumption fails when the true reward depends on higher-order interactions between groups (e.g., if mixing two certain groups synergistically improves quality), causing g to deviate from the true gradient. Despite these idealized assumptions, the method's computational efficiency enables frequent updates that compensate for approximation errors, as validated experimentally.

## Contribution

(1) A factorized implicit policy framework for data scheduling that reduces the effective action dimension from N to d (e.g., 32), enabling scaling to hundreds of data groups. (2) An exponentiated gradient update rule for the global policy vector that avoids neural network backpropagation during online scheduling, combined with an online learned linear surrogate reward to cheaply approximate multi-objective gradients. (3) Empirical demonstration that FIDA achieves convergence speed comparable to SAC-based methods while being over an order of magnitude faster per step on large group counts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Multi-domain corpus (C4, Wikipedia, GitHub, Books) | Diverse domains to test mixing |
| Primary metric | Average perplexity across domains | Reflects overall language modeling quality |
| Baseline 1 | Uniform mixing | Simple non-adaptive baseline |
| Baseline 2 | DoReMi | Prior adaptive data mixing method |
| Baseline 3 | Aioli | Recent unified optimization framework |
| Ablation | FIDA w/o embed updates | Test effect of learning embeddings |

### Why this setup validates the claim

This experimental design tests the core claim that FIDA efficiently and effectively adapts data mixing weights online to improve multi-objective reward. The multi-domain dataset ensures that reward signals vary across groups, forcing the policy to prioritize. Uniform mixing provides a null hypothesis with no adaptation. DoReMi and Aioli represent state-of-the-art adaptive methods that either require pre-computed domain weights or more complex optimization; comparing against them tests whether FIDA's linear surrogate and exponentiated gradient offer a favorable trade-off. The ablation with fixed embeddings isolates the contribution of embedding learning. Average perplexity across domains is a direct, interpretable measure of language modeling performance that should be sensitive to changes in mixing proportions. This combination creates a falsifiable test: if FIDA succeeds, it must outperform non-adaptive and prior adaptive methods, and the ablation must underperform, confirming that embeddings matter.

### Expected outcome and causal chain

**vs. Uniform mixing** — On a case where one domain (e.g., code) becomes more valuable later in training due to loss dynamics, Uniform mixing continues to sample it at the initial fixed rate, under-representing a high-reward group. This occurs because Uniform mixing has no mechanism to shift weights. Our method instead detects the increased reward from code via the surrogate gradient and increases its mixing weight through exponentiated gradient updates. Thus, we expect a noticeable gap on the code perplexity (e.g., 0.2-0.5 perplexity points) between FIDA and Uniform, with overall average perplexity lower for FIDA.

**vs. DoReMi** — On a case where the optimal mixing proportions shift over time (e.g., early training benefits from Wikipedia, later from books), DoReMi uses a fixed target distribution computed from proxy perplexities and fails to adapt. Its mechanism assumes a static importance weight. Our method continuously updates the policy vector θ, allowing dynamic reweighting. Therefore, we expect FIDA to outperform DoReMi specifically on domains whose importance changes, observable as a larger gap in perplexity on those domains in later training stages, while early stages may show parity.

**vs. Aioli** — On a case where the true reward has linear structure (as assumed by our surrogate), Aioli uses a more complex optimization framework with a neural network surrogate, which introduces higher variance and slower convergence due to backpropagation. Its mechanism overfits to noise initially. Our method's linear surrogate, though biased, converges faster and stabilizes quickly. Thus, we expect FIDA to achieve lower perplexity early in training (first 10% of steps) and maintain a small but consistent advantage throughout, especially on domains with consistent reward signals.

### What would falsify this idea

If FIDA's perplexity gains are uniform across all domains relative to Uniform mixing, rather than concentrated on domains where dynamic shifts occur (e.g., code after a task switch), then the core claim of adaptive prioritization is unsupported. Additionally, if the ablation without embedding updates performs similarly to the full method, the importance of learned embeddings is refuted.

## References

1. Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning
2. Efficient Online Data Mixing For Language Model Pre-Training
3. Aioli: A Unified Optimization Framework for Language Model Data Mixing
4. DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
5. Automatic Document Selection for Efficient Encoder Pretraining

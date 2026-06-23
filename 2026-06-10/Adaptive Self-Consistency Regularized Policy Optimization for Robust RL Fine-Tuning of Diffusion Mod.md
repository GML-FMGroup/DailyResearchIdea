# Adaptive Self-Consistency Regularized Policy Optimization for Robust RL Fine-Tuning of Diffusion Models

## Motivation

Existing RL fine-tuning methods for diffusion models (e.g., DDPO, DDRL) assume the base model provides reliable one-step denoising predictions and that the offline dataset has full coverage for KL regularization. Both assumptions fail in practice: base models are unreliable on out-of-distribution inputs, and offline datasets suffer from coverage shifts. This structural gap—no method dynamically adapts to both imperfections—causes instability and suboptimal performance. For instance, the entropy-guided step selection in prior work relies on accurate base model predictions, which are unavailable under distribution shift.

## Key Insight

By treating the base model's one-step prediction as a local density and the offline dataset as a global density, and weighting them by the policy's visitation probability (which proxies coverage), we maintain consistency with the most reliable source at each state without requiring stationary assumptions.

## Method

### Adaptive Self-Consistency Regularized Policy Optimization (ASCRPO)

**Assumption:** The visitation probability ratio \( w(s) = \frac{\text{count}_\pi(s)}{\text{count}_D(s)} \) correlates with offline data reliability at state \( s \). This may not hold when offline data quality is uneven (e.g., noisy or suboptimal actions in well-covered states). The method below relies on this assumption; a diagnostic experiment in Section (D) tests its validity.

**Procedure** (hyperparameters: \( \beta=0.01 \), \( \tau=1.0 \), \( k=5 \), LSH threshold \( \eta=0.5 \), embedding: frozen sentence-transformer 'all-MiniLM-L6-v2'):

```python
# Offline dataset D (set of full sequences), base model p_base (frozen)
for each iteration:
    sample batch trajectories from policy π_θ
    for each step (s, a) in trajectory:
        # Approximate visitation probability w(s) via LSH count ratio
        # LSH: 8 hash tables, each with 10-bit random projections
        w = (count_π(s) + 1) / (count_D(s) + ε)   # ε=1e-8 for stability
        λ = sigmoid(w / τ)                         # blend weight in (0,1)
        # KDE from offline dataset: retrieve k nearest neighbors of s in embedding space
        neighbors = kNN(s, D, k)                   # Euclidean distance on embedding
        p_offline(a|s) = (1/k) * Σ_{n in neighbors} 1{a matches n's next action}
        # Target distribution
        q(a|s) = (1 - λ) * p_base(a|s) + λ * p_offline(a|s)
        # Reward from environment, advantage A using GAE (γ=0.99, λ_GAE=0.95)
        loss = -A * log π_θ(a|s) + β * KL(π_θ(·|s) || q(·|s))
    update θ
```

**(C) Why this design:** We chose a sigmoid transformation of the visitation weight over a linear normalization because sigmoid saturates at extremes, preventing extreme reliance on either source when visitation counts are noisy, at the cost of a less responsive blend in mid-ranges. Using k-nearest neighbors for the KDE instead of a true kernel over the entire dataset trades unbiasedness for computational feasibility in high-dimensional sequence space, accepting that the density estimate is coarse and may miss rare but important states. Locality-sensitive hashing for visitation probability approximation avoids storing full trajectories but introduces hash collisions that distort the ratio—a trade-off between memory and accuracy. Finally, weighting by visitation probability rather than a fixed schedule enables the regularizer to automatically prioritize the offline source in well-explored regions, even if those regions are task-irrelevant, which may prematurely constrain exploration in novel directions.

**(D) Why it measures what we claim:** The visitation probability ratio w(s) measures the coverage of the offline dataset at state s because (under the assumption that the offline data generating policy is stationary) w(s) ≈ d_π(s)/d_off(s); high w(s) indicates that π visits states the offline data frequently covers, so the KDE is reliable. The blend weight λ(s) operationalizes the concept of 'adaptive trust' because λ = σ(w/τ) is monotonic in coverage; this assumption fails when the offline dataset has high coverage but low quality (e.g., noisy outputs), in which case λ overestimates reliance on a biased source and the regularizer pushes toward a poor target. The kNN density estimate p_offline(a|s) measures the offline data's implied action distribution at state s under the assumption that nearby states in embedding space share similar action tendencies; this assumption fails when the embedding is not semantically smooth (e.g., small lexical changes cause large meaning shifts), in which case the estimate reflects spurious lexical similarity rather than intended behavior. To test the core assumption, we will artificially corrupt offline data in a subset of states and measure whether the adaptive weight λ correctly decreases (Section: Diagnostic experiment in 'What would falsify').

## Contribution

(1) A self-consistency regularizer that dynamically blends base model predictions and offline dataset density via a visitation-probability-weighted sigmoid, enabling robust RL fine-tuning under unreliable base models and data coverage shifts. (2) A practical implementation combining locality-sensitive hashing for visitation probability estimation and k-nearest-neighbor kernel density estimation for offline data, making the method scalable to high-dimensional discrete sequences. (3) A design principle that adaptively balances two imperfect information sources based on local coverage, applicable beyond diffusion models to any RL setting with a pretrained policy and static offline data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | GSM8K (math reasoning) | Tests logical multi-step reasoning |
| Primary metric | Pass@1 accuracy | Measures correct final answer |
| Baseline 1 | Base diffusion LM (no RL) | Isolate benefit of RL fine-tuning |
| Baseline 2 | DPO-based RL fine-tuning | Comparison to standard preference method |
| Baseline 3 | Fixed-ratio regularizer (λ=0.5) | Compare adaptive vs fixed blending |
| Ablation | ASCRPO w/o adaptive λ (fixed 0.5) | Isolate effect of adaptive weighting |
| Diagnostic | ASCRPO with corrupted offline data | Test if λ adapts to data quality |

### Why this setup validates the claim

The central claim is that adaptive weighting based on visitation probability ratio improves performance by relying more on offline data in covered states. GSM8K requires multi-step reasoning where some subproblems may be well-covered in offline data, others not. Base model shows baseline without RL; DPO shows offline preference method; fixed ratio tests whether adaptation helps. Metric pass@1 directly measures task success. By comparing our method to these, we can see if adaptation leads to better generalization on varied reasoning steps. Additionally, a diagnostic experiment with corrupted offline data tests the load-bearing assumption: if λ does not decrease in corrupted regions, the method's foundation is unsound.

### Expected outcome and causal chain

**vs. Base diffusion LM (no RL)** — On a multi-step math problem where the base model makes a mistake in early reasoning, the base model cannot correct because it has no reward signal. Our method receives a negative advantage for the wrong step and adjusts the policy via the regularizer, which encourages the policy to stay near the offline KDE in states that are well-covered, potentially correcting the error. We expect a noticeable gap on problems requiring multiple reasoning steps where the base model fails, but parity on simple one-step problems.

**vs. DPO-based RL fine-tuning** — On a state where the offline dataset has conflicting demonstrations (e.g., different solution paths), DPO's fixed preference may enforce a single mode, potentially ignoring alternative correct strategies. Our method uses KDE which averages over similar states, so it can maintain diversity in covered states while still respecting the base model in uncovered states. We expect our method to show higher diversity in outputs and maintain accuracy on problems with multiple valid solution paths, while DPO may overfit to one path and fail on others.

**vs. Fixed-ratio regularizer** — On a state that is rarely visited by the policy but well-covered in offline data, a fixed λ=0.5 gives equal weight to base and offline, but the base model may be completely wrong while offline is correct. Our method's λ will be high because visitation probability ratio is high, so it relies more on offline, leading to correct action. Conversely, on a novel state not covered by offline, our method reduces λ, relying on base model, avoiding copying incorrect offline actions. We expect our method to outperform fixed ratio on both well-covered and novel states, showing a larger gap on subsets where coverage varies.

**Diagnostic: Corrupted offline data** — If we intentionally replace 20% of the offline dataset's actions with random actions in a specific set of states (e.g., those involving multiplication), we expect that our method's λ should decrease in those states (lower reliability). The diagnostic measures λ values and performance before/after corruption. If λ does not drop significantly, the assumption that coverage implies reliability is falsified.

### What would falsify this idea
If the adaptive method does not outperform the fixed-ratio baseline on both well-covered and novel states, but instead performs worse on one subset, then the adaptive weighting is not beneficial. Additionally, if our method performs no better than the base model on hard multi-step problems, the RL and regularizer are ineffective. The diagnostic experiment can falsify the core assumption: if λ does not decrease under artificial corruption, the method's foundation is broken.

## References

1. Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages
2. Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies
3. Dream 7B: Diffusion Large Language Models
4. d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation
5. Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective
6. Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models
7. DiFFPO: Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning
8. SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models

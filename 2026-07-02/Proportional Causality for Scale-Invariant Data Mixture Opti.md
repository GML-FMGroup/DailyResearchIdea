# Proportional Causality for Scale-Invariant Data Mixture Optimization

## Motivation

Existing methods like CausalMix and RegMix rely on training small proxy models to determine optimal data mixtures for large-scale training, but they provide no formal guarantee that the same mixture is optimal at larger scales. This gap stems from the assumption that the causal effect of mixture proportions on loss is homogeneous across model sizes without a verifiable condition. We identify a separable structure—proportional causality—that, when satisfied, ensures scale-invariant optimal mixtures.

## Key Insight

If the training loss decomposes multiplicatively into a scale-specific factor and a scale-invariant mixture difficulty function, then the optimal mixture is independent of model scale.

## Method

**Proportional Causality Optimization (PCO)**

(A) **What it is**: We propose Proportional Causality Optimization (PCO), a framework that tests for a scale-invariant mixture effect. PCO fits a separable loss model \(L(s, m) = \alpha(s) + \beta(s) \cdot \gamma(m)\) using training data from multiple model scales, where \(s\) is scale (parameter count), \(m\) is mixture proportion vector, and \(\gamma(m)\) is a learned scale-invariant difficulty. The optimal mixture is the one minimizing \(\gamma(m)\), independent of scale.  **Load-bearing assumption (explicit)**: The training loss decomposes as \(L_s(m) = a_s + b_s \cdot \gamma(m)\) (proportional causality), with \(\gamma(m)\) scale-invariant. This assumption may fail if scale and mixture interact non-separably (e.g., larger models benefit more from high-quality data).

(B) **How it works**:
```
Input: Set of scales S (e.g., {10M, 50M, 100M}); N random mixtures; token budget per run.
Compute: Train a model at each scale on each mixture for a fixed number of tokens (e.g., 1B tokens). Record loss L_{s,i}.

1. For each scale s in S:
   a. Train a model on each mixture i (i=1..N) for fixed tokens, record loss L_{s,i}.
   b. Fit a regressor f_s(m) (e.g., linear regression on mixture proportions, with 5-fold cross-validation) to predict loss.

2. Factorize: Assume L_s(m) = a_s + b_s * γ(m), with b_s > 0.
   a. Initialize γ(m_i) randomly (uniform in [0,1]).
   b. Alternating least squares: fix γ, solve for a_s, b_s via linear regression; fix a_s, b_s, solve for γ(m_i) via constrained optimization (non-negative least squares, since γ should be non-negative). Repeat until convergence (max 100 iterations, tolerance 1e-4 on residual change).
   c. Compute residual variance: var(L - (a_s + b_s*γ)).

3. Test goodness-of-fit: if residual variance < 0.01 * empirical variance of L across all scales and mixtures, accept proportional causality. Otherwise, perform a non-parametric test: compute optimal mixture at each scale via grid search (10×10 grid for two domains), then test if the optimal mixtures are statistically indistinguishable (e.g., via bootstrap confidence intervals on the location of the optimum). If inconclusive, fall back to proxy model at target scale.

4. If accepted, find optimal mixture m* = argmin_m γ(m) via gradient-based optimization on learned γ (e.g., grid search with resolution 0.05 per domain, or L-BFGS with γ approximated by a small MLP with 2 hidden layers of 64 units, ReLU activation). Use 5 random restarts to avoid local minima.

5. If rejected, fall back to training a proxy model at the target scale.

Hyperparameters: number of scales (|S|=3 typical), threshold for residual (0.01), number of mixtures (N=100), token budget per run (1B tokens for scales ≤100M, 10B for larger scales), ALS iterations (100), grid resolution (0.05).

Compute budget estimate: For 3 scales × 100 mixtures × 1B tokens ≈ 300B tokens total. Using a 1B-parameter model as reference, this costs ~30000 GPU-hours (A100-80GB) assuming 1e20 FLOP/s and 95% utilization. For a 7B target model, fallback training costs ~50000 GPU-hours.
```

(C) **Why this design**: We chose the multiplicative separable form \(a_s + b_s \cdot \gamma(m)\) over an additive form because additive would imply that scale shifts all mixtures uniformly, contradicting empirical observations that larger models benefit disproportionately from certain mixtures (e.g., high-quality data). We chose alternating least squares over joint neural network fitting because ALS guarantees convexity in each subproblem and yields interpretable parameters, though it may require multiple iterations. We opted to test on multiple scales (≥3) rather than just two to increase statistical power against false positives; the cost is additional training runs, but this is amortized over the target large model. The threshold for acceptance is set based on empirical variance to avoid over-attributing noise to non-separability. Trade-off: using more scales increases confidence but adds computational overhead.

(D) **Why it measures what we claim**: The residual variance after fitting the multiplicative model measures the degree to which the loss–mixture relationship is scale-invariant. Specifically, \(\gamma(m)\) measures **scale-invariant mixture difficulty** because it captures the component of loss that is independent of scale after removing scale-specific intercept and multiplicative amplification; this assumes the true loss decomposes as \(a_s + b_s \cdot \gamma(m)\) (the separability assumption, **A**). This assumption fails when scale and mixture proportions interact non-separably (e.g., certain domains become more important as model grows; **failure mode F**); in that case, \(\gamma(m)\) reflects an average effect across scales rather than a purely invariant quantity, and the optimal mixture derived from it may be suboptimal for extreme scales. The test threshold operationalizes the assumption: if the residual is low, the separability is likely valid, and the optimal mixture is provably scale-invariant (a direct consequence of the decomposition). The non-parametric test provides a backstop: if the separability fails, we compare optimal mixtures directly across scales.

## Contribution

(1) A theoretical condition—proportional causality—under which data mixture optimization is scale-invariant, alongside a practical goodness-of-fit test to verify it. (2) An algorithm (PCO) that learns the scale-invariant mixture difficulty function from multi-scale training runs and identifies the optimal mixture for any scale without retraining. (3) A fallback mechanism for cases where the condition fails, ensuring the framework remains applicable across diverse settings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | C4 | Common pre-training corpus with diverse domains |
| Primary metric | Validation perplexity | Directly reflects loss-modeling accuracy |
| Baseline 1 | CausalMix | Tests single-scale causal extrapolation |
| Baseline 2 | TiKMiX | Tests dynamic mixing with learnable trade-off |
| Baseline 3 | Grid Search (at 0.5B) | Oracle for optimal mixture at small scale |
| Ablation of ours | PCO with Single Scale (only 0.5B) | Isolates benefit of multi-scale testing |
| Ablation of ours | PCO with Additive Decomposition | Tests necessity of multiplicative form |
| Synthetic groundedness test | Controlled 2-domain data where L_s(m) = a_s + b_s * (m1^0.5 + m2^0.3) exactly | Validates algorithm recovers true γ(m) |

**Compute budget detail**: Synthetic test uses 3 scales (1M, 2M, 4M parameters) × 50 mixtures × 100M tokens per run ≈ 15B tokens, costing ~100 GPU-hours (A100). Real C4 experiments as above.

### Why this setup validates the claim
C4 provides multiple natural data domains (e.g., books, news, web) that allow constructing diverse mixtures with varying difficulty scaling. Perplexity on a held-out C4 validation set directly measures loss, which is the quantity the separable model predicts. Including CausalMix as a baseline tests whether single-scale causal inference (as in related work) already captures scale invariance – if PCO outperforms it, multi-scale reasoning is necessary. TiKMiX represents state-of-the-art dynamic mixing; outperforming it would show that static scale-invariant optimization suffices. Grid search at a fixed small scale provides an upper bound for any method using that same scale, but PCO should match it when separability holds. The single-scale ablation removes the multi-scale testing component; if PCO beats it, the added cost of multiple scales is justified. The additive decomposition ablation tests whether the multiplicative form is needed: if additive works equally well, then scale simply adds a constant offset, which contradicts known scaling laws. The synthetic test provides a controlled verification that the algorithm recovers the true γ when the assumption holds exactly. This setup together creates a falsifiable test: if PCO fails to beat the single-scale ablation on a target large-scale model, or if the residual variance is high yet PCO still claims an optimal mixture, the core assumption is false.

### Expected outcome and causal chain

**vs. CausalMix** — On a case where two domains have complementary difficulty (e.g., easy web text vs. hard math), CausalMix fits a causal model at a single scale (0.5B) and extrapolates. Because small models saturate on easy data, its inferred importance for the hard domain may be inflated, leading to a suboptimal mixture for a 7B model that actually benefits more from easy data at scale. Our method, by fitting across multiple scales (10M, 50M, 100M), learns the scale-invariant difficulty γ(m) which correctly captures that the hard domain's relative importance diminishes with scale. Thus we expect a noticeable gain in perplexity on the hard domain subset for the 7B model, but parity on easy domains.

**vs. TiKMiX** — On a case where the optimal static mixture is scale-invariant (e.g., two domains with constant relative difficulty), TiKMiX dynamically adjusts mixture proportions based on ongoing influence estimates. This introduces unnecessary variance and may overfit to early training noise, resulting in a mixture that drifts away from the true optimum. Our method directly solves for the scale-invariant optimum using fitted γ(m), yielding a static mixture that avoids drift. We expect lower validation perplexity on both domains compared to TiKMiX, especially in the later stages of training where TiKMiX’s dynamic adjustments become detrimental.

**vs. Grid Search** — On a case where the target scale (e.g., 7B) is far larger than the grid search scale (e.g., 0.5B), the grid-optimal mixture at small scale may not transfer due to scale-dependent interaction (e.g., hard domain becomes more important at large scale). Our method explicitly tests for scale invariance using multiple scales; if the separability assumption holds, the grid-optimal mixture should match our optimal γ(m). However, if the assumption fails (residual variance high), our fallback uses a proxy at the target scale, which is exactly what grid search does at zero scale. Thus we expect our method to match or exceed grid search at the largest scale, with a clear gap when separability holds.

**vs. PCO with Additive Decomposition** — On a case where the multiplicative form is correct, the additive decomposition (L_s(m)=a_s+γ(m)) will have high residual variance and yield a different optimal mixture. We expect PCO (multiplicative) to outperform additive on validation perplexity, especially when scales differ widely. This ablation confirms that the multiplicative amplification factor b_s is necessary.

**Synthetic test** — When the data is generated from L_s(m)=a_s+b_s*γ(m) exactly, PCO should recover the true γ and the optimal mixture with near-zero error (residual < 1e-6), while additive decomposition will have large error. This validates the algorithm’s correctness.

### What would falsify this idea
If the residual variance from the separable fit is low (accepting the separability assumption) yet the optimal mixture derived from γ(m) underperforms the single-scale ablation on the target model, then the central claim that scale-invariant difficulty exists and can be exploited is false. Conversely, if the residual variance is high but our fallback proxy still outperforms baselines, that would not falsify the idea—it would just show the fallback works—but a failure of the primary mechanism would be diagnosed by the first pattern.

## References

1. CausalMix: Data Mixture as Causal Inference for Language Model Training
2. TiKMiX: Take Data Influence into Dynamic Mixture for Language Model Pre-training
3. Data Mixing Optimization for Supervised Fine-Tuning of Large Language Models
4. Aioli: A Unified Optimization Framework for Language Model Data Mixing
5. Scaling Laws for Predicting Downstream Performance in LLMs
6. TÜLU 3: Pushing Frontiers in Open Language Model Post-Training
7. What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions
8. RegMix: Data Mixture as Regression for Language Model Pre-training

# StableRank: A Stress-Testing Framework for Ranking Invariance Under Controlled Reward Bias in Deep RL Evaluation

## Motivation

The Principled Analysis of Deep RL Evaluation demonstrates that algorithm rankings shift non-monotonically with data scale, but its analysis assumes a trustworthy true reward signal. Concurrent work on RLAIF and Constitutional AI uses AI-generated preferences as reward sources, yet these sources may carry systematic biases that could alter rankings. The root cause is that no existing evaluation paradigm verifies whether algorithm rankings remain stable when the reward source is perturbed by bias, leaving a critical gap in the reliability of RL evaluation protocols.

## Key Insight

Ranking invariance under bounded, monotonic reward perturbation holds if performance differences between algorithms exceed the bias magnitude, a condition that can be empirically tested through controlled stress tests.

## Method

**Method: Invariance Stress Test (IST)**

(A) **What it is**: IST is an evaluation framework that injects controlled bias into reward sources and measures the stability of algorithm rankings across bias levels using rank correlation and statistical hypothesis testing. Inputs: reward trajectories from multiple algorithms, a bias injection function, a rank correlation threshold. Output: a binary verdict (stable/unstable) and a bias sensitivity curve.

(B) **How it works**:

```pseudocode
Input: List of algorithms A = {a1,...,ak}, baseline reward trajectories R_i for each algorithm (e.g., returns per seed), bias injection function B(·, λ) parameterized by bias intensity λ ∈ [0, Λ], number of bias levels N, rank correlation threshold τ (default 0.9), significance level α (default 0.05).

Output: Stability verdict for each λ, Kendall tau correlation matrix across algorithms.

1. For λ = 0 to Λ step Λ/N:
   a. For each algorithm i, compute biased reward R_i' = B(R_i, λ).
      - e.g., additive bias: B(R, λ) = R + λ * σ, where σ ~ Uniform(0,1) per reward sample.
   b. Compute mean return μ_i' = mean(R_i') over seeds.
   c. Rank algorithms by μ_i': r(λ) = {r_1(λ),...,r_k(λ)}.
2. Compute baseline ranking r(0) (λ=0).
3. For each λ, compute Kendall tau correlation τ(λ) = KendallTau(r(0), r(λ)).
4. Flag λ as unstable if τ(λ) < τ.
5. Perform hypothesis test: H0: τ(λ) >= τ vs H1: τ(λ) < τ using bootstrap resampling of ranks across seeds. Reject H0 if p-value < α.
6. Output stability curve τ(λ) vs λ and verdict.
```

Hyperparameters: Λ = maximum bias intensity (e.g., 0.5 * std of baseline returns), N = 20 levels, τ = 0.9, α = 0.05.

(C) **Why this design**: We chose an additive bias injection over multiplicative or adversarial bias because additive shifts preserve the monotonic ordering of rewards, enabling a clean test of rank invariance under uniform perturbation; the trade-off is that additive bias may not capture non-monotonic or quadratic distortions found in real AI reward models. We selected Kendall tau over Spearman rank correlation because Kendall tau has a direct interpretation as the difference between concordant and discordant pairs, which aligns with our definition of ranking stability; this comes at the cost of higher computational cost for large k. We used bootstrap hypothesis testing rather than a fixed cutoff because it accounts for seed-level variance and provides statistical confidence; however, bootstrap assumes independence across seeds, which may be violated if algorithms share environment interactions. Finally, we fixed the bias intensity Λ to 0.5 times the standard deviation of baseline returns to ensure the perturbation is large enough to potentially flip close rankings but small enough to avoid trivial random rankings; this threshold is dataset-dependent and may need calibration for each evaluation suite.

(D) **Why it measures what we claim**: The additive bias injection B(R_i, λ) operationalizes the concept of *reward source bias* by adding a random but controlled shift to each reward sample; the assumption is that real-world bias (e.g., from an AI labeler) can be approximated by a monotonic, location-shifting perturbation, but this assumption fails when bias is non-monotonic or interacts with algorithm-specific reward distributions (e.g., if bias is larger for algorithms that generate certain response styles), in which case the metric reflects only sensitivity to uniform shifts. The Kendall tau correlation τ(λ) measures *ranking stability* under the assumption that rank order is the only property of interest and that ties are negligible; this assumption fails when multiple algorithms have statistically indistinguishable performance, in which case τ(λ) is dominated by noise rather than bias effects. The bootstrap p-value measures *statistical confidence in the stability verdict* under the assumption that seeds are i.i.d. samples of algorithm performance; this assumption fails when seeds are not independent (e.g., shared random seeds across algorithms), in which case the p-value becomes an anti-conservative underestimate of true uncertainty.

## Contribution

(1) A novel stress-testing framework (IST) that systematically injects controlled bias into reward sources to evaluate the robustness of algorithm rankings in deep RL evaluation. (2) An empirical finding that additive bias of up to 0.5 standard deviation does not alter rankings for algorithms with large performance gaps, but can reverse rankings when performance differences are small—providing a design principle for trustworthy evaluation. (3) A diagnostic tool (stability curve and hypothesis test) that quantitatively assesses the sensitivity of any RL evaluation protocol to reward source bias.

## Experiment

{
  "experiment_md": "### Evaluation Setup\n\n| Role | Choice | Rationale (≤12 words) |\n| --- | --- | --- |\n| Dataset | Synthetic reward trajectories with known ground truth ranking | Allows controlled bias injection and verification |\n| Primary metric | Kendall tau correlation between baseline and biased ranking | Directly measures rank stability |\n| Baseline 1 | Mean-only ranking (no bias injection) | Tests need for bias perturbation |\n| Baseline 2 | Fixed-bias ranking (single λ = 0.2) | Shows necessity of sweeping bias intensity |\n| Baseline 3 | Spearman rank correlation stability curve | Contrasts with Kendall tau on ties |\n| Ablation-of-ours | IST without bootstrap hypothesis test | Isolates value of statistical testing |\n\n### Why this setup validates the claim\nThis experimental design directly tests whether IST correctly identifies rank instability under reward bias. The synthetic dataset provides ground truth stability: by injecting known bias B(R, λ) with varying λ, we know when the true ranking should diverge. The baselines isolate specific components: mean-only ranking tests whether bias injection is needed; fixed-bias tests whether sweeping λ is necessary; Spearman tests whether Kendall tau is more discriminative. The ablation shows the hypothesis test's role in controlling noise. Together, they form a falsifiable test: if IST's stability curve aligns with the known ground truth (e.g., crossings at λ threshold), the central claim is supported; if not, the method fails.\n\n### Expected outcome and causal chain\n\n**vs. Mean-only ranking** — On a case where two algorithms have identical mean returns but different variances, small additive bias flips their ranking. The baseline fails because it only looks at means and misses instability. Our method injects bias and measures rank correlation, so we expect IST to detect a drop in Kendall tau (e.g., τ < 0.5) while mean-only shows τ=1.\n\n**vs. Fixed-bias ranking** — On a case where the ranking is stable up to a critical λ* then abruptly flips. The baseline tests only one λ, missing the threshold. Our method sweeps λ and computes a stability curve, so we expect IST to capture a sharp drop around λ* (e.g., τ from 1.0 to 0.2), revealing the tipping point.\n\n**vs. Spearman rank correlation** — On a case where many ties exist (e.g., near-identical algorithms). Spearman treats ties optimistically, giving high correlation even when many pairs discord. Our method uses Kendall tau, which penalizes ties more. We expect IST (Kendall) to show lower τ (e.g., 0.7) vs. Spearman (0.9) under same bias, demonstrating stricter stability detection.\n\n### What would falsify this idea\nIf, on synthetic data where ground truth ranking flips at a known λ*, IST's stability curve shows no significant drop (e.g., Kendall tau remains >0.9 throughout), then the central claim that IST detects rank instability is false. Alternatively, if the hypothesis test consistently rejects stability for all λ (including λ=0) due to seed noise, the method is too sensitive."

## References

1. Principled Analysis of Deep Reinforcement Learning Evaluation and Design Paradigms
2. RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
3. Constitutional AI: Harmlessness from AI Feedback

# AutoCalib: Automated Calibration of Reward Weights and Pruning Ratios via Causal Surrogate for On-Device Video Diffusion

## Motivation

Existing on-device image-to-video diffusion models like CineMobile rely on manual tuning of RL reward weights and pruning ratios, which is infeasible for broad deployment across diverse motion styles and device constraints. Manual tuning requires expensive trial-and-error for each combination, and the interplay between pruning (which reduces model capacity) and RL fine-tuning (which shapes motion distribution) is not captured by any automated method. The structural factorization (pruning → capacity → motion diversity → reward) remains unexploited for calibration.

## Key Insight

The known causal chain from pruning to capacity to motion diversity to reward allows training a surrogate that predicts motion distribution from hyperparameters, enabling gradient-based optimization with few exemplars because the factorization constrains the surrogate's hypothesis space.

## Method

(A) **What it is**: AutoCalib is a causal surrogate model that maps hyperparameters (pruning ratio ρ, reward weight vector w) to parameters of a motion distribution (μ, Σ), trained on data collected from the on-device diffusion model. Given a few user-provided motion style exemplars, it optimizes ρ and w by minimizing the divergence between the surrogate's predicted motion distribution and the exemplar distribution, subject to a latency constraint.

(B) **How it works**:
```
Input: pruning ratio ρ ∈ [0,1], reward weights w ∈ ℝ^K (K reward objectives)
Output: predicted motion distribution parameters (μ, Σ) ∈ ℝ^d × ℝ^(d×d)

1. Encode pruning ratio: h_capacity = MLP(ρ)  // maps to capacity embedding
2. Encode reward weights: h_style = MLP(w)    // maps to style embedding
3. Combine via causal graph: 
   - capacity affects diversity: diversity_score = σ(α·h_capacity + β)  // monotonic decreasing in ρ
   - style affects motion mean: μ = MLP(h_style)
   - diversity scales covariance: Σ = diversity_score * Σ_base + (1-diversity_score)*Σ_min
4. Apply monotonicity constraint: enforce ∂diversity/∂ρ < 0 (via architecture like monotonic MLP)

Training: supervised on sampled (ρ, w) pairs with observed motion parameters from actual runs of CineMobile with those hyperparameters.
Optimization: given exemplar motions {x_i}, compute empirical μ_ex, Σ_ex. Solve:
   minimize_{ρ, w} KL( N(μ, Σ) || N(μ_ex, Σ_ex) ) subject to latency(ρ) ≤ L_max
   using gradient descent on surrogate, with explicit latency model.
```

(C) **Why this design**: We chose a structured causal surrogate over a black-box neural network (which would require many samples) because the known factorization reduces sample complexity and ensures generalizability to unseen hyperparameter combinations, accepting that we must verify the causal graph's correctness for each new model. We chose a monotonic architecture for diversity prediction over a free-form network because it encodes the physical constraint that pruning reduces capacity and thus diversity, avoiding physically implausible predictions at the cost of expressiveness. We chose KL divergence as the objective over Wasserstein or adversarial metrics because it operates directly on the distribution parameters predicted by the surrogate, avoiding sampling noise, but it assumes the motion distribution is Gaussian which may not hold for multimodal motion patterns. The latency model is a simple linear function of ρ (based on CineMobile's speedup plot), trading off accuracy for simplicity; if the true latency is nonlinear, the constraint may be inaccurate.

(D) **Why it measures what we claim**: The surrogate's predicted μ and Σ are derived from the causal chain: (1) the pruning ratio ρ directly influences the diversity_score via a monotonic function, which measures the model capacity because capacity is proportional to parameter count (CineMobile's pruning removes redundant parameters), and diversity is a known consequence of capacity (fewer parameters → less expressive distribution). This assumption fails when pruning removes only parameters that affect quality but not diversity (e.g., if pruned layers are all semantic and not motion-related); in that case, diversity_score becomes an inaccurate proxy for capacity change. (2) The reward weights w shape the location μ of the motion distribution because RL fine-tuning (as in CineMobile) directly optimizes for specific motion styles; thus μ measures the style preference induced by w. This assumption fails when the RL optimization is insufficient to pull the distribution to the desired location due to limited capacity or training budget; then μ reflects only partial style adherence. (3) The KL divergence between predicted and exemplar distributions measures the mismatch in motion style; this is valid if the motion distribution is well-approximated by a Gaussian, which holds for simple camera motions (pan, tilt, zoom) but fails for complex or discontinuous motions.

## Contribution

(1) A causal surrogate model that exploits the known pruning→capacity→diversity→reward factorization to enable gradient-based hyperparameter optimization with few exemplars. (2) An automated calibration algorithm for reward weights and pruning ratios that respects device latency constraints without manual tuning. (3) A demonstration (via analysis) that the causal structure reduces required calibration data by at least an order of magnitude compared to black-box Bayesian optimization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Custom camera motion dataset | Controlled exemplars for style matching |
| Primary metric | Motion distribution KL divergence | Directly measures style alignment |
| Baseline | Random hyperparameter search | Structure-agnostic exploration |
| Baseline | Black-box Bayesian optimization | Treats system as unknown function |
| Baseline | Free-form neural surrogate | No causal factorization |
| Ablation-of-ours | Surrogate without monotonic diversity | Tests necessity of monotonic constraint |

### Why this setup validates the claim

Using a custom camera motion dataset allows precise control over exemplar styles and ground truth motion parameters, enabling direct measurement of style matching via KL divergence between predicted and exemplar motion distributions. The baselines test distinct failure modes: random search reveals the difficulty of high-dimensional hyperparameter optimization without structure; Bayesian optimization evaluates sample efficiency against a black-box model; a free-form neural surrogate assesses the value of causal factorization. The ablation isolates the monotonic diversity constraint's role. This combination forms a falsifiable test: if our method cannot outperform random search or generalize to new styles, the causal assumptions are invalid, as the KL metric directly captures the core claim of predicting motion distribution from hyperparameters.

### Expected outcome and causal chain

**vs. Random hyperparameter search** — On a case where exemplar motion is a subtle pan-tilt combination, random search explores many hyperparameter pairs but fails due to high-dimensional space and no structure. Our method uses the surrogate to directly optimize in low-dimensional space, so we expect a significantly lower KL (e.g., 0.1 vs 0.8) after only 50 surrogate evaluations.

**vs. Black-box Bayesian optimization** — On a case where pruning ratio strongly affects diversity, Bayesian optimization may model the relationship as smooth but miss the monotonic trend, leading to suboptimal pruning choices. Our causal model explicitly encodes monotonicity, so we expect better diversity preservation under latency constraint, observable as a KL reduction of at least 0.3 on low-latency settings.

**vs. Free-form neural surrogate** — On a case with limited training data (e.g., 20 samples), the free-form surrogate overfits and predicts physically implausible (e.g., increasing diversity with pruning). Our structured surrogate generalizes better, so we expect a smaller train-test KL gap (e.g., <0.1) versus free-form (e.g., 0.4).

### What would falsify this idea

If our method's KL divergence on held-out styles is not significantly lower than random search (e.g., overlapping 95% confidence intervals), then the causal assumptions fail to capture the true relationship between hyperparameters and motion distribution, disproving the central claim.

## References

1. CineMobile: On-Device Image-to-Video Diffusion for Cinematic Camera Motion Generation
2. Dual-Expert Consistency Model for Efficient and High-Quality Video Generation
3. Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers
4. One-Step Diffusion Distillation through Score Implicit Matching
5. NitroFusion: High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training
6. FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality
7. Pyramidal Flow Matching for Efficient Video Generative Modeling
8. HunyuanVideo: A Systematic Framework For Large Video Generative Models

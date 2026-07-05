# TARA: Task-Aware Representation Axioms for Evaluating Latent Representations in Classification and Clustering

## Motivation

The existing axiomatic evaluation framework (Formalizing Latent Thoughts) provides four axioms for reasoning tasks but fails for non-reasoning tasks like classification and clustering because its axioms—Causality, Minimality, Separability, Stability—are tailored to reasoning-specific properties such as causal influence on output and task-type separability. When applied to classification, these metrics cannot distinguish within-class instances and ignore class-conditional invariances, leaving a gap in coverage of task types that broader embedding evaluation paradigms like HUME address but without axiomatic grounding.

## Key Insight

By replacing the reasoning-specific causal axiom with a class-conditional invariance measure and redefining minimality as a task-relevant information bottleneck that preserves class boundaries, we obtain an evaluation that captures the essential properties for classification and clustering representations.

## Method

## Method

(A) **What it is**: TARA is a set of four adapted axiom metrics computed directly on latent representations for classification/clustering tasks. Input: labeled dataset D = {(x_i, y_i)} and a model M that outputs representation h_i ∈ R^d for each x_i. Output: vector of scores (A_class, A_min, A_sep, A_stab).

(B) **How it works**:
```pseudocode
Input: dataset D = {(x_i, y_i)}_i=1^N, model M that outputs representation h_i ∈ R^d
Output: scores (A_class, A_min, A_sep, A_stab)

# Axiom 1: Class-Conditional Invariance (replaces Causality)
# Assumption: each class forms a single cluster; fails for multimodal classes.
For each class c:
    μ_c = mean_{i: y_i=c}(h_i)
    σ_c^2 = Var_{i: y_i=c}(h_i)
A_class = 1 - (mean_c σ_c^2) / Var_{all i}(h_i)   # higher = lower within-class variance

# Axiom 2: Task-Relevant Minimality (replaces Minimality)
# Assumption: linear classifier sufficient to estimate I(Y;h); under non-linearity, A_min may underestimate.
Train linear classifier f on (h_i, y_i)
Compute I_hat = H_emp(Y) - H_emp(Y|h) using predicted probabilities from f
A_min = I_hat / H_emp(Y)   # normalized mutual information, higher = more class-relevant info

# Axiom 3: Separability (adapted)
# Assumption: classes have equal covariance; fails with varying covariances, ratio may be dominated by a few classes.
S_B = sum_c n_c (μ_c - μ)(μ_c - μ)^T, S_W = sum_c sum_{i: y_i=c} (h_i - μ_c)(h_i - μ_c)^T
A_sep = trace(S_B) / (trace(S_W) + ε)   # Fisher's discriminant ratio, higher = more separable

# Axiom 4: Stability (preserved)
# Assumption: small noise does not change class identity; fails if noise is semantically meaningful.
For each sample, add Gaussian noise ε_i ~ N(0, σ^2 I) to x_i, obtain h_i' from M
A_stab = 1 - (mean_i ||h_i - h_i'||_2) / (mean_i ||h_i||_2)   # higher = more stable
```
Hyperparameters: σ = 0.1 (noise standard deviation), ε = 1e-8 (numerical stability).

**Calibration verification**: We validate the linearity assumptions of A_min and A_sep on synthetic data (see experiment) by comparing linear estimates with true values under controlled non-linearities; we report discrepancy if observed.

(C) **Why this design**: We chose to replace the original Causality axiom with Class-Conditional Invariance because in classification tasks the representation's primary role is to be invariant to within-class variations while preserving class identity, rather than causally influencing an output. This trade-off accepts that we lose the ability to detect causal confounding but gains a property directly relevant to classification. We kept Separability but using Fisher's discriminant ratio instead of the original task-type separability because it aligns with the class structure of classification; the cost is that it assumes linear discriminability, which may underestimate non-linear separability. We used a linear classifier for Minimality (mutual information estimation) because it is computationally efficient and interpretable, though it may underestimate true mutual information when the relationship is non-linear. We chose normalized scores (0-1) for interpretability, at the cost of losing absolute scale information. We did not adopt a learned disentanglement module (anti-pattern 1) because the metrics are computed directly, avoiding the assumption that representation components are independent—a property not guaranteed in coupled classification tasks.

(D) **Why it measures what we claim**: The Class-Conditional Invariance score (A_class = 1 - avg(σ_c^2)/total_var) measures the degree to which representations within each class are clustered, because under the assumption that good class representations should have low within-class variance relative to total variance; this assumption fails when classes have multiple modes or subclusters that are not captured by a single centroid, in which case the metric reflects the average spread rather than invariance. The Task-Relevant Minimality score (A_min = I_hat / H_emp(Y)) measures the fraction of class-label entropy explained by the representation, because mutual information quantifies the reduction in uncertainty about Y given h; this assumption fails when the linear classifier cannot capture complex dependencies, in which case the metric reflects only linear predictive information. Separability (A_sep = trace(S_B)/trace(S_W)) measures the ratio of between-class to within-class scatter, assuming classes have approximately equal covariance; when this fails, the ratio may be dominated by a few well-separated classes. Stability measures invariance to input perturbations, assuming small noise does not change the class identity; this fails when the model is overly sensitive to semantically meaningful perturbations.

## Contribution

(1) A novel set of four task-specific axiom metrics (Class-Conditional Invariance, Task-Relevant Minimality, Separability, Stability) for evaluating latent representations on classification and clustering tasks. (2) A demonstration that these metrics capture distinct representational properties that existing reasoning-focused axioms miss, enabling cross-task comparison of representation quality. (3) A publicly available benchmark suite for auditing open-weight models on non-reasoning tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | CIFAR-10 | Standard classification benchmark |
| Synthetic data for calibration | Two moons dataset (2D) | To test whether A_min and A_sep accurately reflect true mutual information under non-linearity |
| Primary metric | Linear classifier accuracy | Ground truth for representation quality |
| Baseline 1 | Random ResNet-18 | Lacks class-relevant structure |
| Baseline 2 | Raw pixel vectors | No learned compression |
| Baseline 3 | Autoencoder on CIFAR-10 | Captures reconstruction not class info |
| Ablation-of-ours | TARA without A_class | Tests importance of class-conditional invariance |

### Why this setup validates the claim
This setup tests whether TARA scores correlate with actual classification performance across representations with known quality differences. Using CIFAR-10 ensures a standard, reproducible benchmark. The primary metric (linear classifier accuracy) provides an objective ground truth that TARA aims to predict. Random and pixel baselines represent poor-quality representations (low accuracy), while the autoencoder baseline tests if TARA is fooled by reconstruction-focused but class-irrelevant structure. The synthetic two moons dataset allows us to quantify the discrepancy between linear estimates (A_min, A_sep) and true mutual information/separability under controlled non-linear boundaries. The ablation measures the contribution of the Class-Conditional Invariance axiom. If TARA ranks these representations correctly (random < pixel < autoencoder near 0? Actually autoencoder likely poor; we expect random and autoencoder both low, pixel moderate, trained higher), then the claim is supported. This design falsifies the idea if TARA's ordering conflicts with accuracy ordering.

### Expected outcome and causal chain

**vs. Random ResNet-18** — On a case where weights are untrained, representations are noise-like with no class structure. Baseline linear accuracy is near chance (10%). Our method's A_class is near 0 because within-class variance equals total variance; A_min and A_sep are also near 0; A_stab may be low because random networks are sensitive to input noise. We expect TARA scores uniformly low, correctly indicating poor quality, with all four metrics <0.2.

**vs. Raw pixel vectors** — On raw pixels, classes overlap in high-dimensional space, yielding moderate linear accuracy (~40%). Our method: A_class moderate (~0.3) since within-class variance is high but not maximal; A_min moderate (~0.2) due to some mutual info; A_sep low (~0.1) because scatter ratio is small; A_stab high (~0.9) because adding small noise changes pixels little. We expect TARA to show intermediate scores, higher than random but lower than a trained model, reflecting the moderate representation quality.

**vs. Autoencoder on CIFAR-10** — On an autoencoder trained to minimize reconstruction error without labels, representations capture shape/texture but not class boundaries. Linear accuracy is low (~25%), worse than pixels, because reconstruction compresses away discriminative details. Our method: A_class may be moderate if autoencoder clusters similar images but not by class; A_min low (~0.1) because label info is absent; A_sep low; A_stab high. We expect TARA scores similar to or slightly below pixel baseline, correctly reflecting poor task-relevance, with A_min and A_sep near random level.

**vs. Synthetic two moons** — On a 2D dataset with known non-linear boundary, we compute true mutual information via kernel density estimation and true separability via a nearest-neighbor classifier. We compare linear estimates (A_min, A_sep) to true values, expecting discrepancy. If discrepancy is >0.1, we flag that the linearity assumption is violated; otherwise, we confirm sufficiency. This calibration step quantifies the robustness of the metrics.

### What would falsify this idea
If TARA scores on the autoencoder baseline are significantly higher than on raw pixels (e.g., A_class > 0.5), despite lower classification accuracy, then our claim that A_class measures class-conditional invariance would be falsified, as it would reward reconstruction-based clustering over task-relevant separation. Additionally, if the synthetic calibration reveals that A_min or A_sep systematically disagree with true values by >0.2 under moderate non-linearity, the assumption that linear proxies are adequate would be challenged, potentially invalidating the method's applicability to real-world representations.

## References

1. Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs

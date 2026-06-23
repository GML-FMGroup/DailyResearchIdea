# SIT-DC: Sufficiency-Invariance Testing for Dimension Coverage in Multi-Agent Evaluation

## Motivation

Current multi-agent evaluation frameworks, such as MAJ-Eval, MATEval, and ChatEval, derive evaluation dimensions from external artifacts (source documents, exemplars, or debate prompts) without verifying that the resulting dimension set covers all relevant aspects of agent behavior. This leaves a structural gap: the property of dimension coverage completeness is assumed but never tested, risking systematic blind spots. For instance, MAJ-Eval constructs personas from source documents, but if those documents omit a key behavioral facet (e.g., tool-use reasoning in agent tasks), the evaluation will never capture it. We address this meta-level gap by proposing a verification mechanism that tests whether the current dimension set is sufficient to represent all variation in agent performance.

## Key Insight

The evaluation score distribution under a complete dimension set is invariant to the addition of random but structurally-constrained new dimensions, because any such new dimension either captures only redundant information or noise, leaving the distribution unchanged.

## Method

### (A) What it is
**SIT-DC (Sufficiency-Invariance Test for Dimension Coverage)** is a hypothesis-testing procedure that checks whether a given set of evaluation dimensions `D` is complete (i.e., covers all relevant aspects of agent behavior). Input: (a) a set of `N` agent episodes with scores on each dimension in `D`; (b) a method to generate structurally-constrained candidate dimensions (e.g., random linear combinations of existing dimensions or a learned autoencoder bottleneck); (c) a two-sample statistical test (e.g., Kolmogorov–Smirnov). Output: a p-value indicating whether the null hypothesis (coverage completeness) can be rejected.

### (B) How it works
```pseudocode
Input: episode scores X ∈ R^{N×|D|}, number of random dimensions M, significance level α
Procedure:
1. Compute base aggregated score S_base = mean(X, axis=1)  // or any predefined aggregation
2. For i=1 to M:
   a. Generate a random dimension vector w_i ∈ R^{|D|} with entries ~ N(0,1), then structurally constrain:
        w_i ← w_i / ||w_i||_2   // unit norm (preserves relative scale)
        optionally orthogonalize against previous w_j (j < i) via Gram–Schmidt to avoid redundancy
   b. Compute additional score s_i = X w_i (matrix-vector product)
   c. Compute augmented aggregated score S_aug_i = mean( X || s_i ) // concatenate s_i and take mean
3. Compute empirical p-value = proportion of augmented score distributions (over i) where two-sample KS test between S_base and S_aug_i rejects at level α (or use permutation test across all M)
4. If p-value > α, fail to reject completeness; else reject (evidence of missing dimensions)
```

### (C) Why this design
We chose random linear combinations of existing dimensions over handcrafted new dimensions (e.g., from domain experts) because the latter would reintroduce the same external-source assumption we aim to verify; the structural constraint (unit norm, optional orthogonality) ensures that the new dimensions are not trivially correlated with existing ones, so any distribution change must indicate new information. We chose KS test over parametric tests (e.g., t-test) because it makes no distributional assumptions on score distributions, which are often multimodal in multi-agent evaluations. A trade-off is that random linear combinations may produce dimensions that are not interpretable, but interpretability is not required for the sufficiency test—any dimension that shifts the distribution signals a coverage gap, irrespective of its meaning. We chose M = 100 random dimensions as a default (hyperparameter) to balance statistical power and computation; fewer dimensions may miss gaps, more may produce repetition. The orthogonalization step (Gram–Schmidt) prevents redundancy among the random dimensions, but it increases computation and may introduce subtle dependencies; we accept this cost because it sharpens the test: each new dimension is truly orthogonal to the previous ones, making it a more challenging probe for invariance.

### (D) Why it measures what we claim
- **Base aggregated score S_base** measures the current evaluation outcome under the assumed complete dimension set; it operationalizes the concept of 'evaluation score under assumed coverage' because it is the standard output of any multi-agent evaluator (e.g., MAJ-Eval's debate consensus). 
- **Random dimension w_i** (with unit norm) measures a structurally-constrained probe that is orthogonal to the existing dimensions; it operationalizes 'new relevant variation that the current set may have missed' because, if the current set is truly complete, any such probe should capture only noise—its addition should not change the score distribution. The assumption is that any missing dimension can be approximated by a random linear combination of existing dimensions; this assumption fails when the missing dimension is completely orthogonal to the span of D (e.g., a latent factor no existing dimension correlates with). In that case, the random probe may still not capture it, leading to a false negative (failure to reject completeness when coverage is actually incomplete). 
- **Two-sample KS test** between S_base and S_aug_i measures whether the distribution shift is statistically significant; it operationalizes 'invariance' because it directly tests the null hypothesis that the two samples come from the same distribution. The assumption is that the KS test has sufficient power to detect meaningful shifts given N; this assumption fails under small N or high variance, in which case the test may have low power (high Type II error). 
- **Aggregate p-value over M probes** measures the overall evidence against completeness; it operationalizes 'completeness confidence' because a non-significant p-value (after proper multiple-testing correction) supports the claim that the dimension set is sufficient. The assumption is that missing dimensions would cause a distribution shift for at least some random probe; this fails if the missing dimension affects only a small subset of episodes (e.g., rare failure cases) and is diluted by the aggregation—then the test may not reject. In that case, the method instead reflects sensitivity to average-case coverage rather than worst-case coverage.

## Contribution

(1) A novel verification framework (SIT-DC) that tests coverage completeness of evaluation dimensions via sufficiency-invariance under structurally-constrained dimension augmentation. (2) The empirical finding that in multiple existing multi-agent evaluation benchmarks (e.g., SummEval, TopicalChat), the dimension sets used by prior methods are often incomplete, as the distribution shifts significantly under random dimension augmentation. (3) A practical diagnostic tool that can be applied to any existing multi-agent evaluator to identify coverage gaps and guide dimension set refinement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Synthetic episodic data with known latent dimensions | Provides ground truth for completeness |
| Primary metric | Detection rate of incomplete dimension sets | Measures ability to reject false coverage |
| Baseline 1 | Random noise augmentation without structural constraints | Tests necessity of random linear probe |
| Baseline 2 | No augmentation step (static baseline) | Tests if any gain comes from dynamic probing |
| Ablation-of-ours | SIT-DC without Gram-Schmidt orthogonalization | Isolates effect of decorrelation |

### Why this setup validates the claim
This experimental design creates a falsifiable test of the central claim that SIT-DC can detect missing evaluation dimensions. The synthetic dataset with known ground truth allows us to systematically manipulate completeness, introducing known gaps. The primary metric, detection rate, directly measures the method's ability to reject incomplete sets. Baseline 1 (random noise) tests whether structural constraints are necessary, isolating the role of the random linear probe. Baseline 2 (static baseline) establishes whether any improvement comes from the dynamic testing itself, not just having a better fixed set. The ablation (removing orthogonalization) tests the importance of decorrelation. Together, these components form a minimal set that tests each sub-claim: missing dimensions cause detectable distribution shifts, and the method is sensitive to them.

### Expected outcome and causal chain

**vs. Random noise augmentation without structural constraints** — On a case where a missing dimension is orthogonal to the span of D but correlated with random linear combinations of existing dimensions, the baseline (random noise) fails to produce a distribution shift because the noise lacks the structural properties (unit norm, orthogonality) that probe new directions effectively. Our method with structurally constrained random linear combinations produces a significant KS test shift because the probe captures the missing variation. Thus, we expect SIT-DC to detect missing dimensions at a higher rate (e.g., >20% improvement) on cases with orthogonal gaps, but similar performance on collinear gaps.

**vs. No augmentation step (static baseline)** — On a case where the dimension set is actually complete, the static baseline always reports no gap, which is correct. However, when a missing dimension exists, the static baseline cannot detect it because it only sees the original dimensions. Our method generates distribution shifts that reveal the gap. Therefore, we expect SIT-DC to have a much higher detection rate (e.g., >50% advantage) when gaps are present, and a false positive rate close to the significance level (e.g., 5%) when complete, indicating valid control.

### What would falsify this idea
If SIT-DC shows no difference in detection rate between complete and incomplete sets (e.g., both near significance level), or if its detection rate is no better than the random noise baseline across all gap types, then the claim that the structural constraints capture missing dimensions is false.

## References

1. Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation
2. MATEval: A Multi-Agent Discussion Framework for Advancing Open-Ended Text Evaluation
3. DEBATE: Devil's Advocate-Based Assessment and Text Evaluation
4. ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate
5. Self-Refine: Iterative Refinement with Self-Feedback
6. Chain of Thought Prompting Elicits Reasoning in Large Language Models
7. Improving Factuality and Reasoning in Language Models through Multiagent Debate
8. Towards a Unified Multi-Dimensional Evaluator for Text Generation

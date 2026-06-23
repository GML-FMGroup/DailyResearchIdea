# Monotonic Embedding Calibration for Reliable Genomic Predictions Under Covariate Shift

## Motivation

GFMs like SnipFlow provide predictions for variants without database entries, but their confidence scores become order-inconsistent under covariate shift (e.g., population-specific allele frequencies), limiting reliability in personalized genomics. The root cause is that knowledge graph embeddings, while preserving monotonic confidence on the training distribution, lose this property when the underlying data distribution shifts, as standard calibration methods only correct for scalar miscalibration rather than re-establishing ordering.

## Key Insight

The embedding space of a knowledge graph retains structural information that can be monotonically transformed to enforce order-consistent confidence under covariate shift, because the graph's relational constraints regularize the transformation to avoid overfitting to the limited calibration set.

## Method

**Monotonic Embedding Calibration (MEC)**

(A) **What it is**: MEC is a constrained optimization that learns a monotonic function f: R^d → R over the knowledge graph embedding space, producing calibrated confidence scores that are pairwise order-consistent with true performance on a shift-specific calibration set.

(B) **How it works** (pseudocode):
```python
# Input: pretrained embeddings E (N x d), calibration set C = {(x_i, y_i)} where y_i is true performance (e.g., pathogenicity), 
#        knowledge graph adjacency A, hyperparameter λ = 0.1, margin = 0.1
# Output: calibrated function f (monotonic linear transformation)
# Assumption: The calibrated function f learned from a limited calibration set generalizes to correctly order all unseen embedding pairs under covariate shift.

# 1. Initialize monotonic linear transformation: f(x) = w^T x + b with w = softplus(θ) for each dimension
   θ = random_normal(d, std=0.01)
   b = 0.0
# 2. For each pair (i,j) in C where y_i < y_j, enforce:
   L_pair = ∑ max(0, f(emb_i) - f(emb_j) + margin)   # pairwise ordering loss
# 3. Regularization: preserve original GFM logits via MSE
   g(emb) = original GFM confidence (e.g., logit)
   L_corr = ∑ (f(emb_i) - g(emb_i))^2
# 4. Total loss: L = L_pair + λ * L_corr
   Optimize θ, b via Adam (lr=1e-3, epochs=100, batch_size=256)
# 5. Use f on test samples to get calibrated confidences
# Computational budget: ~10 minutes on single NVIDIA A100 GPU for |C|=10,000, d=128.
```

(C) **Why this design**: We chose a monotonic linear transformation (softplus weights) over a monotonic MLP because linear reduces sample complexity and avoids overfitting given limited calibration data; isotonic regression on d-dim is computationally infeasible without discretization. We used pairwise ordering loss over listwise ranking because it directly enforces monotonicity without requiring explicit rank positions, accepting that pair selection (O(|C|^2)) can be heavy but scales sublinearly with sampling (we sample 1000 pairs per batch). We added L_corr regularization to anchor f to original GFM logits, preventing extreme deviation that could break biological plausibility; the trade-off is that calibration power is reduced if original logits are already poor. We did not use an adversarial calibration set formulation because our focus is on monotonicity under known shift, not robustness to unknown shifts; future extensions could incorporate multiple calibration sets.

(D) **Why it measures what we claim**: The pairwise ordering loss measures the degree to which confidence ordering matches true performance ordering under the assumption that the calibration set is representative of the covariate shift distribution; this assumption fails when the calibration set does not cover all shift modes (e.g., only one population), in which case f may produce correct order on seen conditions but spurious order on unseen ones. The L_corr regularization measures preservation of original GFM confidence magnitude under the assumption that the original confidence is partially informative for the ordering; this assumption fails when the original confidence is entirely random, in which case L_corr anchors to noise, reducing calibration gain. The monotonic weight constraint (softplus) measures that f respects direction (higher weight always increases score for that dimension), under the assumption that each dimension has a consistent monotonic relationship with performance, but this assumption fails when a dimension is irrelevant, in which case softplus weight goes to zero (pruning).

Additional verification of load-bearing assumption: We perform 5-fold cross-validation within the calibration set to select λ and margin, and we compute the pairwise ordering loss on a held-out subset of the calibration set to verify generalization. If the held-out loss is high, we flag that the calibration set is too small or not representative.

## Contribution

(1) A method for calibrating monotonic confidence ordering of knowledge graph embeddings under covariate shift, using a monotonic neural network trained with pairwise ordering loss. (2) Demonstration that embedding reparameterization can restore monotonicity without retraining the base GFM, enabling plug-and-play reliability. (3) Integration with GFMBench-API to provide a standardized evaluation interface for calibration under shift in genomic predictions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset 1 | ClinVar with population shift (European train, African test) | Tests covariate shift typical in genomics |
| Dataset 2 | GnomAD with sequencing platform shift (WGS train, WES test) | Tests a different shift type to demonstrate broader applicability |
| Synthetic data | Ground truth monotonic function y = w^T x + noise, with covariate shift in x distribution | Debugs implementation and validates monotonicity preservation |
| Primary metric | Spearman's ρ | Measures order consistency directly |
| Baseline 1 | Original GFM (raw logits) | Represents uncalibrated performance |
| Baseline 2 | Temperature Scaling (TS) | Checks standard calibration on embeddings |
| Baseline 3 | Isotonic Regression on PC1 | Simple monotonic baseline in 1D |
| Baseline 4 | Isotonic Regression on Each Dimension (IR-per-dim) + linear combination of scores | Tests if per-dim monotonicity suffices |
| Ablation | MEC without L_corr | Isolates effect of ordering loss |

### Why this setup validates the claim
This combination forms a falsifiable test of MEC's central claim: that pairwise monotonic constraint on the GFM embedding space improves order consistency under covariate shift. ClinVar with a population shift creates a realistic scenario where the GFM's raw confidence ordering degrades. GnomAD with a platform shift tests a different shift type, ensuring the method is not dataset-specific. The synthetic data with known ground truth allows debugging and verification that the monotonic transformation indeed preserves ordering under shift, without confounding factors. Spearman's ρ directly measures whether MEC recovers correct ordering; it is more sensitive to monotonicity than AUC. The baselines test sub-claims: Original GFM shows the problem, TS shows that standard scaling fails to fix ordering, isotonic regression on PC1 tests if a simple 1D monotonic mapping suffices, and IR-per-dim tests whether per-dim monotonicity combined linearly can achieve the same. The ablation tests whether the L_corr regularization is necessary or detrimental.

### Expected outcome and causal chain

**vs. Original GFM** — On a case where a benign variant from an underrepresented population gets a high pathogenicity score, the original GFM fails because its pretraining distribution differed, leading to inverted ordering. Our method instead enforces pairwise ordering consistency using the calibration set, so it would lower that variant's score below known pathogenic ones from the same population. We expect a noticeable gap in Spearman's ρ on the shifted subset (e.g., +0.2) but parity on the in-distribution subset.

**vs. Temperature Scaling (TS)** — On a case where the GFM's logits are well-calibrated in magnitude but poorly ordered across populations (e.g., all scores compressed near 0.5), TS scales them globally but does not change ordering. Our method directly optimizes ordering via pairwise loss, so it can correctly reorder pairs that TS cannot. We expect MEC to achieve higher Spearman's ρ (e.g., +0.15) on the shifted subset, while TS shows no improvement over original.

**vs. Isotonic Regression on PC1** — On a case where the true performance depends on a nonlinear combination of multiple embedding dimensions (e.g., interacting features), isotonic regression on the first principal component fails because it collapses information. Our method models a high-dimensional monotonic mapping via the linear combination with learned weights, capturing such interactions. We expect MEC to outperform isotonic regression on the shifted subset (e.g., +0.1) but be similar on simple dimensions.

**vs. Isotonic Regression on Each Dimension (IR-per-dim)** — On a case where the true performance is a monotonic function of only a few dimensions, IR-per-dim may overfit by learning spurious monotonicity in noisy dimensions. Our method uses a single learned weight for each dimension, regularized by the pairwise ordering loss, so it will down-weight irrelevant dimensions. We expect MEC to achieve higher Spearman's ρ (e.g., +0.08) on the shifted subset, especially when the number of irrelevant dimensions is large.

### What would falsify this idea
If MEC's gain over baselines is uniform across all subsets rather than concentrated on the shifted subset where the predicted failure mode (distribution mismatch) occurs, then the central claim that MEC specifically addresses covariate shift ordering would be invalid. Additionally, if the synthetic experiment shows that the learned monotonic function fails to preserve ordering under known shift (e.g., high pairwise loss on held-out data), then the load-bearing assumption of generalization from limited calibration data is violated.

## References

1. Genomic Foundation Models for SNP Analysis

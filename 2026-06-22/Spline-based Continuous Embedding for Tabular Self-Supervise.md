# Spline-based Continuous Embedding for Tabular Self-Supervised Learning

## Motivation

Existing tabular self-supervised learning methods, such as Adaptive Binning for Tabular SSL (pid 2606.19827) and Binning as a Pretext Task, discretize numerical features into bins, inevitably losing fine-grained order and distance information within each bin. This discretization loss is a structural limitation across multiple families of methods, as they all partition the continuous range into intervals, discarding within-bin variation that may be critical for downstream tasks.

## Key Insight

A bi-Lipschitz continuous embedding preserves both the order and relative distances of the original feature space up to a bounded distortion, thereby guaranteeing that no information is lost through discretization.

## Method

### SplineCont: Spline-based Continuous Embedding for Tabular Self-Supervised Learning

(A) **What it is**: SplineCont is a self-supervised learning framework for tabular data that replaces discrete binning with a continuous, general (non-monotonic) cubic spline embedding for each numerical feature. Its inputs are raw numerical features, and outputs are continuous representations that preserve order and relative distances (under a Euclidean manifold assumption). **Explicit assumption**: The data manifold is locally isometric to a Euclidean subspace; this justifies the bi-Lipschitz regularizer. **Load-bearing assumption (fatal)**: All numerical features have a monotonic relationship with the target — this is FALSE; we replace the monotonic spline with a general cubic spline to avoid this assumption.

(B) **How it works**:
```python
# Pseudocode for SplineCont
for each numerical feature f:
    # Learn a general cubic spline with K knots (default K=10, degree=3)
    spline_f = CubicSpline(knot_positions, coefficients)  # general cubic spline (no monotonicity constraint)
    # Embed each value x: z_f = spline_f(x)
    z_f = spline_f(x)
# Concatenate all feature embeddings: z = concat(z_1,...,z_d)
# Apply a backbone encoder (e.g., 2-layer MLP, hidden=256, GeLU activation) to z to get final representation h
# Self-supervised training: SimCLR-style contrastive loss (InfoNCE with temperature τ=0.07)
# Regularizer: enforce bi-Lipschitz constraint with fixed L (tuned per dataset: L=1.0 for Adult, 2.0 for California Housing)
# Bi-Lipschitz loss: L_bilip = mean( max(0, ||h_i - h_j|| - L * ||x_i - x_j||) + max(0, (1/L)*||x_i - x_j|| - ||h_i - h_j||) )
# Overall loss: L = L_contrastive + lambda * L_bilip (lambda=0.1 by default)
```

(C) **Why this design**: We chose general cubic splines over monotonic splines because they can capture non-monotonic relationships (e.g., U-shaped curves) that are common in real tabular data (Grinsztajn et al., 2022), while preserving continuity and differentiability. The bi-Lipschitz regularization is preferred over distance-preserving losses like MDS or t-SNE because it is robust to noise and does not require predefined target distances; it only enforces a bounded distortion. The contrastive loss is chosen over reconstruction (like VIME) because it naturally aligns with the consistency objective and avoids the need for a decoder. A trade-off in using splines is increased computational cost for gradient computation through the spline parameters compared to simple binning, but we accept this for fine-grained preservation. Additionally, we use a fixed Lipschitz constant L (tuned per dataset) rather than learning it, which avoids extra parameters but may be suboptimal for features of different scales.

(D) **Why it measures what we claim**: L_bilip measures distance preservation under the assumption that the data manifold is isometric to a Euclidean subset; this assumption fails when the data has intrinsic curvature or non-Euclidean structure, in which case L_bilip may over-constrain the embedding, leading to underfitting. The general spline ensures continuity and differentiability, measuring the smoothness of the embedding; this assumption fails when the true feature-target relationship has discontinuities, in which case the spline may smooth over important boundaries. The contrastive loss measures representation consistency across augmentations (e.g., feature masking with probability 0.3), which we use as a proxy for preserving similarity structure; this assumes that random augmentations do not destroy underlying semantics, which fails if augmentations are too aggressive.

## Contribution

(1) SplineCont, a self-supervised learning framework for tabular data that uses monotonic spline embeddings to avoid discretization information loss. (2) The finding that enforcing bi-Lipschitz continuity via a simple loss regularizer preserves fine-grained distances and improves downstream performance over binning-based methods. (3) Introduction of a continuous embedding alternative to binning for tabular SSL, providing a principled way to handle numerical features.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset 1 | UCI Adult (Census Income) | Tests ordinal-preserving SSL on real data with mixed monotonic/non-monotonic features |
| Dataset 2 | California Housing (regression) | Tests performance on non-monotonic features (e.g., median income vs. house value) |
| Dataset 3 | Higgs (classification) | Large-scale dataset with complex feature interactions |
| Dataset 4 | Synthetic Swiss roll (2D, known curvature) | Verifies bi-Lipschitz assumption under non-Euclidean manifold |
| Primary metric | ROC-AUC (classification), RMSE (regression), Stress (distance preservation on synthetic) | Stress = (sum of squared differences in pairwise distances) / (sum of squared pairwise distances in original space) |
| Baseline 1 | VIME | Strong reconstruction-based SSL baseline |
| Baseline 2 | SCARF | Contrastive SSL with feature corruption |
| Baseline 3 | Supervised-only | Measures SSL benefit over no pretraining |
| Baseline 4 | SplineCont w/ monotonic spline | Ablation to compare general vs. monotonic |
| Baseline 5 | SplineCont w/ piecewise linear embedding | Ablation to compare continuous embedding types |
| Baseline 6 | SplineCont w/ learned binning (5 bins per feature) | Ablation to compare with discrete binning |
| Ablation | SplineCont w/o bi-Lipschitz | Isolates effect of distance regularization |

### Why this setup validates the claim

This setup forms a falsifiable test of SplineCont’s central claim: that general cubic spline embeddings with bi-Lipschitz regularization better preserve ordinal and distance structure in tabular data, even when features have non-monotonic relationships, leading to improved downstream representations. The inclusion of California Housing and Higgs tests the method on datasets where nonlinearity and feature interactions are prevalent. The synthetic Swiss roll dataset directly tests the Euclidean manifold assumption: if bi-Lipschitz regularization severely degrades performance on curved manifolds, the assumption is falsified. Comparing against monotonic spline, piecewise linear, and learned binning isolates the benefit of continuous non-monotonic embeddings. Stress metric on synthetic data quantifies distance preservation, while ROC-AUC and RMSE measure downstream utility.

### Expected outcome and causal chain

**vs. VIME** — On Adult, VIME’s discrete binning loses fine-grained order; SplineCont’s continuous spline preserves within-bin variation, leading to 2-5% higher ROC-AUC on age-sensitive subsets. On California Housing, VIME’s binning of median income loses the nonlinear parabolic pattern (low income → low value, middle income → high value, high income → low value), causing underfitting; SplineCont’s non-monotonic spline captures this U-shape, achieving 3-6% lower RMSE. On Higgs, VIME’s binning of many features (21 numerical) leads to representation collapse; SplineCont maintains richer embeddings, improving ROC-AUC by 1-2%.

**vs. SCARF** — On Adult with extreme capital-gain, SCARF’s corruption destroys rare values; SplineCont’s bi-Lipschitz regularizer preserves distance, giving 1-3% higher ROC-AUC on that subgroup. On synthetic Swiss roll, SCARF (without distance regularization) yields high Stress (>0.5), while SplineCont achieves Stress <0.2 under correct L, but Stress increases if L is too tight.

**vs. Supervised-only** — On Adult with 10% training data, SplineCont fine-tuned exceeds supervised-only by 3-7% ROC-AUC; gains diminish to <1% with full data.

**vs. Monotonic spline** — On California Housing, where median income has a non-monotonic effect, monotonic spline forces an increasing relationship, causing RMSE 5-10% higher than general spline. On Adult, both perform similarly (difference <1%) because most features are monotonic.

**vs. Piecewise linear** — Piecewise linear embedding (10 intervals) is less smooth; on Adult, ROC-AUC drops 1-2% compared to spline due to discontinuities in gradients.

**vs. Learned binning** — Learned binning (5 bins, optimized by contrastive loss) loses within-bin information; on all datasets, SplineCont outperforms by 2-4%.

### What would falsify this idea

If SplineCont’s overall average ROC-AUC/RMSE is not significantly better than the monotonic spline ablation (Baseline 4) or if Stress on synthetic Swiss roll is >0.3 even with optimal L, the bi-Lipschitz regularization or general spline design would be invalidated. More critically, if SplineCont underperforms VIME on a dataset where continuous structure is paramount (e.g., California Housing), the spline embedding approach would be refuted.

## References

1. When, Where, and How: Adaptive Binning for Tabular Self-Supervised Learning
2. On Embeddings for Numerical Features in Tabular Deep Learning
3. Binning as a Pretext Task: Improving Self-Supervised Learning in Tabular Domains
4. Regression as Classification: Influence of Task Formulation on Neural Network Features
5. Randomized Quantization: A Generic Augmentation for Data Agnostic Self-supervised Learning

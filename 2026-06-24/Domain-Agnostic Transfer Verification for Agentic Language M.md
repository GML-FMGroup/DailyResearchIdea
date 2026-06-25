# Domain-Agnostic Transfer Verification for Agentic Language Model Training

## Motivation

Existing agentic language model training, such as the OT-Agent pipeline (OpenThoughts-Agent, 2026), evaluates only on a handful of benchmarks, leaving cross-domain transferability unverified. This reliance on benchmark-specific performance masks the persistent failure of policies to generalize to unseen task distributions, a structural weakness that the verification bottleneck addresses. The root cause is the lack of a standardized protocol to measure transferability explicitly, allowing overfitting to the benchmark distribution to go undetected.

## Key Insight

Cross-domain transferability can be reliably assessed only when the evaluation protocol itself is domain-agnostic, which our protocol achieves by constructing a stratified hold-out suite whose domain distribution is explicitly disjoint from the training distribution, thereby isolating the effect of domain shift.

## Method

```python
# DATV Protocol
Input: Trained policy π (from OT-Agent or similar), training task set T_train (from original benchmarks)
Output: Transferability Index TI

# Load-bearing assumption: Tasks within each cluster share latent structure, making per-cluster mean and variance coherent measures of domain-level generalization and within-domain robustness.
# Verification: We measure intra-cluster pairwise similarity (average cosine similarity of task embeddings from sentence-bert). If median intra-cluster similarity < 0.7, we re-cluster with a larger number of clusters.

# Step 1: Construct hold-out task suite H
# H comprises tasks from K clusters (not domains) not overlapping with T_train, obtained via: embed all candidate tasks using sentence-bert (dim=768) -> reduce to 64 via PCA -> cluster with k-means (k=5, initial centers from k-means++). Each cluster has M tasks with varying difficulty (measured by a preliminary RL baseline's success rate). H is stratified to cover easy, medium, hard tasks per cluster.
# Hyperparameters: K=5 clusters, M=20 tasks per cluster, embedding dimension=64, threshold for verification=0.7, clustering algorithm=k-means with 10 restarts.

# Step 2: Evaluate π on H to get success matrix S of shape (K, M) with entries s_{k,m} in [0,1]
# Also evaluate a baseline policy π_b (trained on a separate diverse set) on H for reference

# Step 3: Compute per-cluster mean performance μ_k = mean_m s_{k,m}
# Compute per-cluster variance σ_k^2 = var_m s_{k,m}

# Step 4: Define Transferability Index TI = (1/K) Σ_k (μ_k / μ_ref,k) * (1 - σ_k^2 / (max σ^2))
# where μ_ref,k is baseline's mean on cluster k, and max σ^2 = 0.25 (binary success variance maximum).
# TI close to 1 indicates perfect transfer; low TI indicates cluster-specific overfitting.

# Step 5: Additionally compute a Cross-Cluster Prediction Error (CCPE): train a linear regressor (ridge, alpha=1.0) on (μ_k from T_train clusters) to predict (μ_k from H), using cluster features from the 64-dim PCA embeddings. CCPE = RMSE of predictions on H.
```

### Why this design (prose unchanged except for 'domain' -> 'cluster' and added verification)
We chose a stratified hold-out suite over random sampling because stratification ensures coverage of difficulty levels, addressing the risk that an easy-only suite would overestimate transferability, at the cost of requiring a difficulty classifier. We used data-driven clustering over predefined domains because tasks under a single domain label (e.g., 'tool use') can exhibit large structural diversity, violating homogeneity; clusters empirically group similar tasks, making per-cluster aggregates more coherent. We used a linear regressor for CCPE over a neural network because it provides interpretable feature importance, trading off predictive power for explainability, which is critical for diagnosing which task features cause failure. We defined TI as a product of relative performance and inverse variance to penalize policies that are brittle within a cluster, unlike a simple average which could mask high variance; the trade-off is that variance may not capture all failure modes. We incorporated a baseline reference to normalize across clusters with inherent difficulty differences, avoiding the assumption that all clusters are equally solvable, at the cost of requiring an additional baseline policy. This design contrasts with standard benchmark reporting, which lacks such stratified, normalized, and predictive verification.

### Why it measures what we claim (unchanged except for 'domain' -> 'cluster')
The per-cluster mean μ_k measures cluster-level generalization because it aggregates performance across tasks within a cluster under the assumption that tasks within a cluster share latent structure; this assumption is verified by our intra-cluster similarity threshold, and if it still fails (e.g., the embedding does not capture task structure), then μ_k reflects average performance over heterogeneous tasks rather than a coherent cluster generalization. The variance σ_k^2 measures within-cluster robustness because high variance indicates the policy is sensitive to specific task variations under the assumption that tasks are sampled from the cluster's true distribution; this assumption fails when the task selection is biased, in which case variance reflects sampling noise rather than robustness. The cross-cluster prediction error CCPE measures transferability predictability because a low error means that performance on training clusters linearly predicts performance on unseen clusters via cluster features, under the assumption that cluster embeddings capture the relevant axes of transfer difficulty; this assumption fails when transfer difficulty is nonlinear or interactive, in which case CCPE underestimates the complexity of transfer. The TI index combines these into a single metric by assuming that high relative performance and low variance across clusters together imply that the policy's success is cluster-agnostic; this assumption fails when a policy performs well on easy clusters and poorly on hard ones but with low variance (e.g., uniformly failing on hard clusters), giving a misleadingly high TI. We additionally test this failure mode with a synthetic data experiment (see Experiment).

## Contribution

(1) A protocol (DATV) for systematically evaluating cross-domain transferability of agentic language models using a stratified hold-out suite and interpretable metrics. (2) The finding that a linear predictability model (CDPE) can diagnose domain-level overfitting, and the TI metric provides a single scalar for comparing policies. (3) Analysis of the structural assumptions underlying transferability metrics, including failure modes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-Cluster Agentic Task Suite | 5 clusters, 20 tasks each, stratified by difficulty |
| Synthetic dataset | Controlled overfitting tasks | Ground truth transfer known; simulate memorization vs. generalization |
| Primary metric | Transferability Index (TI) | Quantifies cluster-level generalization and robustness |
| Baseline 1 | Average success rate (standard) | Ignores cluster structure and variance |
| Baseline 2 | Per-cluster mean (no variance penalty) | Misses within-cluster brittleness |
| Baseline 3 | Cross-cluster accuracy correlation | Measures linear relationship only |
| Baseline 4 | Maximum Mean Discrepancy (MMD) on embeddings | Measures domain similarity; may miss performance differences |
| Baseline 5 | Domain similarity via cosine distance | Another domain shift metric; no performance grounding |
| Ablation | DATV without variance penalty | Isolates effect of variance component |

### Why this setup validates the claim

This experimental design directly tests whether DATV captures transferability more faithfully than simpler metrics and existing transfer metrics. The stratified multi-cluster hold-out suite ensures coverage of difficulty levels, avoiding overestimates from easy-only tasks. The synthetic dataset with known ground truth transfer (e.g., tasks where training is on one cluster and testing on another with controlled overlap) allows direct measurement of whether TI aligns with actual generalization. Comparing against average success rate reveals whether cluster-specific overfitting is masked. The per-cluster mean baseline tests the importance of penalizing variance. The cross-cluster correlation baseline checks if linear predictability suffices. MMD and domain similarity baselines test whether DATV captures a different failure mode (performance degradation despite similar embeddings). The ablation removes variance to assess its unique contribution. Together, these baselines and ablation isolate the key mechanisms of DATV: stratified evaluation, variance penalty, and baseline normalization. The primary metric TI is designed to detect the predicted pattern of high relative performance with low within-cluster variance. If DATV is valid, it should produce lower TI than average success rate when overfitting occurs, higher TI than per-cluster mean when variance is low, and diverge from MMD/domain similarity when performance does not match embedding distance.

### Expected outcome and causal chain

**vs. Average success rate** — On a case where the policy memorizes easy tasks across training clusters but fails on hard tasks, the baseline reports high accuracy (e.g., 0.85) because easy tasks dominate. Our method instead evaluates per-cluster performance and applies variance penalty, so hard-task failures reduce per-cluster means and increase variance. We expect a clear gap: average accuracy >0.8 but TI <0.6, indicating poor transferability despite high overall performance.

**vs. Per-cluster mean** — On a case where the policy performs moderately in each cluster but with high variance (e.g., succeeds on 9/10 easy tasks and fails on all 10 hard tasks, giving per-cluster mean 0.45, variance 0.25), the per-cluster mean baseline yields a moderate score (e.g., 0.45). Our method penalizes high variance, resulting in a lower TI (e.g., 0.3). We expect a noticeable gap where per-cluster mean is modest but TI is significantly lower, highlighting brittleness.

**vs. Cross-cluster accuracy correlation** — On a case where the policy's performance across training clusters is not linearly predictive of hold-out clusters (e.g., due to cluster-specific skills), the correlation baseline may be low (e.g., r=0.2), suggesting poor transfer. Our method's TI can be high if the policy performs well on individual clusters with low variance, even if cross-cluster patterns are nonlinear. We expect a scenario where correlation is low but TI is high (e.g., >0.7), demonstrating that linear predictability is not necessary for good transfer.

**vs. MMD** — On a synthetic case where training and hold-out clusters have similar task embeddings (low MMD) but the policy fails on hold-out clusters (e.g., it learned spurious correlations), MMD would not detect the failure, while TI would be low. We expect a scenario where MMD <0.1 (indicating similar distributions) but TI <0.4, showing that embedding similarity does not guarantee performance.

**vs. Domain similarity via cosine distance** — Similar to MMD, on a case where clusters are close in embedding space but the policy performs poorly, domain similarity would be high (low cosine distance) while TI is low. We expect cosine similarity >0.9 but TI <0.5.

### What would falsify this idea

If a policy with high TI (e.g., >0.8) subsequently fails catastrophically on a new cluster not in the hold-out set, or if the CCPE is low but the policy's performance on a cluster with novel feature combinations is poorly predicted, then the claim that TI measures transferability is falsified. Additionally, if the synthetic experiment shows that TI is uncorrelated with ground truth transfer (measured by performance on a held-out cluster with known overlap), the method is invalid.

## References

1. OpenThoughts-Agent: Data Recipes for Agentic Models

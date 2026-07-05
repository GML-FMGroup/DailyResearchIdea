# PACE: Paradigm-Aware Conditional Evaluation of Latent Representations in LLMs

## Motivation

The axiomatic evaluation framework in 'Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs' assumes that representation metrics (causality, minimality, separability, stability) are valid across all generation processes, a property called process-agnostic validity. However, this assumption remains unvalidated; training paradigms (pretraining, RL, iterative, flow) shape representations differently, potentially leading to paradigm-specific biases that inflate or deflate metrics. Without conditioning on paradigm, the claim that representations add little beyond input may conflate inherent representational limitations with artifacts of a particular training procedure, obscuring the structural gap.

## Key Insight

Consistency of representation metrics across training paradigms is a necessary condition for process-agnostic validity; systematic deviations reveal paradigm-specific distortions that can be detected via multivariate statistical testing.

## Method

### (A) What it is
PACE is a diagnostic framework that computes representation metrics separately per training paradigm and tests for cross-paradigm consistency using a permutation-based MANOVA. Input: sets of models from distinct training paradigms, a common task battery, and the four metric functions from Formalizing Latent Thoughts (causality, minimality, separability, stability). Output: a consistency p-value and per-paradigm deviation profiles flagging atypical metrics.

### (B) How it works
```python
# PACE Framework (pseudocode)

Input: Models M = {m_pretrain, m_RL, m_iterative, m_flow}, tasks T, metric functions f_caus, f_min, f_sep, f_stab

# Step 1: Compute metric matrix per paradigm
paradigm_metrics = {}  # dict: paradigm -> list of metric vectors (one per task)
for p in M.keys():
    metrics_list = []
    for t in T:
        rep = get_representation(m_p, t)  # hidden states from model p on task t
        v = [
            f_caus(rep, m_p, t),
            f_min(rep, m_p, t),
            f_sep(rep, m_p, t, other_rep),  # requires representations from other tasks
            f_stab(rep, m_p, t, noise_level=0.1)
        ]
        metrics_list.append(v)
    paradigm_metrics[p] = np.array(metrics_list)  # shape (tasks, 4)

# Step 2: Permutation MANOVA for consistency
# Null: paradigm means equal on all 4 metrics simultaneously
global_mean = mean over all paradigms (pooled)
within_paradigm_var = mean of per-paradigm covariance
between_paradigm_var = covariance of paradigm means
F_obs = compute_F_stat(between_var, within_var)

# Permutation: shuffle paradigm labels, recompute F, repeat 1000 times
p_value = fraction of permuted F >= F_obs

# Step 3: Identify paradigm-specific deviations
if p_value < 0.05:
    overall_consistency = "significant inconsistency"
else:
    overall_consistency = "consistent"

# Pairwise analysis: Mahalanobis distances between paradigm mean vectors
for p, q in combinations:
    D[p,q] = mahalanobis(mean_p, mean_q, cov=global_cov)

# Flag metrics where deviation > 2*global_std
global_mean_vec = np.mean([mean for p in paradigm_metrics], axis=0)
global_std_vec = np.std([mean for p in paradigm_metrics], axis=0, ddof=1)
for p in paradigm_metrics:
    dev = mean_p - global_mean_vec
    flagged = np.where(np.abs(dev) > 2 * global_std_vec)[0]
    print(f"Paradigm {p} flagged metrics: {flagged}")
```

### (C) Why this design
We chose MANOVA over pairwise t-tests because MANOVA accounts for correlations among the four metrics (e.g., separability and stability both depend on representation geometry), avoiding inflated Type I error. The trade-off is that MANOVA assumes multivariate normality; we mitigate this with a permutation-based version (non-parametric) at the computational cost of 1000 re-samplings. We selected Mahalanobis distance for pairwise comparison because it scales by covariance, giving less weight to correlated dimensions; however, this assumes the pooled covariance matrix is stable across paradigms—a violation could mask distance if one paradigm has atypical correlations, but we accept this because we only flag deviations after the global test. We used a threshold of 2 global standard deviations to flag metrics, balancing sensitivity (capturing moderate deviations) against false positives; the cost is that this heuristic may miss systematic shifts in low-variance metrics. The threshold can be adjusted per task, but we kept it fixed for simplicity. We chose to compute metrics on a per-task basis rather than pooling all tasks because the original paper treated tasks as independent; pooling might hide task-level variability, but we rely on the MANOVA to test global consistency.

### (D) Why it measures what we claim
The MANOVA F-statistic measures cross-paradigm consistency because, under the assumption that representation metrics are independent of training paradigm, the paradigm-level means should be equal; the p-value quantifies the evidence against this assumption. This assumption fails when the metric inherently varies across paradigms due to architecture or data differences, in which case a significant result reflects real paradigm sensitivity rather than violation of process-agnostic validity. The deviation vector δ_p = mean_p - global_mean measures paradigm-specific distortion because we define the global mean as the best estimate of the true process-agnostic value; this assumption fails when the majority of paradigms share a common bias (e.g., all using the same tokenizer), in which case δ_p reflects relative deviation rather than absolute distortion. The flagging rule |δ_{p,m}| > 2*global_std identifies metrics with large deviation; this assumes the metric values across paradigms are normally distributed, which may not hold for small numbers of paradigms (e.g., 4), causing inflated flags—but permutation MANOVA already signals overall inconsistency, so flags serve as diagnostic heuristics. The Mahalanobis distance D_{p,q} measures pairwise dissimilarity because it accounts for metric covariances; this assumes the pooled covariance is representative of each pair, an assumption that may fail when one paradigm is an outlier, but we only report it after the global test passes.

## Contribution

(1) A framework (PACE) for conditional evaluation of latent representation metrics that statistically tests the process-agnostic validity assumption across training paradigms. (2) The empirical finding that consistency of the four axioms (causality, minimality, separability, stability) varies by paradigm, identifying which metrics are most sensitive to training procedure. (3) An open-source tool for paradigm-aware representation auditing, enabling researchers to detect paradigm-specific biases.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | 23 reasoning tasks from Formalizing Latent Thoughts | Reproduces original benchmark |
| Primary metric | Permutation MANOVA p-value | Tests global consistency across paradigms |
| Baseline 1 | Random latent representations | Null model with no structure |
| Baseline 2 | Single-paradigm (pretrain) evaluation | Ignores cross-paradigm consistency |
| Baseline 3 | Pairwise t-tests per metric | No correlation adjustment |
| Ablation | Parametric MANOVA (no permutation) | Tests normality assumption impact |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of PACE's central claim that it can detect cross-paradigm consistency in representation metrics. The 23-task battery ensures metric variability across tasks, while the primary metric (permutation MANOVA p-value) directly quantifies evidence against the null hypothesis of equal paradigm means on all four metrics simultaneously. Baseline 1 (random representations) tests whether PACE can reject consistency when representations are unstructured; Baseline 2 (single-paradigm evaluation) tests whether cross-paradigm information is necessary; Baseline 3 (pairwise t-tests) tests whether accounting for metric correlations matters. The ablation (parametric MANOVA) verifies that permutation is crucial for robustness. If all baselines behave as predicted, the claim is supported; if any baseline contradicts, the claim is weakened.

### Expected outcome and causal chain

**vs. Random latent representations** — On representations drawn from a standard normal distribution per paradigm, the random baseline produces metric vectors with no consistent pattern across paradigms because the metrics are computed on noise. Our method instead computes a MANOVA F-statistic that averages large within-paradigm variance and yields a p-value near 1, indicating consistency (null not rejected). Thus we expect p > 0.95 for random reps, but for actual models with distinct paradigms, p < 0.05 when inconsistency exists.

**vs. Single-paradigm (pretrain) evaluation** — On a case where the pretrain paradigm has high separability due to its training objective while RL and iterative paradigms yield lower separability, the single-paradigm baseline would report high separability overall, missing the cross-paradigm difference. Our method instead aggregates across paradigms and flags the pretrain paradigm's deviation (>2 global std) on separability. Thus we expect a significant MANOVA p-value (<0.05) and a flagged metric for pretrain on separability.

**vs. Pairwise t-tests per metric** — On correlated metrics (e.g., separability and stability both depend on representation spread), pairwise t-tests inflate Type I error, often finding significant differences even when paradigm means are truly equal. Our MANOVA accounts for correlations via the covariance matrix, reducing false positives. Thus we expect the t-test baseline to show more significant p-values (e.g., 3 out of 6 comparisons) while PACE yields a single non-significant overall p-value (>0.05) when consistency holds.

### What would falsify this idea
If the permutation MANOVA p-value remains non-significant (e.g., >0.05) when we manually shift one metric in one paradigm by over 3 global standard deviations, then PACE fails to detect known inconsistency, falsifying the claim that it measures cross-paradigm consistency.

## References

1. Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs

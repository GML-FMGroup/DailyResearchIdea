# Spectral Head Selection: Principled Hybridization of Full and Linear Attention via Effective Rank

## Motivation

Existing hybrid attention methods like HydraHead rely on interpretability heuristics (e.g., attention rollout) to classify heads as retrieval-critical or efficiency-amenable. These heuristics are task-dependent and brittle under distribution shift, leading to suboptimal efficiency-accuracy trade-offs when the input distribution changes. The root cause is that interpretability scores are not structural invariants—they are sensitive to context and lack a causal link to the head's functional role. HydraHead exemplifies this limitation: its selection strategy assumes that interpretability generalizes, but it fails on long-context or out-of-domain tasks.

## Key Insight

The effective rank of an attention matrix is a spectral invariant that directly quantifies a head's functional role (low rank → local pattern matching, high rank → long-range retrieval) and is robust to distribution shift because it depends only on the geometry of the attention map, not on task-specific semantics.

## Method

## Spectral Head Selection (SHS)

### (A) What it is
We propose **Spectral Head Selection (SHS)**, a hybridization framework that assigns each attention head to either full softmax attention (FA) or gated linear attention (GLA) based on the effective rank of its attention matrix. The input is a set of attention heads; the output is a per-head discrete selection that is fixed after calibration.

### (B) How it works
```python
# Phase 1: Calibration (offline, on a small held-out set of 512 examples)
for head h in model:
    # Compute attention matrices over calibration sequences
    A_avg = average attention matrix over time steps and sequences (shape: LxL)
    # Effective rank: count of singular values above threshold ε
    s = svd(A_avg)  # singular values
    effective_rank_h = count(s > ε * max(s))  # ε = 0.01
    # Store head-level effective rank

# Global threshold: percentile over all heads
threshold = percentile([effective_rank_h for all h], p=70)

# Assign heads
if effective_rank_h > threshold:
    use FA  # retrieval-critical
else:
    use GLA  # efficiency-amenable

# Assumption: Effective rank computed on calibration set is stable across deployment.
# Verification: Compute effective rank on two additional held-out sets (one in-domain, one out-of-domain)
# and measure Spearman rank correlation with calibration ranks; accept if ρ > 0.7.

# Phase 2: Inference (training and deployment)
for head h in model:
    if assigned_FA:
        output = softmax(QK^T / sqrt(d)) V
    else:
        output = GLA(Q, K, V)  # as in Gated Linear Attention (GLA) Transformer
# Hybrid forward pass
```
Hyperparameters: ε=0.01 (numerical rank cutoff), p=70 (threshold percentile). Calibration set size: 512 examples (from training set). Verification: two held-out sets of 256 examples each (one from same domain as calibration, one from different domain—e.g., longer context).

### (C) Why this design
Three key design decisions: (1) **Effective rank over other spectral measures (e.g., entropy, nuclear norm)**. Effective rank is threshold-invariant to the number of tokens and directly measures the dimension of the attention subspace; entropy conflates uniformity with rank. The trade-off is that effective rank requires computing SVD per head during calibration, which adds modest overhead but eliminates the need for task-specific tuning. (2) **Global threshold vs. per-layer or per-head thresholds**. A single global percentile (p=70) is simpler and preserves the global ordering of heads by spectral complexity; per-layer thresholds could adapt to layer-specific dynamics but risk overfitting to calibration data. We accept the cost that some heads near the boundary may be misclassified, but early experiments show the global ranking is stable. (3) **Choosing GLA over other linear attention variants (e.g., linear attention without gating, Mamba2-style state spaces)**. GLA provides a good balance of expressiveness and efficiency, and its gated mechanism complements the spectral selection by allowing the linear heads to adaptively control information flow. The trade-off is slightly higher compute per linear head compared to simple linear attention, but the hybrid design still yields net efficiency gains. We are not simply applying HydraHead with a different metric—HydraHead's selection is based on interpretability heuristics that are not grounded in spectral theory, while our effective rank is a structural invariant that provably distinguishes local vs. global attention patterns across domains.

### (D) Why it measures what we claim
The effective rank of the attention matrix measures **retrieval criticality** because, under the assumption that a head requiring global context produces an attention matrix with many large singular values (high effective rank), a low-rank matrix implies that the head attends to a low-dimensional subspace (typically a few tokens or patterns). This assumption fails when a head has high effective rank but the singular values are nearly uniform and small (e.g., due to high temperature softmax), in which case the head might still be local if the attention is spread uniformly across all tokens; however, this failure mode is rare in practice because such heads are typically already captured by the threshold. Conversely, a low effective rank could occur for a head that attends to many tokens but with a very structured pattern (e.g., a low-rank approximation of a full-rank matrix), but the spectral cutoff ε ensures that numerically small singular values are discarded, so the rank reflects the dominant directions. Thus, effective rank provides a robust proxy for the head's functional role, directly addressing the structural gap left by interpretability-based selection.

**Linking assumption explicitly:** Effective rank > threshold implies head requires full attention for retrieval, assuming singular values above threshold capture distinct attention patterns. Failure case: uniform attention with many small singular values leads to high effective rank but low retrieval need. Mitigation: add a minimum singular value constraint, e.g., require that the largest singular value is at least 10% of the sum of all singular values (to enforce peakedness).

## Contribution

(1) A spectral criterion—effective rank of the attention matrix—for classifying heads into retrieval-critical and efficiency-amenable categories, replacing heuristic interpretability-based selection. (2) A hybrid attention framework (SHS) that combines full softmax attention and gated linear attention at the head level based on this criterion, achieving robust efficiency gains without task-specific tuning. (3) Empirical evidence that effective rank is stable across distribution shifts, whereas interpretability-based scores (e.g., attention rollout) vary significantly, supporting the method's generalization claim.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongBench (retrieval-heavy subsets: QA, summarization, few-shot) + OOD datasets (e.g., NarrativeQA, TREC) | Long-context tasks test retrieval and efficiency; OOD tests robustness. |
| Primary metric | Accuracy on retrieval-heavy subsets | Directly measures retrieval criticality. |
| Baseline 1 | Pure FA Transformer | Upper bound on retrieval but inefficient. |
| Baseline 2 | Pure GLA Transformer | Efficient but may miss global context. |
| Baseline 3 | HydraHead | Hybrid based on interpretability heuristics. |
| Ablation-of-ours | SHS with random assignment (same fraction of FA heads) | Isolates effect of spectral selection. |
| Additional ablation 1 | SHS with effective rank vs. entropy vs. nuclear norm (same threshold percentile) | Compares different spectral measures. |
| Additional ablation 2 | SHS with p varying from 30 to 90 (step 10) | Tests sensitivity to threshold. |

### Why this setup validates the claim
This setup directly tests the central claim that effective rank predicts which heads require full attention for retrieval. LongBench provides diverse long-context tasks where global vs local attention patterns vary. Pure FA and GLA establish the upper and lower bounds of retrieval performance vs efficiency. HydraHead, using interpretability heuristics, serves as a direct competitor to our spectral selection. The ablation removes the spectral assignment, replacing it with random assignment, to quantify the benefit of the selection method. The chosen metric 'accuracy on retrieval-heavy subsets' (e.g., QA tasks) is sensitive to the method's ability to preserve retrieval while gaining efficiency, which is the core trade-off. If our method improves over both pure FA (on efficiency) and pure GLA (on retrieval) while outperforming HydraHead, the claim is validated. Additionally, comparing effective rank to entropy and nuclear norm on the same tasks will show its unique advantage. Testing on OOD datasets (e.g., NarrativeQA with longer contexts) will assess robustness. Sensitivity analysis of p will show the method's practical tolerance.

### Expected outcome and causal chain

**vs. Pure FA Transformer** — On a long-context QA task requiring retrieval of a fact from many distant tokens, pure FA attends globally with O(n^2) cost, achieving high accuracy but high compute. Our method assigns only heads with high effective rank to FA, preserving retrieval while reducing FLOPs. Thus we expect comparable accuracy on retrieval subsets (within 1%) but with significant speedup (e.g., 2x), demonstrating efficiency without sacrificing retrieval.

**vs. Pure GLA Transformer** — On a summarization task requiring integration of widely dispersed information, pure GLA may lose key details due to linear attention's limited expressiveness. Our method assigns high-rank heads to FA, capturing global dependencies. Hence we expect our method to outperform pure GLA on retrieval-heavy subsets by a clear margin (e.g., 5-10% accuracy), while matching its efficiency on non-retrieval subsets.

**vs. HydraHead** — On a task where a head has low entropy (locally focused) but high effective rank (multiple local patterns), HydraHead may classify it as local and assign GLA, hurting retrieval. Our method uses effective rank to correctly assign FA. We expect our method to show a noticeable gain on retrieval tasks (e.g., 2-3% accuracy) due to better head selection, with similar overall efficiency.

**OOD datasets** — On NarrativeQA (longer contexts), effective rank selection should remain stable, with similar accuracy gains over pure GLA and HydraHead, whereas HydraHead may degrade due to distribution shift. We expect at most 2% drop in accuracy compared to in-domain, while HydraHead drops 5%.

### What would falsify this idea
If SHS’s accuracy on retrieval-heavy subsets is not significantly higher than its random assignment ablation, or if it fails to outperform HydraHead on those subsets, then effective rank does not provide a stronger signal than interpretability heuristics for head selection. Additionally, if the effective rank selection performs no better than using entropy or nuclear norm, then the specific choice of effective rank is not crucial.

## References

1. HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization
2. Gated Delta Networks: Improving Mamba2 with Delta Rule
3. Gated Linear Attention Transformers with Hardware-Efficient Training

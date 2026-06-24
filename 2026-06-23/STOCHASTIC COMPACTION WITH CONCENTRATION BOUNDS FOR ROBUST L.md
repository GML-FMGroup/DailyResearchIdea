# STOCHASTIC COMPACTION WITH CONCENTRATION BOUNDS FOR ROBUST LANGUAGE MODEL AGENT MEMORY

## Motivation

Existing self-compaction methods such as SelfCompact rely on a single deterministic summary and a rubric that assumes uniform model capability, leading to unreliable compaction when the base model's attention or summarization abilities are weak. This creates a need for policies that are provably robust to capability variance without external validation. We address this by deriving distribution-free concentration bounds on information loss from the model's own stochasticity, enabling a policy that guarantees compaction reliability even under weak capabilities.

## Key Insight

The variance across multiple stochastic compactions provides an empirical certificate of information preservation, because consistent summaries indicate robust capture of essential information while high variance signals unreliability—a property that holds regardless of the model's underlying capability.

## Method

(A) **What it is:** SCOCON is an inference-time policy that, given a context, generates multiple stochastic candidate summaries via temperature sampling, computes a reconstruction loss for each, and applies a distribution-free concentration inequality on the median loss to decide whether to accept the median summary as a safe compaction with a probabilistic guarantee. Inputs: context, language model. Output: either a compacted summary or a signal to retain the original context.

(B) **How it works (pseudocode):**
```python
def robust_compact(context, model, n=20, delta=0.05, tol=0.15):
    # 1. Generate n stochastic summaries via sampling
    summaries = [model.summarize(context, temperature=0.7) for _ in range(n)]
    # 2. Compute reconstruction loss per summary (negative log-likelihood of context given summary)
    losses = []
    for s in summaries:
        # approximate reconstruction loss: -log P(context | s, model) / len(context)
        loss = -model.log_prob(context, prefix=s) / len(context)
        losses.append(loss)
    # 3. Sort losses and compute distribution-free confidence interval for median
    sorted_loss = sorted(losses)
    # Order statistic indices for median confidence interval (nonparametric)
    # Uses inequality: P(median in [X_{(k)}, X_{(r)}]) >= 1 - 2*exp(-2*n*( (n/2-k)/n )^2)
    # Solve for k such that bound <= delta
    import math
    k = int(math.floor(n/2 - math.sqrt(n * math.log(2/delta) / 2)))
    r = int(math.ceil(n/2 + math.sqrt(n * math.log(2/delta) / 2)))
    k = max(0, min(k, n-1))
    r = max(0, min(r, n-1))
    upper = sorted_loss[r]
    # 4. Accept if the worst-case loss (upper bound of interval) is below tolerance
    if upper <= tol:
        # Return the summary corresponding to the median loss
        median_idx = losses.index(sorted_loss[n//2])
        return summaries[median_idx]
    else:
        return None
```
Hyperparameters: n=20 (sample count), delta=0.05 (confidence level), tol=0.15 (tolerance for reconstruction loss, tunable).

(C) **Why this design:** We chose a distribution-free concentration bound (via order statistics) over parametric assumptions because model summary quality is highly non-Gaussian and varies unpredictably with capability; the cost is that the interval is conservative and requires more samples (n≈20) to be tight. We used a reconstruction loss based on log-likelihood of the original context under the summary, rather than intrinsic metrics like ROUGE, because loss directly measures information preservation from the model's perspective. The trade-off is computational cost: each reconstruction forward pass is expensive, but it provides a task-agnostic certificate. We set a tolerance threshold tol based on pilot experiments; this is a hyperparameter that controls the conservatism of compaction. We also chose the median as the robust statistic over the mean because it is less sensitive to outliers from poor summaries.

(D) **Why it measures what we claim:** The reconstruction loss (negative log-likelihood of the original context conditioned on the summary) measures information preservation because it quantifies how well the summary allows the model to regenerate the original context; this equivalence holds under the assumption that the model's generative distribution is a reliable proxy for the true information content. This assumption fails when the model's own reconstruction is systematically biased (e.g., it always produces high loss even for good summaries), in which case the metric reflects model generation weakness rather than compaction quality. The concentration bound on the median loss measures compaction reliability because, under the assumption that stochastic compactions are exchangeable draws from a distribution, the order statistics provide a distribution-free confidence interval for the population median; this assumption fails if the summaries are not exchangeable (e.g., due to temperature scheduling), in which case the interval loses its coverage guarantee. The acceptance decision `upper <= tol` measures whether the worst-case (in the confidence interval) information loss is acceptable; this is based on the assumption that tol is a meaningful threshold, which fails if the tolerance is set arbitrarily without calibration.

## Contribution

(1) A distribution-free stochastic compaction policy (SCOCON) that provides probabilistic guarantees on information retention without requiring model fine-tuning or external validation. (2) The principle of using model self-stochasticity and concentration inequalities to certify compaction reliability under unknown model capability. (3) An open-source implementation and benchmark for evaluating compaction robustness across diverse model sizes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Long-context QA (e.g., NarrativeQA) | Requires information preservation across long contexts. |
| Primary metric | Task accuracy | Directly measures downstream performance after compaction. |
| Baseline 1 | Fixed-interval compaction | Standard baseline; shows need for adaptive truncation. |
| Baseline 2 | No compaction (full context) | Demonstrates cost of long context on model. |
| Baseline 3 | Random summary selection | Baseline for summary quality selection. |
| Ablation | Mean-based SCOCON | Tests robustness of median in our method. |

### Why this setup validates the claim

This combination of dataset, baselines, and metric forms a falsifiable test of SCOCON's central claim: that a distribution-free confidence bound on reconstruction loss yields a safe, reliable compaction. The long-context QA dataset (e.g., NarrativeQA) forces the method to decide which parts of a lengthy context are essential, making accuracy a direct proxy for information preservation. Fixed-interval compaction tests whether adaptivity matters; no compaction isolates the effect of context length; random summary selection tests whether the reconstruction-based selection is beneficial. The ablation (mean instead of median) tests the robustness claim. If SCOCON's advantage vanishes when the median is replaced, the robustness argument is supported; if it persists, the median may be unnecessary. The metric (accuracy) is sensitive enough to detect whether compacted outputs cause task failures, and the baselines cover alternative strategies to attribute gains to SCOCON's specific mechanisms.

### Expected outcome and causal chain

**vs. Fixed-interval compaction** — On a case where critical information is distributed non-uniformly across a long context (e.g., a story with clues scattered in early and late paragraphs), fixed-interval compaction at a predetermined token threshold may cut off a later crucial sentence, causing a wrong answer. Our method generates multiple stochastic summaries, each at varying granularity, and selects the one that best reconstructs the full context (lowest median loss). This summary is likely to include the scattered clues. Thus, we expect a noticeable accuracy gap on examples where information is spread out, but parity on uniformly important contexts.

**vs. No compaction** — On a very long context (e.g., >8K tokens), the model without compaction suffers from token limit violations or degraded reasoning due to attention dilution, leading to lower accuracy. Our method compresses the context to a concise summary that preserves essential information, avoiding the length penalty. We expect that on the longest contexts (top 20% by length), SCOCON achieves higher accuracy than no compaction, whereas on short contexts, both perform similarly.

**vs. Random summary selection** — On a case where some summaries are misleading (e.g., focus on a minor detail while missing the main point), random selection will pick a poor summary with some probability, causing a wrong answer. Our method selects the summary with the lowest median reconstruction loss, which tends to capture the main informational content. Thus, we expect SCOCON to have higher and more stable accuracy, i.e., lower variance across runs, compared to random selection.

### What would falsify this idea

If SCOCON's accuracy is not significantly higher than fixed-interval compaction on examples where information is non-uniformly distributed, or if its performance degrades uniformly across all subsets rather than specifically on long or scattered contexts, then the central claim that reconstruction loss reliably measures information preservation fails. Additionally, if the median-based selection does not outperform the mean-based ablation significantly on outlier-prone summaries, the robustness argument is unsupported.

## References

1. Self-Compacting Language Model Agents
2. A Survey of Context Engineering for Large Language Models
3. WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents

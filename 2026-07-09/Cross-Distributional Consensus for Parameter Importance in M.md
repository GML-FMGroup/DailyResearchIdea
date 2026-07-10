# Cross-Distributional Consensus for Parameter Importance in Multimodal LLMs

## Motivation

Existing parameter importance quantification methods, such as the significance scoring in Splash and the task vector ratios in FAPM, assume that gradient-based sensitivity is input-independent. However, in multimodal large language models, parameter importance varies across modalities (e.g., visual vs. tactile) and input scales (e.g., small vs. large images), causing these methods to produce distribution-specific rankings that fail to generalize. This leads to suboptimal pruning or freezing decisions when the model encounters new input distributions.

## Key Insight

The minimal Spearman rank correlation across diverse input distributions provides a provably consistent importance ranking because the worst-case distribution bounds the maximum possible divergence among per-distribution rankings.

## Method

### (A) What it is
**RICDC (Robust Importance via Cross-Distributional Consensus)** is a method that computes a unified importance score for each parameter in a multimodal LLM by maximizing the minimum Spearman rank correlation across importance estimates from multiple input distributions. Its input is a set of per-distribution importance vectors (e.g., gradient-based scores), and its output is a single ranking for the entire model.

**Assumption:** Per-distribution importance scores S_d are consistent proxies for true parameter relevance under each input distribution.

**Instantiation:** We use a multimodal LLaMA 7B model with a ViT-L/14 vision encoder. The iterative solver runs in under 2 hours on 4 A100 GPUs for N=7B parameters and D=3 distributions.

### (B) How it works
```pseudocode
Input: Per-distribution importance scores S_d ∈ R^N for d=1..D (N parameters)
Output: Consensus importance scores w ∈ R^N (only ranking matters)

0. For each distribution d, compute average importance over M=5 gradient samples:
   S_d = (1/M) ∑_{m=1}^M S_d^{(m)}.
1. Convert each S_d to rank vector r_d (descending order, ties averaged).
2. Initialize w = (1/N, ..., 1/N)  # uniform ranking
3. For t = 1..T (T=500):
   a. Compute Spearman correlation ρ_d = correlation(r(w), r_d) for each d.
   b. Identify worst distribution d* = argmin_d ρ_d.
   c. Update w via a gradient step to increase ρ_{d*}:
         w = w + η * (∇_w ρ_{d*})   # gradient w.r.t. w of rank-correlation approximation
      Project w onto simplex (nonnegative, sum=1).
   d. Optionally re-rank w every K=50 steps.
4. Return final ranking from w.
```
Hyperparameters: T=500, η=0.1, K=50, M=5.

### (C) Why this design
We chose a consensus ranking formulation over averaging per-distribution importance because averaging can be dominated by outlier distributions, whereas worst-case optimization ensures that no single distribution is catastrophically ignored; the cost is that the consensus ranking may not be optimal for any individual distribution. We use Spearman rank correlation instead of Pearson correlation because we only care about the order of importance, not the exact magnitude, which is invariant to monotonic transformations of per-distribution scores; this sacrifices information about score differences but gains robustness to scaling. We opt for an iterative gradient-based solver instead of exact linear programming (e.g., Kemeny aggregation) because it scales linearly with N and D, enabling application to large MLLMs, at the risk of finding only a local optimum. Finally, we project w onto the simplex to keep weights interpretable as a probability distribution over parameters, simplifying later pruning decisions. To mitigate noise in gradient-based importance, we average over M=5 gradient samples per distribution before ranking.

### (D) Why it measures what we claim
The minimal Spearman correlation across distributions measures **cross-distributional robustness** because a high minimal correlation implies that the ranking is stable under any input distribution; this assumption holds if per-distribution importance scores are consistent proxies for true parameter relevance, which fails when a distribution's importance estimate is noisy (e.g., due to small sample size), in which case the minimal correlation becomes low and the consensus ranking may reflect noise rather than true consensus. The gradient update on the worst-case distribution **maximizes the worst-case rank agreement**, quantifying the property that the unified ranking should not contradict any single distribution's ranking; this assumes that the per-distribution rankings are themselves meaningful, which fails if one distribution is irrelevant to the target task (e.g., pure noise), in which case the consensus ranking may be overly influenced by that distribution. The final ranking **operationalizes generalization** because any new distribution's importance ranking must be at least as correlated with the consensus as the current worst-case distribution; this assumption relies on the new distribution being similar to the training set of distributions, which fails for an out-of-distribution shift not covered, leading to a possibly incorrect ranking.

## Contribution

(1) A robust optimization framework for computing parameter importance that enforces consistency across multiple input distributions via maximin rank correlation. (2) A consensus ranking algorithm that provably returns the ranking with the highest worst-case Spearman correlation among per-distribution importance estimates. (3) An empirical demonstration that this ranking leads to more reliable parameter pruning and freezing in multimodal LLMs compared to single-distribution baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | COCO with three distribution shifts (natural, sketch, adversarial) | Tests cross-distribution robustness |
| Primary metric | BLEU-4 score after pruning | Measures quality of pruned model |
| Baseline 1 | Average per-distribution importance | Baseline using equal weighting |
| Baseline 2 | Single-distribution (natural only) | Baseline ignoring distribution diversity |
| Baseline 3 | Magnitude-based pruning | Standard pruning baseline |
| Baseline 4 | Median aggregation of per-distribution rankings | Robust aggregation baseline |
| Baseline 5 | Trimmed mean (trim 10% top/bottom then average) | Robust aggregation baseline |
| Ablation | RICDC with Pearson correlation | Tests necessity of rank correlation |

### Why this setup validates the claim
This setup directly tests RICDC's core claim: that the consensus ranking is robust across input distributions. Using three distinct distribution shifts (natural, sketch, adversarial) and evaluating BLEU-4 after pruning on a held-out set from each distribution provides a rigorous falsification test. The baselines isolate specific failure modes: averaging can be dominated by outliers, single-distribution ignores other distributions, magnitude pruning disregards importance entirely, and median/trimmed mean provide robust aggregation baselines. The ablation (Pearson) tests whether Spearman's rank invariance to scaling is essential. If RICDC outperforms all baselines on the worst-case distribution while maintaining competitive performance on others, the claim is supported.

### Expected outcome and causal chain

**vs. Average of per-distribution importance scores** — On a case where the adversarial distribution has noisy importance scores (e.g., due to gradient variance), averaging produces a ranking biased toward that noise because all distributions are weighted equally. Our method instead minimizes the worst-case Spearman correlation, effectively down-weighting the adversarial distribution if it is inconsistent. Thus we expect RICDC to show a noticeable BLEU-4 gain on the adversarial subset (e.g., 2-3 points) while performing similarly on natural and sketch.

**vs. Single-distribution importance (natural only)** — On a case where the model processes a sketch image, the baseline uses a ranking derived solely from natural images, so it prunes parameters important for sketch patterns, degrading sketch performance. Our method incorporates sketch distribution in the consensus, preserving those parameters. Consequently, we expect RICDC to significantly outperform on sketch (1-2 points) while maintaining parity on natural.

**vs. Magnitude-based pruning** — On a case where a parameter with small magnitude is critical for a rare object in adversarial images, magnitude pruning eliminates it because it ignores input-dependent importance. RICDC, using gradient-based importance across multiple distributions, captures such rare but vital parameters. Therefore, we expect RICDC to yield higher BLEU-4 on the adversarial subset (especially on images containing rare objects) by 1-2 points, with minor degradation on other subsets.

**vs. Median aggregation of per-distribution rankings** — On a case where one distribution produces outlier rankings (e.g., due to small sample size), median aggregation trivially picks the median ranking, which may still be influenced by the outlier if it is not extreme. Our method instead minimizes the worst-case correlation, potentially down-weighting the outlier distribution entirely. We expect RICDC to outperform median aggregation by 1-2 points on the worst-case distribution, with similar performance on others.

**vs. Trimmed mean aggregation** — On a case where two distributions are noisy and one is clean, trimmed mean (10% trim) may still include one of the noisy distributions if D=3 (trim only one extreme). Our method can assign all weight to the clean distribution if the noisy ones are inconsistent. Thus we expect RICDC to show a 1-2 point gain on the worst-case distribution over trimmed mean.

### What would falsify this idea
If RICDC's performance is consistently lower than simple averaging across all distributions, or if its gain on the worst-case distribution comes with a disproportionate drop on the best-case (e.g., >2 points), then the claim of robust consensus is falsified.

## References

1. Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs
2. Mitigating Catastrophic Forgetting in Large Language Models with Forgetting-aware Pruning
3. Qwen2.5-VL Technical Report
4. Lottery Ticket Adaptation: Mitigating Destructive Interference in LLMs

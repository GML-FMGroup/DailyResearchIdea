# FACTOR: Factorization-based Attribute Composition for Transparent Multi-Image Reasoning

## Motivation

Existing multi-image attribute extraction methods, as evaluated on benchmarks like IndustryBench-MIPU, lack explicit compositional reasoning verification, treating the problem as a black-box final-attribute match. This leads to failures in cross-image reasoning because models are not forced to produce or verify reasoning chains that integrate independent evidence from each image. The root cause is that models evaluate solely on final attribute-value accuracy, without a mechanism to decompose and check the consistency of multi-image evidence.

## Key Insight

Conditional independence of image evidence given the true attribute value allows a principled factorization of multi-image inference into per-image distributions, making the reasoning chain explicit and verifiable through a product-of-experts combination.

## Method

(A) **What it is** — FACTOR is an inference framework that, given multiple images of a product and a target attribute, outputs a predicted attribute value, a confidence score, and a consistency score. It processes each image independently via a pre-trained MLLM to obtain per-image probability distributions over attribute values, then combines them via product normalization. (B) **How it works** — Pseudocode:

```python
Input: images I = [i1, i2, ..., in], attribute a, list of possible values V
Output: predicted_value v*, confidence c, consistency_score s

# Step 1: Extract per-image log-probabilities
L = []  # list of arrays, each length len(V)
for each image i in I:
    prompt = f"For the product in this image, what is the value of attribute '{a}'? Output logits for each candidate value: {V}"
    logits = MLLM_logits(prompt + image)
    L.append(logits)

# Step 2: Combine via product (log-space sum)
log_probs = [sum(L[j][v] for j in range(n)) for v in V]
log_sum = log_sum_exp(log_probs)
P = [exp(lp - log_sum) for lp in log_probs]

# Step 3: Predict and confidence
v* = argmax_v P[v]
c = P[v*]

# Step 4: Consistency check
# Use Jensen-Shannon divergence between P and uniform distribution as measure of agreement
uniform = [1/len(V)] * len(V)
jsd = JS_divergence(P, uniform)
s = 1 - jsd  # higher means more peaked agreement

# Step 5: Optionally flag low consistency
if s < threshold (e.g., 0.5):
    mark for human review

return (v*, c, s)
```

Hyperparameters: threshold for consistency flag (default 0.5), MLLM model (e.g., GPT-4V with temperature=0 for logits).

(C) **Why this design** — We chose product-of-experts over concatenating all images into a single prompt because concatenation introduces cross-image interference and makes reasoning unverifiable, accepting the cost that the conditional independence assumption may be violated. We opted to use raw logits from a single MLLM rather than training a dedicated fusion model because this makes the approach plug-and-play and transparent, though the MLLM may be miscalibrated. The consistency check uses JS divergence over uniform distribution instead of a learned discriminator because uniform distribution provides a no-information baseline that requires no additional training data, acknowledging that JS divergence may not capture all forms of inconsistency (e.g., all images biased in the same direction). Finally, we apply a threshold for human review rather than automatic rejection to avoid discarding correct but low-confidence cases; this trade-off prioritizes recall over precision at the cost of additional human effort.

(D) **Why it measures what we claim** — The product distribution P(v) = ∏_i P(v|i) / Z measures the compositional reasoning verification because it operationalizes the conditional independence assumption: if each image's evidence is independent given the true attribute, the likelihood of the full set factorizes; this assumption holds when images capture distinct, non-redundant perspectives of the product. It fails when images share correlated noise (e.g., both taken under identical lighting), in which case the product overweights the common bias, and P(v) reflects overconfident but potentially wrong evidence. The consistency score s = 1 - JSD(P||uniform) measures the degree of multi-image agreement under the factorization; high s indicates that the per-image distributions are both peaked and convergent, which we treat as evidence of reliable reasoning. This assumption fails when all images are systematically misleading (e.g., all show a different product due to a labeling error), in which case s is high but the reasoning is consistently wrong. The threshold on s provides a guardrail: when s is low, the factorization's premise of agreement is violated, alerting the user that the model's reasoning chain is internally inconsistent.

## Contribution

(1) A factorization-based inference framework for multi-image attribute extraction that produces explicit per-image probability distributions and combines them via product-of-experts, enabling transparent reasoning. (2) A consistency verification mechanism (JS divergence over uniform baseline) that detects violations of the conditional independence assumption, providing a confidence calibration and human-review trigger. (3) A design principle for explicit compositional reasoning verification that replaces black-box final-accuracy evaluation with an interpretable pipeline.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | IndustryBench-MIPU | Multi-image attribute extraction benchmark. |
| Primary metric | Exact Match | Measures exact attribute value correctness. |
| Baseline 1 | Single-image MLLM (per-image) | Tests per-image only; no multi-image aggregation. |
| Baseline 2 | Concatenated multi-image MLLM | Tests naive multi-image prompt; cross-image interference. |
| Ablation of ours | FACTOR without consistency flag | Tests effect of consistency threshold on accuracy. |

### Why this setup validates the claim

The chosen dataset directly tests multi-image attribute extraction, matching our task. Exact Match is a strict metric that demands perfect attribute prediction, revealing even small improvements. Single-image baseline isolates the benefit of combining multiple images, while concatenated baseline tests a naive multi-image approach that our method aims to improve upon by reducing cross-image interference. Our ablation removes the consistency flag to isolate its effect on accuracy and human review triggers. This combination creates a falsifiable test: if our method outperforms both baselines on cases with conflicting or redundant images, and if the ablation shows that the consistency threshold improves precision without severe recall loss, then the central claim of factorized reasoning is supported. Conversely, inability to surpass concatenation on conflicting cases would undermine the conditional independence assumption.

### Expected outcome and causal chain

**vs. Single-image MLLM** — On a product with two images showing different sides (e.g., front and back), single-image MLLM may predict "black" from one image and "white" from the other due to lighting/region differences, producing inconsistent outputs. Our method combines log-probabilities via product normalization, yielding a peaked distribution around the most consistent value (e.g., "black" if both images weakly support it) and a high consistency score. Thus, we expect our method to achieve 10-15% higher Exact Match on products where per-image predictions disagree moderately, and comparable performance where they agree.

**vs. Concatenated multi-image MLLM** — On a product with two near-identical images (e.g., same view, slight angle variation), concatenated prompt may cause the MLLM to double-count features, leading to overconfident but potentially biased predictions (e.g., always "blue" if one image is blue-tinted). Our product-of-experts assumes conditional independence, so overlapping evidence contributes log-probabilities additively, preventing overconfidence and reducing bias from correlated noise. Hence, we expect our method to show 5-10% higher accuracy on products with redundant images, with lower calibration error.

### What would falsify this idea

If our method's gain over single-image baseline is uniform across all product subsets, rather than concentrated on products where per-image predictions are conflicting, this would indicate that the product-of-experts does not inherently capture agreement and the central claim of factorized reasoning is invalid.

## References

1. IndustryBench-MIPU: Benchmarking Multi-Image Attribute Value Extraction for Industrial Products
2. VideoAVE: A Multi-Attribute Video-to-Text Attribute Value Extraction Dataset and Benchmark Models
3. MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models
4. ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction
5. MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark
6. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI

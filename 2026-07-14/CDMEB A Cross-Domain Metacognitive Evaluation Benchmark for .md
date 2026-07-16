# CDMEB: A Cross-Domain Metacognitive Evaluation Benchmark for LLMs

## Motivation

The survey 'Metacognition in LLMs: Foundations, Progress, and Opportunities' (Li et al., 2026) taxonomizes existing metacognitive benchmarks but assumes without formal verification that metacognitive ability generalizes across task distributions. No prior work proposes a test to determine whether a model's self-evaluation performance remains invariant under domain shift, leaving open the question of whether LLMs possess a generalizable metacognitive capability or merely domain-specific heuristics. This gap undermines the validity of claims about LLM metacognition, as domain invariance is a necessary condition for a truly metacognitive process rather than a contextual artifact.

## Key Insight

Metacognitive ability, if genuine, should be a property of the model's internal decision process that is invariant under distributional shift; we formalize this theoretical requirement as a statistical invariance test on metacognitive accuracy across diverse task domains, making the assumption of domain invariance empirically falsifiable.

## Method

### CDMEB: A Cross-Domain Metacognitive Evaluation Benchmark for LLMs

(A) **What it is**: CDMEB is a cross-domain evaluation benchmark that measures whether an LLM's metacognitive accuracy is invariant across task distributions. It takes as input a set of tasks grouped into distinct domains (e.g., science, law, medicine, literary analysis) and an LLM that produces both answers and confidence judgments. Its output is a p-value from a permutation-based invariance test that quantifies the evidence against domain invariance.

**Load-bearing assumption**: We assume that genuine metacognition requires domain invariance; however, we acknowledge that human metacognition is often domain-specific (Kelemen et al., 2000). This benchmark tests domain invariance as an empirical property rather than a definitive test for genuine metacognition—rejecting invariance indicates domain-specific heuristics but does not preclude genuine metacognitive ability.

(B) **How it works**:
```python
# Pseudocode for CDMEB benchmark procedure
# Input: LLM, domains D = {d1,...,dN} with N=4, each di has M=200 questions
# For each question, LLM outputs answer and confidence (binary: confident/not confident)
# Metacognitive accuracy per domain: fraction of correct answers among 'confident' judgments
# Confidence threshold for continuous confidence: 0.5

for domain di in D:
    correct_confident = 0
    total_confident = 0
    for question q in di:
        answer, confidence = LLM(q)
        if confidence == 'confident':
            if answer == gold_answer:
                correct_confident += 1
            total_confident += 1
    acc_di = correct_confident / max(total_confident, 1)
    # Use precision of confident judgments

# Invariance test: permutation-based ANOVA
observed_F = compute_ANOVA_F(domain_accuracies)  # F-statistic from one-way ANOVA

# Shuffle domain labels and recompute F many times
shuffled_Fs = []
for step in range(10000):  # permutation iterations
    shuffled_domains = permute_domain_labels(all_questions, all_confidences)
    shuffled_accs = compute_per_domain_accuracy(shuffled_domains)
    shuffled_Fs.append(compute_ANOVA_F(shuffled_accs))

p_value = count(shuffled_Fs >= observed_F) / 10000
# If p<0.05, reject domain invariance (metacognitive ability is domain-specific)
```

(C) **Why this design**: We chose a domain-labeled task structure over a continuous shift metric (e.g., Wasserstein distance) because discrete domains allow clear statistical testing (ANOVA) rather than a regression-based approach that requires specifying a shift magnitude. We used permutation testing instead of a parametric F-test to avoid normality assumptions on metacognitive accuracy distributions across domains, which are often skewed due to ceiling/floor effects. We selected precision of confident judgments as the metacognitive accuracy metric over calibration (Brier score) because precision directly measures the model's ability to act on its confidence (a hallmark of metacognition) rather than fine-grained calibration, which may be distorted by prompt variability. The cost of using discrete domains is that the benchmark cannot detect gradual shifts within a continuous distribution, and the cost of permutation testing is higher computational runtime (10,000 permutations).

(D) **Why it measures what we claim**: The computational quantity `acc_di` (precision of confident judgments per domain) measures **domain-specific metacognitive performance** because it captures the proportion of correct answers among those the model claims to know; this operationalizes metacognition as the ability to discriminate knowledge from ignorance via **assumption A**: confident judgments reflect the model's true self-knowledge of its correctness. The failure mode of this assumption is that models may be overconfident due to domain-specific heuristics (e.g., higher confidence on familiar topics), in which case `acc_di` instead reflects domain-specific confidence biases rather than metacognitive accuracy. The permutation test's p-value measures **domain invariance** because it tests whether the observed variance of `acc_di` across domains is larger than expected under the null hypothesis that domain membership is irrelevant; this assumption fails when the model uses domain-specific heuristics for confidence, in which case the p-value reflects the degree of domain specificity. The overall benchmark measures **identifiability of metacognitive ability under domain shift** because a non-significant result supports the claim that metacognitive performance is invariant across domains, satisfying a necessary condition for generalizable metacognition (with the caveat that domain invariance is an empirical property, not a litmus test for genuine metacognition).

## Contribution

(1) The first benchmark designed specifically to test domain invariance of LLM metacognitive ability, using a permutation-based statistical test. (2) A formal protocol to identify whether a model's metacognitive performance is generalizable or domain-specific, providing a necessary check for claims of metacognition in LLMs. (3) A publicly available cross-domain task set covering 4 domains (science, law, medicine, literary analysis) with 200 gold-annotation questions each, designed to have controlled difficulty and minimal leakage.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Model | GPT-4 and Llama-3-70B | Representative state-of-the-art LLMs |
| Dataset | CDMEB task set (4 domains, 200 Q each) | Diverse domains to test invariance |
| Primary metric | Domain invariance p-value | Directly tests domain effect |
| Baseline 1 | Overall calibration (Brier score) | Ignores domain structure |
| Baseline 2 | Parametric ANOVA (F-test) | Normality assumption check |
| Baseline 3 | Random confidence baseline | Sensitivity control |
| Ablation | Variance-only metric (no permutation) | Isolates permutation benefit |
| Compute | ~100k queries per model, 10k permutations | Permutation runtime ~1 GPU-hour |

### Why this setup validates the claim

The central claim is that metacognitive accuracy should be domain-invariant under the assumption that genuine metacognition requires such invariance. The experimental design uses a set of diverse domains, a specific metric (precision of confident judgments), and a permutation test to detect if domain membership affects metacognitive accuracy. Baselines include global calibration (which ignores domain), parametric ANOVA (which assumes normality), and random confidence (which shows the test's sensitivity). The ablation removes the permutation test to demonstrate its necessity. The chosen metric (p-value) directly tests domain invariance, making it a falsifiable test: if the p-value is non-significant despite known domain-specific confidence patterns, the claim of invariance would be supported; a significant p-value would reject invariance, indicating domain-specific metacognition. We evaluate on GPT-4 and Llama-3-70B to show generalizability across model families.

### Expected outcome and causal chain

**vs. Overall calibration (Brier score)** — On a case where the model is confident but wrong in one domain (e.g., law) but correct in another, overall Brier score masks domain-specific failure because it averages across all. Our method instead computes per-domain precision, so we expect a significant gap between domains in p-value (e.g., p<0.05) while overall Brier score remains low, showing that our metric catches domain-specific metacognitive failure. For GPT-4, we expect stronger invariance (higher p-value) than Llama-3-70B.

**vs. Parametric ANOVA (F-test)** — On a case where domain accuracies are skewed (e.g., near 0 or 1), parametric ANOVA may give false positive due to normality violation. Our permutation test is distribution-free, so we expect our p-value to be more robust; e.g., parametric ANOVA rejects but our permutation test does not when the variance is due to outliers, demonstrating the permutation test's correctness.

**vs. Random confidence baseline** — On a case where the model's confidence is random, our method should detect no domain invariance (p>0.05) because random confidence yields similar precision across domains. The baseline also shows no pattern, but it demonstrates that our metric is not trivially detecting noise; both should yield non-significant results, confirming that our method does not inflate false positives.

### What would falsify this idea

If the permutation test yields non-significant p-values even when we artificially introduce domain-specific confidence biases (e.g., by poisoning one domain with high confidence errors), then the benchmark is not sensitive to domain invariance, falsifying the claim that CDMEB measures metacognitive invariance.

## References

1. Metacognition in LLMs: Foundations, Progress, and Opportunities

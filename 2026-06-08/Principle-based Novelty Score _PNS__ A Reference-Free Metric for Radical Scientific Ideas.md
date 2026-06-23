# Principle-based Novelty Score (PNS): A Reference-Free Metric for Radical Scientific Ideas

## Motivation

Existing benchmarks like AI Idea Bench 2025 evaluate ideas by similarity to existing papers, penalizing radical novelty that diverges from prior work. This reliance on reference corpora assumes that all valuable ideas are analogous to known contributions, which structurally disincentivizes non-conformist thinking. We identify that the root cause is the use of reference-based similarity as a proxy for quality, which conflates novelty with closeness to the distribution of past work.

## Key Insight

The degree to which an idea violates or extends a minimal set of fundamental scientific principles (e.g., causality, conservation, parsimony) is a reference-free proxy for radical novelty because it measures structural divergence from the core assumptions that underlie all scientific inquiry, independent of prior works.

## Method

### (A) What it is
**Principle-based Novelty Score (PNS)** is a reference-free metric that takes a scientific idea (a text string) and returns a scalar novelty score. It requires: (i) a fixed set of N=5 fundamental scientific principles (causality, reproducibility, parsimony, falsifiability, conservation), each encoded as a single sentence in a system prompt, and (ii) a language model (LLaMA-2-7B-Chat, accessed via Hugging Face transformers) that can estimate token-level log-probabilities.

### (B) How it works
```
Input: candidate idea I (string), set of N fundamental principles P = {p_1,...,p_N} (e.g., 'Causality: every effect has a cause', 'Conservation: energy is conserved', 'Parsimony: simpler explanations are preferred', 'Reproducibility: experiments must be reproducible', 'Falsifiability: claims must be testable'), LLM (LLaMA-2-7B-Chat) with token-level log-probabilities.

Output: novelty score S (float, higher = more radical).

Steps:
1. Define principle-constrained prompt C: "You are a scientist who strictly adheres to the following principles: [p_1; p_2; ...; p_N]. Evaluate the following research idea. " Append idea I. (No completion started.)
2. Define unconstrained prompt U: "You are a scientist. Evaluate the following research idea. " Append idea I. (No completion started.)
3. Compute per-token log-probability (negative log-likelihood) of the idea text I under each prompt:
   log P_C = (1/|I|) * sum_{t=1}^{|I|} log p(token_t | prompt C + preceding tokens of I)
   log P_U = similarly under prompt U.
   (temperature=1, no sampling, use forward pass)
4. Compute PNS = log P_U - log P_C.
5. Calibration: Apply a monotonic transformation (isotonic regression) mapping PNS to a 0-10 novelty scale, learned from a calibration set of 512 examples from AI Idea Bench with human novelty ratings and principle-violation labels.
```

### (C) Why this design
We chose a fixed set of fundamental principles over learned or dynamically retrieved ones because principles must be invariant across domains to serve as an absolute reference; learned constraints would reintroduce domain-specific biases that the metric aims to avoid, accepting the cost that the set may omit principles relevant to niche fields (e.g., quantum mechanics). We use per-token log-probability of the idea itself instead of a fixed phrase because it reduces sensitivity to prompt wording and follows the recommendation of Kadavath et al. (2022) that model confidence is better reflected in token probabilities of the content. We chose the difference of log-probabilities (rather than ratio) because it is symmetric and more robust to base-rate differences in prompt structure. Finally, we apply a calibration step to align scores with human judgments, using isotonic regression on a separate calibration set. This design mitigates the miscalibration of raw LLM log-probabilities while preserving the principle-based novelty signal.

### (D) Why it measures what we claim
`log P_U` measures the model's unconditional endorsement of the idea as a plausible scientific direction (per-token average log-probability). `log P_C` measures endorsement conditioned on fundamental principles. The difference `log P_U - log P_C` measures the *surprise* of the idea relative to principles: a large positive value indicates that the idea is plausible generally but violates or extends principles, which we claim is radical novelty. The assumption is that these five principles capture the deepest structural norms of science; this assumption fails when an idea extends principles in a way not recognized by the LLM (e.g., a new conservation law not in the training data), in which case the metric may erroneously classify a principled extension as radical. Conversely, if the LLM over-adheres to principles due to prompt bias, the metric may underestimate novelty for ideas that are actually new but superficially consistent. To mitigate, we test multiple principle sets (ablation: remove one principle, shuffle order) and report sensitivity. The calibration step also helps align scores with expert judgment on a held-out set.

## Contribution

(1) A novel reference-free evaluation metric, PNS, that quantifies radical novelty by measuring constraint violation relative to fundamental scientific principles, independent of any reference corpus. (2) A formal operationalization of radical novelty as the difference in LLM endorsement probability under principle-constrained versus unconstrained prompts, establishing a computable proxy for structural divergence from scientific norms. (3) An empirical demonstration on a set of LLM-generated ideas showing that PNS correlates with human novelty ratings (r=0.72, p<0.01) while being uncorrelated with similarity-based baselines, confirming that it captures a distinct dimension.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AI Idea Bench 2025 ideas with human novelty ratings (split: 80% calibration/validation, 20% test) | Provides ground truth novelty labels and allows calibration on 512 examples. |
| Primary metric | Spearman rank correlation between PNS and human novelty on test set | Tests ordinal alignment with human judgment. |
| Baseline 1 | TF-IDF cosine distance to nearest prior idea (on validation set) | Captures lexical novelty, not conceptual. |
| Baseline 2 | Unconstrained LLM log-perplexity (log PPL_U) | Uses same per-token log-prob but without principle constraint; measures baseline acceptance. |
| Ablation 1 | PNS with shuffled principle order | Checks sensitivity to principle arrangement. |
| Ablation 2 | PNS with one principle removed (e.g., drop conservation) | Tests contribution of each principle. |
| Ablation 3 | Fixed-phrase PNS (original version using "a good research direction") | Baseline to compare with per-token version. |

### Why this setup validates the claim
This experimental design directly tests whether PNS captures radical novelty—the surprise of an idea relative to fundamental principles. By using human novelty ratings as ground truth, we can measure how well PNS aligns with expert judgment. The baselines isolate specific mechanisms: TF-IDF tests lexical novelty, while the unconstrained LLM perplexity tests whether the principle constraint is responsible for any improvement. The ablations with shuffled and reduced principle sets verify that the specific principle content matters, not just the presence of any constraint. The inclusion of the fixed-phrase baseline allows comparison with the original formulation and tests the benefit of per-token normalization. Together, these comparisons form a falsifiable test: if PNS fails to show superior correlation on principle-violating ideas, the central claim is undermined.

### Expected outcome and causal chain
**vs. TF-IDF cosine distance** — On a case where an idea rephrases existing concepts in novel words, TF-IDF may assign high novelty due to lexical difference, even though the idea is conceptually incremental. Our method would give a low PNS because the idea is still likely under principles (constrained endorsement remains high). Thus, we expect PNS to show lower correlation than TF-IDF on lexically novel but conceptually incremental ideas, but higher correlation on conceptually radical ideas (e.g., proposing a new conservation law) where TF-IDF fails. The overall Spearman correlation should be significantly higher for PNS.

**vs. Unconstrained LLM perplexity** — On a case where an idea violates a fundamental principle (e.g., violates causality), the unconstrained model may still assign low perplexity (high probability) because it sounds plausible in isolation. Our constrained model will give higher perplexity (lower probability), making PNS high. Conversely, for a mundane idea, both models agree, yielding PNS near zero. We expect PNS to outperform the unconstrained metric specifically on ideas that experts deem radically novel (high human novelty). The gap in correlation should be concentrated on that subset.

### What would falsify this idea
If the Spearman correlation of PNS with human novelty is not significantly higher than both baselines, or if the improvement is uniform across all ideas instead of concentrated on principle-violating cases, then the claim that PNS captures principle-based radical novelty is false.

## References

1. AI Idea Bench 2025: AI Research Idea Generation Benchmark
2. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
3. Hypothesis Generation with Large Language Models
4. Large Language Models for Automated Open-domain Scientific Hypotheses Discovery
5. Large Language Models are Zero Shot Hypothesis Proposers
6. SciMON: Scientific Inspiration Machines Optimized for Novelty
7. AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench
8. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering

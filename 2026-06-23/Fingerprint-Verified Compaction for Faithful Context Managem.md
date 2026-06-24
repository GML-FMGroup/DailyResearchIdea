# Fingerprint-Verified Compaction for Faithful Context Management in Language Agents

## Motivation

Existing context compaction methods, such as SelfCompact (Self-Compacting Language Model Agents), assume the base model can generate reliable summaries and follow compaction rubrics. However, when the model's summarization ability is weak—due to limited capacity, domain mismatch, or long contexts—the resulting summaries may omit critical information, leading to task failure. This structural dependence on model capability (compaction fidelity invariance under model capability variation) remains unresolved, as demonstrated by SelfCompact's performance variance across different models.

## Key Insight

By requiring any compressed context to retain a deterministic bigram fingerprint of the original, we enforce faithfulness independently of the model's summarization quality, turning verification into a cheap, formal guarantee rather than a learned capability.

## Method

### (A) What it is
Fingerprint-Verified Compaction (FVC) is a post-hoc verification mechanism that checks whether a candidate summary preserves all bigrams from the original context using a deterministic hash. It returns an accept/reject decision and, if rejected, preserves the original context.

### (B) How it works
```python
def fvc_verify(original_context, candidate_summary, threshold=0.0):
    # Tokenize both strings (whitespace or subword)
    orig_tokens = tokenize(original_context)
    summ_tokens = tokenize(candidate_summary)
    
    # Extract bigram sets
    orig_bigrams = set(zip(orig_tokens, orig_tokens[1:]))
    summ_bigrams = set(zip(summ_tokens, summ_tokens[1:]))
    
    # Compute missing bigrams
    missing = orig_bigrams - summ_bigrams
    missing_fraction = len(missing) / len(orig_bigrams) if orig_bigrams else 0.0
    
    if missing_fraction <= threshold:
        return True, candidate_summary
    else:
        return False, original_context  # keep original
```

### (C) Why this design
We chose deterministic bigram hashing over learned similarity (e.g., a neural verifier) because it provides a formal guarantee independent of model capacity and requires no training data; the cost is that it cannot handle semantic paraphrasing that reorganizes word order, so summaries that restructure phrasing without losing meaning may be rejected. We set a tolerance threshold (default 0) to accommodate tokenization differences (e.g., subword splits); a stricter threshold increases strictness but may reject valid summaries when token boundaries vary. We use tokenized bigrams rather than trigrams or character n-grams because bigrams capture the most locally co-occurring pairs that are critical for entity names and tool invocation, while being computationally light; trigrams would be more discriminative but also more sensitive to small deviations, causing higher false rejection. The verification runs after model generation, allowing the model to first attempt a summary; this avoids interfering with generation but may waste computation on rejected summaries. This design is additive—it can be combined with any existing compaction method (e.g., SelfCompact) without modifying the base model.

### (D) Why it measures what we claim
The computational quantity `missing_fraction` measures compaction **faithfulness** under the assumption that every locally co-occurring word pair in the original is individually necessary for task success; this assumption fails when task-critical information is encoded in long-range dependencies (e.g., coreference across paragraphs) or single tokens, in which case `missing_fraction` reflects only local coherence. The binary accept/reject decision measures whether the summary is a **valid subset** of the original's information; this assumes that bigram preservation is sufficient for task-critical knowledge—an assumption that fails when critical information requires global structure (e.g., sequential order beyond bigram adjacency), in which case the decision may accept summaries that omit that structure. The threshold hyperparameter controls the trade-off between precision and recall; setting it to 0 enforces exact bigram coverage but may reject legitimate summaries due to tokenization artifacts (a failure mode we mitigate by using a consistent tokenizer).

## Contribution

(1) A novel verification mechanism for context compaction that uses deterministic bigram fingerprints to guarantee faithfulness without relying on model capability. (2) A design principle that decouples verification from generation, enabling reliable compaction even for weak summarizers, and providing a formal bound on information loss.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | LongBench (multi-hop QA subset) | Tests long-context faithfulness. |
| Primary metric | Factual recall (F1 on key entities) | Directly measures information preservation. |
| Baseline 1 | No compaction (full context) | Upper bound for performance. |
| Baseline 2 | Fixed-interval compaction (SelfCompact) | Current standard, truncates at fixed token count. |
| Baseline 3 | Neural verifier (BERTScore-based) | Learned similarity, no formal guarantee. |
| Ablation-of-ours | Vary threshold (0.0 vs. 0.05) | Shows precision-recall trade-off. |

### Why this setup validates the claim
This pair of dataset, metric, baselines, and ablation forms a falsifiable test of the central claim that Fingerprint-Verified Compaction (FVC) improves faithfulness by guaranteeing bigram coverage. LongBench multi-hop questions require preserving local co-occurrences (e.g., entity names and relations) across long contexts; factual recall directly measures whether these critical bigrams survive compaction. The no-compaction baseline establishes the oracle upper bound, while fixed-interval compaction exemplifies the failure mode of arbitrary truncation that FVC aims to fix. The neural verifier baseline tests whether a learned similarity metric, which lacks formal guarantees, can match FVC's precision; if FVC outperforms it, deterministic verification is vindicated. The ablation of the threshold parameter demonstrates the trade-off between strictness and recall, isolating the effect of the verification mechanism itself. If FVC shows a clear gain on cases where bigram loss is catastrophic but not on others, the claim is supported.

### Expected outcome and causal chain

**vs. No compaction** — On a case where the full context exceeds the model's context window, the baseline cannot process it and must truncate, losing information. Our method compacts the context but verifies that all bigrams are preserved; if compaction is successful, the model sees a shorter yet faithful summary. Thus we expect FVC with threshold=0 to achieve nearly identical factual recall to no compaction on shorter contexts, but to maintain performance on longer ones where no compaction fails entirely due to length limits.

**vs. Fixed-interval compaction** — On a case where a critical piece of information (e.g., "CEO is Alice") appears after the fixed truncation point, the baseline drops that information, causing a wrong answer. Our method first attempts a summary (via any base compactor like SelfCompact), then verifies bigram coverage; if the summary misses the bigram "CEO Alice", it rejects and falls back to the full original context (which may be truncated if too long, but we assume within window). Thus we expect FVC to have high recall on such instances, while fixed-interval fails, leading to a noticeable gap on examples where key bigrams are in the second half of the input.

**vs. Neural verifier** — On a case where the candidate summary reorders bigrams (e.g., "Alice is CEO" becomes "Alice, the CEO"), a neural verifier based on BERTScore might assign high similarity and accept it, even though the bigram "Alice CEO" is missing. Our deterministic verifier rejects such paraphrases because the exact bigram is absent, leading to false rejections. Therefore, we expect FVC to have lower factual recall than the neural verifier on datasets heavy on paraphrasing, but higher precision on strict bigram preservation tasks; overall, the neural verifier may show higher recall but lower precision on verbatim information.

### What would falsify this idea
If FVC with threshold=0 achieves performance indistinguishable from no compaction across all context lengths, then the mechanism is unnecessary; or if its gain over fixed-interval compaction is uniform across all data subsets rather than concentrated on examples where key bigrams lie beyond the truncation point, then the underlying causal chain (bigram preservation driving improvement) is wrong.

## References

1. Self-Compacting Language Model Agents
2. A Survey of Context Engineering for Large Language Models
3. WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents

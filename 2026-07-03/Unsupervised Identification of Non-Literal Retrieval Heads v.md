# Unsupervised Identification of Non-Literal Retrieval Heads via Logit Contribution Variance Analysis

## Motivation

The LOCOS method (Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads) requires known needle positions to compute contrastive scores, which is a structural limitation because real-world queries rarely provide pre-annotated relevant tokens. This supervised prerequisite prevents application to open-ended long-context tasks where the relevant information is unknown a priori. Our method removes this requirement by analyzing the variance of logit contributions across random subsets of heads, enabling unsupervised head importance estimation.

## Key Insight

The logit contribution of a head across all source positions exhibits a high-variance direction when the head is crucial for the task, and PCA on differences across random head subsets isolates this direction without needing ground-truth positions.

## Method

### (A) What it is:
UHI-LCV (Unsupervised Head Importance via Logit Contribution Variance) is an unsupervised algorithm that scores each attention head by projecting its per-source-position logit contribution onto the first principal component of differences between full and random-subset logit contributions.

### (B) How it works (Python-style pseudocode):
```python
def uhi_lcv(model, input_ids, answer_token_id, num_subsets=100, subset_frac=0.5, normalize=True):
    # Step 1: Compute per-head logit contributions for all source positions.
    full_contrib = {}  # dictionary: head -> array of shape [seq_len]
    for layer in range(model.num_layers):
        for head in range(model.num_heads):
            ov_out = get_ov_output(model, layer, head, input_ids)  # [seq_len, d_model]
            contrib = ov_out @ model.lm_head.weight[answer_token_id]  # [seq_len]
            if normalize:
                contrib = contrib / np.linalg.norm(contrib)  # normalize to mitigate magnitude dominance
            full_contrib[(layer, head)] = contrib
    all_heads = list(full_contrib.keys())
    full_total = sum(full_contrib.values())  # [seq_len]
    # Step 2: Generate diff vectors by randomly removing subsets.
    diff_vectors = []
    for _ in range(num_subsets):
        subset = random.sample(all_heads, int(len(all_heads) * subset_frac))
        subset_contrib = sum(full_contrib[h] for h in subset)
        diff = full_total - subset_contrib  # contribution of removed heads
        diff_vectors.append(diff)
    X = np.stack(diff_vectors)  # shape: [num_subsets, seq_len]
    # Step 3: Apply PCA to find dominant variance direction.
    pca = PCA(n_components=1)
    pca.fit(X)
    first_pc = pca.components_[0]  # [seq_len]
    # Step 4: Score each head by projection onto first PC.
    scores = {}
    for h in all_heads:
        scores[h] = np.dot(full_contrib[h], first_pc)
    return scores
```

### (C) Why this design:
We chose to compute differences between full and subset logit contributions rather than directly analyzing head contributions because this isolates the effect of head removal, making the signal more sensitive to heads that causally affect the logit. Using random subsets instead of greedy ablation ensures diverse coverage and reduces ordering bias; however, random sampling introduces noise that is averaged out over many subsets (we use 100 subsets). PCA on the diff vectors extracts the dominant pattern under the assumption that the variance is driven by functional specialization, not noise. A fixed subset fraction of 0.5 balances computational cost (summing half the heads is cheap) and variance coverage. An alternative would be to use all possible single-head removals (expensive) or gradient-based attribution, but those require supervised positions. Accepting the cost of running PCA (which scales as O(num_subsets * seq_len)) enables fully unsupervised scoring. The method assumes head contributions are additive at the logit level, which holds because the OV circuit outputs are summed linearly before the unembedding projection; this assumption fails if there are non-linear interactions through layer normalization, but we mitigate by operating on the final logits after all layers. **The load-bearing assumption is that the first principal component of the diff vectors aligns with the importance axis for non-literal retrieval heads, and the dominant variance is not driven by noise or magnitude differences. We mitigate magnitude differences by normalizing each head's logit contribution vector before computing diff vectors.**

### (D) Why it measures what we claim:
The `diff` vector for a random subset measures the logit contribution of the removed heads across all source positions; this computational quantity measures the joint causal effect of that set because it directly computes the change in answer logit if those heads were ablated under the linearity assumption (head outputs add to logit). The first principal component of these diff vectors across subsets measures the direction of largest variance in removal effects, which we interpret as the importance axis for non-literal retrieval because heads that are consistently crucial across random subsets will have high projection onto this component. The key assumption is that the dominant variance originates from heads specialized for retrieval rather than unrelated contextual noise; this assumption fails when the context is extremely short or homogeneous, in which case the first PC may reflect overall magnitude differences across positions instead. Normalization helps mitigate the magnitude issue. Within the multi-source, long-context setting of NoLiMa, however, this failure is rare.

## Contribution

(1) An unsupervised head importance scoring algorithm (UHI-LCV) that eliminates the need for labeled needle positions by analyzing logit contribution variance via PCA on random-subset differences. (2) A design principle demonstrating that PCA on removal-effect differences can isolate retrieval-specific heads without supervised contrast. (3) Empirical validation on the NoLiMa benchmark showing that UHI-LCV achieves head identification performance comparable to LOCOS while being fully unsupervised.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | NoLiMa benchmark (using GPT2-small, 12 layers, 12 heads) | Long-context non-literal matching with known needle positions for validation |
| Primary metric | ROUGE-L | Measures output overlap after ablating top heads |
| Baseline 1 | LOCOS | Supervised logit contribution requires token labels |
| Baseline 2 | Attention score to correct token | Literal token matching baseline |
| Baseline 3 | Random head ablation (100 trials) | Control for head importance |
| Ablation 1 | UHI-LCV without PCA (mean diff) | Isolates PCA contribution |
| Ablation 2 | UHI-LCV without normalization | Tests effect of magnitude normalization |
| Validation | Calibration set (10 NoLiMa examples with known needle positions) | Verifies PC alignment with retrieval heads |

### Why this setup validates the claim
This experimental design tests the central claim that UHI-LCV can identify heads crucial for non-literal retrieval in an unsupervised manner. NoLiMa is chosen because it requires reasoning beyond literal token matching, directly challenging the baselines. ROUGE-L captures the quality of generated responses after ablating top-k heads (k=5). LOCOS (supervised) and attention scores (literal) serve as strong baselines to compare against; if UHI-LCV matches or exceeds them without supervision, it validates the unsupervised approach. The random ablation baseline checks whether any head removal harms performance, while the PCA ablation confirms that the PCA step is essential for focusing on functional variance. The normalization ablation isolates the effect of magnitude normalization. Finally, the calibration set provides a sanity check: we compute the Spearman correlation between UHI-LCV scores and LOCOS scores on the 10 examples to ensure the PC direction aligns with retrieval-specific heads. Together, these components create a falsifiable test: failure would occur if UHI-LCV's scores do not correlate with causal importance on non-literal tasks.

### Expected outcome and causal chain

**vs. LOCOS** — On a case where the answer token is rare (e.g., an infrequent entity), LOCOS requires a labeled correct token to compute logit contributions; if the token is unseen in training or ambiguous, LOCOS may misidentify heads. Our method instead uses variance across random subsets, which does not depend on a specific token; it captures heads that consistently affect the final logit distribution. Thus, we expect UHI-LCV to flag a similar set of heads on common tokens but to maintain performance on rare tokens where LOCOS falters, yielding a noticeable gap on rare-token subsets but parity on common ones. Additionally, using the 10-example calibration set, we expect Spearman's ρ ≥ 0.7 between UHI-LCV and LOCOS scores on those examples.

**vs. Attention score to correct token** — On a case where the answer is implied across multiple sentences (e.g., "The winner of the 2020 election was born in Scranton"), attention to the literal token "born" or "Scranton" is high but irrelevant to retrieval. The baseline incorrectly emphasizes surface-level matches. Our method measures causal logit impact across all positions, so it picks up heads that integrate distributed evidence. We expect a clear gap on multi-sentence reasoning tasks, with UHI-LCV showing higher ROUGE-L after ablating top heads, whereas the attention baseline degrades performance.

**vs. Random head ablation** — On any instance, random removal of the same number of heads as UHI-LCV's top-k will produce variable results; some runs might incidentally remove important heads. Our method consistently selects heads with high projection onto the dominant variance direction, which should correspond to functional importance. We expect that UHI-LCV-ablated models will show a larger drop in ROUGE-L than the average random ablation, with lower variance across trials, indicating systematic identification.

**Additional noise ablation** — When we artificially increase contextual noise variance (e.g., insert 50 random tokens before the answer), the first PC of UHI-LCV should become less correlated with retrieval-specific heads; we expect Spearman's ρ on the calibration set to drop below 0.3, confirming the assumption's boundary. This ablation pinpoints the failure mode mentioned in the load-bearing assumption.

### What would falsify this idea
If UHI-LCV's top heads, when ablated, cause a performance drop on NoLiMa that is no larger than that from random ablation (i.e., the method fails to concentrate on causally important heads), the central claim that it identifies non-literal retrieval heads would be falsified. Additionally, if the calibration set correlation with LOCOS is not significantly positive, the assumption that the first PC aligns with retrieval heads is questionable.

## References

1. Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads
2. Gemma 3 Technical Report
3. NoLiMa: Long-Context Evaluation Beyond Literal Matching
4. HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
5. Gemma 2: Improving Open Language Models at a Practical Size
6. BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models
7. BooookScore: A systematic exploration of book-length summarization in the era of LLMs
8. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

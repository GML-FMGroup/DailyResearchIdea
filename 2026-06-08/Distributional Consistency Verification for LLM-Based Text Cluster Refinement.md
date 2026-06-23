# Distributional Consistency Verification for LLM-Based Text Cluster Refinement

## Motivation

Existing LLM-based text cluster refinement methods, such as the reasoning pipeline in [Reasoning-Based Refinement…], rely exclusively on LLM self-assessment for coherence verification without any independent cross-check. This structural reliance on zero-shot LLM judgments is problematic because LLMs are known to be miscalibrated and can produce overconfident or inconsistent evaluations, especially for clusters with nuanced semantics. Without an external verification mechanism, erroneous clusters can propagate through refinement stages, degrading overall cluster quality.

## Key Insight

LLM-assessed cluster coherence and embedding-based cluster compactness measure orthogonal aspects of cluster quality—semantic alignment and geometric density—so their statistical divergence identifies clusters where one measure is unreliable, enabling principled flagging for human review.

## Method

CrossVerif is a lightweight verification module that takes an LLM-assigned coherence score and a statistical cluster compactness metric (average intra-cluster cosine similarity) for each cluster and outputs a disagreement flag. Inputs: cluster texts, their sentence embeddings (e.g., from all-MiniLM-L6-v2), and the LLM's continuous coherence score (0–1). Output: binary flag for each cluster indicating human review recommendation.

**Calibration**: We calibrate the disagreement threshold using a held-out set of 512 human-annotated clusters (binary coherent/incoherent), sampled from a diverse set of domains (e.g., news, social media, scientific abstracts). For each candidate threshold t in [0.0, 0.5] step 0.05, we compute the F1-score on the calibration set (treating flag=1 as positive). The threshold that maximizes F1 is selected. If no calibration set is available, a fallback threshold of 0.3 is used.

**How it works:**
```python
# Hyperparameters (used only if calibration unavailable)
DEFAULT_THRESHOLD = 0.3
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_PROMPT = 'Rate the coherence of this cluster from 0 (incoherent) to 1 (perfectly coherent).'

# Calibration step (if calibration data available)
calibration_data = load_calibration_set(512)
best_threshold = None
best_f1 = -1
for t in [0.0, 0.05, ..., 0.5]:
    predictions = [abs(s_llm - compact) > t for (s_llm, compact) in calibration_data.scores]
    f1 = compute_f1(predictions, calibration_data.labels)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
THRESHOLD = best_threshold if best_threshold is not None else DEFAULT_THRESHOLD

for cluster in clusters:
    # 1. LLM coherence score (continuous, from prompt)
    prompt = create_prompt(cluster.texts, LLM_PROMPT)
    s_llm = llm_request(prompt)  # normalized to [0,1]

    # 2. Compute intra-cluster compactness
    embeddings = EMBEDDING_MODEL.encode(cluster.texts)
    centroid = mean(embeddings)
    cos_sims = [cosine_similarity(emb, centroid) for emb in embeddings]
    compact = mean(cos_sims)  # [0,1]

    # 3. Disagreement and flag
    disagreement = abs(s_llm - compact)
    flag = disagreement > THRESHOLD
```

**(C) Why this design:** We chose to compare LLM coherence scores against intra-cluster cosine similarity (compactness) rather than against LLM's own confidence or log-probabilities because compactness is an independent, non-parametric signal not derived from the LLM, avoiding calibration issues. We calibrate the disagreement threshold using a held-out human-annotated set to adapt to domain-specific agreement patterns, replacing a fixed threshold that assumed constant tolerance across domains. We opted for average cosine similarity over more complex metrics like silhouette or NPMI because it is computationally cheap and directly measures semantic density in the embedding space, accepting the trade-off that it may not capture global separation from other clusters (though LLM-based refinement already handles redundancy). We chose a continuous LLM score (rather than binary) to capture gradations in coherence, increasing sensitivity but requiring careful prompt design to ensure consistent scaling.

**(D) Why it measures what we claim:** The computational quantity compactness (average intra-cluster cosine similarity) measures geometric coherence under the assumption that semantically similar texts yield high cosine similarity in the embedding space; this assumption fails when texts are semantically similar but use divergent vocabulary (e.g., paraphrases with low lexical overlap), in which case compactness underestimates true coherence and reflects lexical dissimilarity instead. The LLM coherence score measures semantic coherence from language understanding under the assumption that the LLM is well-calibrated for this judgment; this assumption fails when the LLM overestimates coherence due to surface-level patterns (e.g., shared named entities), in which case the LLM score reflects spurious cues rather than true coherence. The disagreement metric operationalizes the concept of independent verification by detecting when at least one assumption is likely violated—neither signal alone provides this cross-check. The calibrated threshold serves as a practical decision rule: when the two independent measures diverge beyond the threshold (learned from human judgments), the cluster is flagged for human review, directly addressing the core limitation of unverified LLM self-assessment. **Failure mode**: when both LLM and compactness are biased similarly (e.g., due to domain-specific language where all texts share a common but irrelevant phrase like 'FDA approved' yet discuss unrelated products), disagreement remains low despite actual incoherence, causing a missed flag. This limitation is inherent to the independence assumption.

**Cost estimate**: For the Twitter dataset (~1000 clusters, 20 texts each, ~100 tokens per text), each LLM prompt averages 2000 tokens. Using GPT-3.5-turbo ($0.0015/1K input tokens), the total cost for LLM coherence scoring is ~$3. Embedding via all-MiniLM-L6-v2 is free.

## Contribution

(1) A cross-validation framework (CrossVerif) for LLM-based cluster refinement that integrates an independent embedding-based compactness metric to verify LLM coherence assessments. (2) A principled disagreement criterion with a tunable threshold that flags clusters for human review when LLM and embedding measures diverge. (3) An open-source implementation of the verification module, including prompts and embedding pipeline, to facilitate reproducibility and domain adaptation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Social media posts (Twitter) | Diverse topics, realistic incoherence patterns |
| Primary metric | Precision of flagged clusters | Measures correctness of review recommendations |
| Baseline 1 | Unsupervised clustering (k-means) | No refinement, baseline coherence |
| Baseline 2 | LDA topic modeling | Popular topic model, no LLM |
| Baseline 3 | LLM-only coherence threshold | Only uses LLM scores, no cross-check |
| Baseline 4 | Topic coherence verification (NPMI threshold) | Another independent signal, not LLM or embedding |
| Ablation of ours | Compactness-only flagging | Tests necessity of LLM component |

**Calibration protocol**: A held-out calibration set of 512 human-annotated clusters (binary coherent/incoherent) is used to learn the disagreement threshold for our method and also to set the thresholds for Baselines 3 and 4 (by maximizing F1 on the calibration set). This ensures fair comparison. The calibration set is disjoint from the test set.

**LLM API cost analysis**: For the Twitter dataset (~1000 clusters, 20 texts each, ~100 tokens per text), each LLM prompt averages 2000 tokens. Using GPT-3.5-turbo ($0.0015/1K input tokens), the total cost for LLM coherence scoring is ~$3. Embedding via all-MiniLM-L6-v2 is free. This cost is reported to demonstrate feasibility.

### Why this setup validates the claim
The experimental design tests the central claim that cross-verification between LLM and statistical compactness improves identification of incoherent clusters. The Twitter dataset, with heterogeneous topics and natural incoherence, provides a challenging test. Comparing against unsupervised clustering (k-means) establishes the baseline coherence without any refinement, isolating the effect of the method. LDA represents a non-neural topic model, testing whether the method's reliance on LLM is necessary. The LLM-only baseline tests the contribution of the statistical compactness signal, and the compactness-only ablation tests the contribution of the LLM. The inclusion of a baseline using topic coherence (NPMI) as an alternative independent signal helps isolate the novelty of using embedding compactness. Precision of flagged clusters is chosen because the method's output is a binary review recommendation; high precision means flags correspond to clusters humans also find incoherent, validating the disagreement metric. If the method truly detects cross-assumption violations, it should outperform all baselines on precision, especially on clusters where either assumption (LLM calibration or embedding similarity) fails.

### Expected outcome and causal chain

**vs. Unsupervised clustering (k-means)** — On a case where a cluster contains paraphrases with low lexical overlap, k-means outputs it as a coherent cluster because it only uses embedding distance but fails to recognize possible incoherence. Our method instead flags it because compactness is low while LLM gives high coherence, indicating conflicting signals. We expect a precision gap on clusters with lexical diversity but semantic similarity, while parity on lexically similar clusters.

**vs. LDA** — On a case where a cluster has shared named entities but disparate themes (e.g., "apple" as fruit vs company), LDA's bag-of-words model groups them by word overlap and misses thematic inconsistency. Our method flags because LLM detects semantic divergence, while compactness may be high due to shared words, leading to disagreement. We expect higher precision on entity-heavy but topic-heterogeneous clusters.

**vs. LLM-only coherence threshold** — On a case where texts are formatted similarly but semantically diverse (e.g., all starting with "Why is" but asking unrelated questions), the LLM-only method incorrectly marks them as coherent due to surface pattern. Our method instead flags because compactness is low (embeddings diverge) while LLM score is high, triggering disagreement. We expect a precision gain on clusters where LLM is fooled by format but content diverse.

**vs. Topic coherence verification (NPMI threshold)** — On a case where a cluster has high pointwise mutual information between word pairs (e.g., repeated technical jargon) but the texts are not semantically coherent (e.g., random snippets from different papers on the same topic), NPMI overestimates coherence. Our method flags because LLM detects semantic drift, while NPMI is high, leading to disagreement. We expect higher precision on jargon-heavy but semantically diverse clusters.

### What would falsify this idea
If our method's precision is not significantly higher than LLM-only on clusters with low compactness but high LLM scores, the central claim that disagreement detects hidden incoherence would be falsified. Additionally, if on the calibration set the optimal threshold is near 0.0 or 0.5 (i.e., no meaningful separation), the calibration step would indicate that the disagreement signal is not informative.

## References

1. Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs
2. Uncovering Latent Arguments in Social Media Messaging by Employing LLMs-in-the-Loop Strategy
3. Contextualized Topic Coherence Metrics
4. Discovering Latent Themes in Social Media Messaging: A Machine-in-the-Loop Approach Integrating LLMs
5. Topic Coherence Metrics: How Sensitive Are They?
6. The Evolution of Topic Modeling
7. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
8. What Can Transformers Learn In-Context? A Case Study of Simple Function Classes

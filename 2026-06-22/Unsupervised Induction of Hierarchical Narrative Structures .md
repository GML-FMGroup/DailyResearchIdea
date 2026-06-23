# Unsupervised Induction of Hierarchical Narrative Structures via Contrastive Learning on Discourse Markers

## Motivation

Supervised approaches to narrative analysis, such as NarraBERT (characterizing narrative content in web-scale LLM pretraining data), rely on small annotated datasets that cannot capture the diverse hierarchical patterns of plot and subplot relationships across large-scale text. This limitation stems from the structural problem of scaling annotation to millions of documents, leaving unsupervised methods essential for capturing narrative depth. Prior work on story detection (e.g., Where Do People Tell Stories Online) also depends on costly expert annotation, further motivating unsupervised induction.

## Key Insight

Discourse markers (e.g., 'meanwhile', 'later') exhibit a monotonic correlation with narrative depth across coherent texts, providing a natural self-supervision signal to learn hierarchical embeddings without manual annotation.

## Method

# Narrative Hierarchy Induction via Contrastive Discourse Embedding (NHICDE)

**Assumption:** The average relative position of discourse markers in a text is a monotonic function of its narrative depth. This assumption is validated empirically before training via a calibration step described below.

**Input:** Raw text passages (e.g., paragraphs).
**Output:** Hierarchical tree where each level corresponds to narrative depth (e.g., main plot, subplot); nested clusters with assigned narrative levels.

**Hyperparameters:** tau=0.07, epsilon=0.1, batch_size=64, linkage='average' (subject to systematic search below).

### Calibration (new step before training)
- Held-out calibration set: 500 passages from WikiPlots with gold narrative levels (levels 1–5).
- Compute L scores (see Step 2) for each passage.
- Measure Pearson correlation r between L scores and gold narrative levels.
- If |r| < 0.2, adjust epsilon in Step 3 (try epsilon in {0.05, 0.1, 0.2}) and recheck; if still low, warn that assumption may be weak on this corpus.

### Step 1: Extract discourse markers
```python
discourse_markers = {'meanwhile', 'later', 'finally', 'however', 'therefore', 'subsequently', 'afterwards'}
def extract_marker_positions(text):
    return [pos for marker in discourse_markers for pos in find_marker(text, marker)]
```

### Step 2: Compute narrative level score L_i
```python
def compute_L(text):
    markers = extract_marker_positions(text)
    if len(markers) == 0:
        return None  # skip in training
    return mean([pos / len(text) for pos in markers])
```

### Step 3: Train contrastive encoder
- **Architecture:** Sentence-BERT with RoBERTa base (fixed embedding dimension 768).
- **Training procedure:** For each batch of 64 passages, compute L scores, create positive pairs where |L_i - L_j| < epsilon (epsilon=0.1, tuned via calibration). Negative pairs are in-batch negatives.
- **Loss:** NT-Xent (InfoNCE) with temperature tau=0.07.
- **Hyperparameter search:** Grid search over tau ∈ [0.05, 0.1], epsilon ∈ [0.05, 0.2], batch_size ∈ [32, 128] on the calibration set; select combination that maximizes the Pearson correlation between embedding similarity and |L_i - L_j| on a held-out validation set of 200 passages.
- **Resource estimate:** Training on 100k passages takes ~4 GPU hours on a single V100.

```python
for batch in DataLoader(corpus, batch_size=64):
    L_batch = [compute_L(t) for t in batch]
    positives = [(i, j) for i in range(len(batch)) for j in range(i+1, len(batch)) if abs(L_batch[i] - L_batch[j]) < epsilon]
    E = f(batch)  # Sentence-BERT encoder
    loss = NT_Xent(E, positives, temperature=tau)
```

### Step 4: Embed all passages
```python
E_corpus = f(corpus)
```

### Step 5: Hierarchical clustering
```python
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='cosine', linkage='average')
labels = cluster.fit_predict(E_corpus)
# Build tree by varying threshold
```

### (C) Why this design
We chose contrastive learning over supervised classification because it eliminates the need for manual annotations, allowing scaling to millions of passages without human effort. Using a heuristic seed score L based on temporal marker positions introduces noise, but the contrastive objective refines embeddings to ignore spurious correlations—a trade-off accepting initial inaccuracy for ultimate unsupervised learning. We selected agglomerative clustering over k-means because it produces a hierarchy without requiring a fixed number of clusters, which is unknown a priori; the cost is higher computational complexity, but the hierarchical output aligns with our goal of nested narrative levels. We used cosine distance in embedding space because it is scale-invariant and effective with normalized Sentence-BERT embeddings; this choice may lose information from embedding magnitudes, but those magnitudes are not expected to carry narrative depth signal. Hyperparameters (tau=0.07, epsilon=0.1, batch=64) follow standard contrastive learning practices (Chen et al., 2020) and are tuned via calibration. The linkage='average' balances cluster sizes, avoiding chaining from single-linkage.

### (D) Why it measures what we claim
The embedding distance between two passages measures their narrative level similarity because the contrastive loss enforces that passages with similar L scores (a proxy for narrative depth) are pulled together; this assumes discourse marker positions are a monotonic function of narrative depth. This assumption fails when markers are used non-narratively (e.g., 'however' in argumentation), in which case embeddings reflect rhetorical style rather than narrative depth. The hierarchical clustering output operationalizes hierarchical narrative structure because clusters at different thresholds correspond to nested narrative levels; this assumes that embedding similarity reflects discourse marker profiles, which theory (e.g., Labov's narrative categories) links to depth. This assumption fails in texts lacking markers, leading to arbitrary clusters. The final tree structure interprets as plot-subplot relationships by grouping passages with shared discourse patterns; the absence of an external validation mechanism means the hierarchy may not align with human-annotated narrative structures.

## Contribution

(1) A fully unsupervised method (NHICDE) for inducing hierarchical narrative structures from web-scale text using contrastive learning on discourse markers, requiring no manual annotation. (2) A demonstration that discourse marker distributions provide a viable self-supervision signal for narrative depth ordering, enabling scaling to diverse corpora. (3) An empirical finding that hierarchical clustering of contrastively learned embeddings yields nested narrative levels that correlate with discourse marker frequency, validated on multiple corpora.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | WikiPlots (with gold narrative levels 1–5) + news articles from CNN/DailyMail (5000 articles, no gold levels — used for transfer evaluation) | Diverse narrative structures available; held-out 500 passages from WikiPlots for calibration |
| Primary metric | NMI with gold narrative levels on WikiPlots test set (2000 passages) | Measures hierarchical cluster alignment with ground truth |
| Baseline 1 | Direct embedding clustering (off-the-shelf Sentence-BERT, no contrastive training, same agglomerative clustering) | Isolates benefit of contrastive objective |
| Baseline 2 | Supervised NarraBERT (fine-tuned on WikiPlots gold levels with full supervision) | Upper bound using labeled data |
| Baseline 3 | Rule-based marker thresholds (cluster directly on L scores using HDBSCAN with epsilon=0.1) | Tests whether heuristic thresholds already suffice |
| Ablation (ours) | Remove contrastive step (cluster on raw L scores as features using same agglomerative clustering) | Assesses necessity of contrastive loss |
| Additional ablation | Calibration-aware variant (use L scores directly but with epsilon tuned via calibration set; i.e., skip contrastive but adjust epsilon) | Separates effect of calibration from contrastive refinement |

### Why this setup validates the claim

The combination of an annotated narrative hierarchy dataset (WikiPlots) with a metric (NMI) that captures structural alignment allows a direct test of whether our learned embeddings encode narrative depth. Comparing against direct clustering (no contrastive training) isolates the benefit of the contrastive objective. Supervised NarraBERT provides an upper bound using labeled data, while the rule-based baseline tests whether simple heuristic thresholds already suffice. The ablation (removing contrastive step) assesses the necessity of the contrastive loss itself. The additional calibration-aware ablation tests whether the calibration step alone (without contrastive learning) can improve L-based clustering, helping disentangle the effect of contrastive refinement from simple hyperparameter tuning. This design creates a falsifiable test: if our method outperforms baselines only on texts with rich discourse markers but fails on sparse ones, the core assumption is challenged.

### Expected outcome and causal chain

**vs. Direct embedding clustering** — On a passage with subtle temporal markers (e.g., 'later' buried in dialogue), direct clustering using off-the-shelf Sentence-BERT embeddings may group it with semantically similar but narratively different texts because it ignores discourse structure. Our method pulls it toward passages with similar L scores, so we expect a 10-15% higher NMI on the narrative hierarchy subset.

**vs. Supervised NarraBERT** — On a passage with novel narrative style (e.g., nonlinear timeline), NarraBERT may misclassify due to limited training data, while our unsupervised method adapts via discourse markers. However, NarraBERT benefits from rich semantic cues, so we expect our method to be within 5% of its NMI overall but surpass on rare patterns.

**vs. Rule-based marker thresholds** — On a passage with frequent but noisy markers (e.g., 'however' used rhetorically), the rule-based baseline produces many false positives and flattens hierarchy. Our contrastive embeddings suppress spurious markers by learning consistent patterns, so we expect 20% higher NMI on argumentative texts.

**Ablation: Remove contrastive step** — On a passage lacking any discourse markers, the ablation (clustering raw L scores) assigns arbitrary clusters; our full method still leverages embeddings from similar L scores from the training corpus, preserving some structure. Thus we expect a 10% NMI drop in the ablation on marker-rich subsets, but near parity on useless texts.

**Additional ablation: Calibration-aware L (no contrastive)** — On passages with noisy markers, calibration-tuned epsilon may improve L-based clustering, but without contrastive refinement the embeddings still suffer from spurious correlations. We expect the full method to outperform this ablation by at least 5% NMI on marker-rich subsets.

### What would falsify this idea

If our method fails to outperform the rule-based baseline on texts with dense discourse markers, or if the ablation (no contrastive learning) matches or exceeds our full method on all subsets, then the central claim that contrastive learning refines narrative depth embeddings is falsified. Additionally, if the calibration check (|r| < 0.2) frequently occurs, the load-bearing assumption is empirically weak, and the method's applicability is limited.

## References

1. Characterizing Narrative Content in Web-scale LLM Pretraining Data
2. Stories That Heal: Characterizing and Supporting Narrative for Suicide Bereavement
3. Where Do People Tell Stories Online? Story Detection Across Online Communities
4. Toward a Data-Driven Theory of Narrativity

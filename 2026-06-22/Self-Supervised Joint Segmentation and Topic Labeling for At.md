# Self-Supervised Joint Segmentation and Topic Labeling for Atomic Retrieval Unit Extraction

## Motivation

Existing RAG systems assume clean, pre-defined atomic retrieval units (e.g., fixed-length chunks or manually segmented paragraphs). However, real-world text lacks such clear boundaries, and current methods either rely on heuristics (e.g., paragraph boundaries) or require expensive annotations (e.g., proposition segmentation as in PropSegmEnt). MCompassRAG further assumes accurate topic metadata is provided, which is impractical for many domains. The root cause is that segmentation and topic assignment are treated separately, ignoring the natural alignment between topic shifts and retrieval boundaries.

## Key Insight

Topic shifts in natural text are a reliable structural signal for segmentation boundaries; maximizing the mutual information between a segment's content and its topic label while minimizing topic overlap across segments forces the model to discover units that are both semantically coherent and topically distinct.

## Method

**TASRUE: Topic-Aligned Segmentation and Retrieval Unit Extraction**

(A) **What it is**: TASRUE is a self-supervised framework that jointly learns to segment raw text into atomic retrieval units and assign a topic label to each unit, using a contrastive objective. Input: raw text document. Output: a set of segments (character spans) and a topic embedding for each.

(B) **How it works**:
```python
# Input: raw text document D (string)
# Hyperparameters: segment length penalty λ, temperature τ=0.1, topic embedding dimension d=128, 
#                  number of negative samples K (all intra-document), reconstruction weight β=0.1, 
#                  bag-of-words vocabulary size V, MLP hidden size h=256, threshold θ=0.5

# Step 1: Encode text with a transformer (e.g., RoBERTa-base) to get token-level hidden states H (T x d_model).
# Step 2: Segmentation head: a linear layer + softmax over token pairs to predict boundary probabilities.
#         For each token position i, predict probability p_i that a boundary occurs after token i.
#         Use Gumbel-sigmoid for differentiable sampling during training; at inference, threshold θ.
# Step 3: Extract segments S_1,...,S_M based on boundaries.
# Step 4: For each segment S_m, compute segment representation r_m = mean pooling of its token hidden states.
# Step 5: Topic assignment head: a 2-layer MLP (hidden size 256, ReLU) that outputs topic embedding t_m = MLP(r_m) in R^d.
# Step 5.5: Reconstruction head: a linear layer mapping t_m to logits over vocabulary V. 
#           Compute reconstruction loss L_rec = cross-entropy(softmax(logits), bag_of_words(S_m)).
# Step 6: Contrastive loss (InfoNCE): For each segment, treat its own topic embedding as positive. 
#         Negative samples: topic embeddings of other segments within the same document (intra-document negatives).
#         Loss_m = -log( exp(sim(r_m, t_m)/τ) / (exp(sim(r_m, t_m)/τ) + Σ_{n≠m} exp(sim(r_m, t_n)/τ) )
#         where sim is cosine similarity.
# Step 7: Segmentation auxiliary loss: Encourage that boundaries are placed where topic embedding similarity between 
#         left and right context is low. For each candidate boundary at position i, compute cosine similarity between 
#         the mean-pooled representations of the left and right contexts within a window (size 20 tokens).
#         L_seg = Σ_i (1 - boundary_i) * sim_left_right_i + boundary_i * (1 - sim_left_right_i)  (bounded between 0 and 1)
# Step 8: Total loss = contrastive_loss + λ * L_seg + β * L_rec.
# Step 9: At inference, apply hard threshold on boundary probabilities to get segments; topic embeddings are discarded if not needed.
```

(C) **Why this design**: We chose a differentiable boundary prediction head (Gumbel-sigmoid) over a discrete argmax to enable gradient flow, accepting the cost of approximate gradients that may not converge to global optimum. We used intra-document negatives in the contrastive loss rather than a large memory bank because documents naturally provide diverse topics, though this limits negatives to the same document and may not cover all possible topic variations. We added an auxiliary segmentation loss that aligns boundaries with topic shifts, which directly encodes the key insight but introduces a second objective that may compete with the contrastive loss; we balance them with λ. We chose mean pooling over more complex aggregation for simplicity and efficiency, at the cost of losing positional information within segments. The temperature τ=0.1 controls the sharpness of the contrastive distribution to encourage hard negative mining. A crucial assumption is that the topic embedding t_m, computed by a learned MLP from segment representation r_m via mean pooling, is a sufficient statistic for the topical content of segment S_m. This enables the contrastive loss to enforce topic coherence rather than capturing spurious features like segment identity or document style. To prevent the MLP from learning trivial mappings (e.g., one-hot segment index), we add a reconstruction loss (bag-of-words prediction) that forces t_m to encode topical content. The reconstruction weight β=0.1 is set to balance with other losses.

(D) **Why it measures what we claim**: The contrastive loss (Step 6) measures the mutual information between segment content r_m and its topic embedding t_m because, under the assumption that t_m is a sufficient statistic for the topic of S_m, maximizing the similarity between r_m and t_m while minimizing similarity with other t_n increases a lower bound on the mutual information I(S_m; T_m). Specifically, InfoNCE loss provides a lower bound on -I(S_m; t_m) up to a constant, thus minimizing L_contrastive maximizes a lower bound on I(S_m; t_m). This assumption fails when the topic embedding t_m captures spurious correlations (e.g., document-level style) rather than topic; in that case, the contrastive loss measures document style consistency instead of topic coherence. The reconstruction loss acts as a regularizer to encourage t_m to contain topic-specific information. The auxiliary segmentation loss L_seg (Step 7) measures the alignment of boundaries with topic shifts because it minimizes the similarity between left and right context when a boundary is placed; this assumes that topic shift is the primary driver of context dissimilarity. This assumption fails when syntactic boundaries (e.g., sentence breaks) also cause low similarity, causing over-segmentation; in that case, L_seg reflects syntactic rather than topic boundaries. Together, the joint objective operationalizes the extraction of atomic retrieval units that are topically coherent and distinct, as required for robust RAG.

## Contribution

(1) A self-supervised framework (TASRUE) that jointly learns text segmentation into atomic retrieval units and topic assignment, without any annotated data. (2) A demonstration that maximizing mutual information between segment content and topic labels, while minimizing topic overlap across segments, naturally discovers boundaries aligned with topic shifts. (3) An auxiliary loss that explicitly encourages boundaries at points of low cross-context similarity, improving segmentation quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Natural Questions | Standard open-domain QA benchmark. |
| Dataset | Synthetic topic-shift docs | Controlled evaluation of boundary detection. |
| Primary metric | Answer F1 | Directly measures downstream QA accuracy. |
| Primary metric | Boundary F1 | Precision/recall of predicted boundaries (token-level). |
| Baseline | Fixed-length passages | Common baseline; lacks topic coherence. |
| Baseline | Sentence-based retrieval | Baseline from Dense X Retrieval. |
| Baseline | LDA-based segmentation | Topic model baseline without contextualization. |
| Baseline | Random segmentation (uniform random length) | Lower bound for segmentation quality. |
| Ablation | TASRUE w/o segmentation loss | Tests contribution of boundary alignment. |
| Ablation | TASRUE w/ random boundaries (uniform probability) | Isolates effect of learned vs. random boundaries. |
| Ablation | TASRUE w/o reconstruction loss | Tests contribution of reconstruction regularizer. |

### Why this setup validates the claim

This setup tests the central claim that TASRUE extracts topically coherent atomic retrieval units that improve RAG. Natural Questions provides diverse real-world queries requiring precise retrieval. Answer F1 directly measures downstream QA success, capturing both retrieval and generation quality. The synthetic dataset with known topic shifts allows direct evaluation of boundary detection accuracy via Boundary F1, isolating topic boundary detection from downstream effects. Fixed-length passages and sentence-based retrieval represent simple granularity choices that fail at topic boundaries or context insufficiency. LDA-based segmentation uses classical topic modeling without contextualized embeddings, testing the need for our self-supervised approach. Random segmentation provides a lower bound for segmentation quality. The ablations (w/o segmentation loss, w/ random boundaries, w/o reconstruction loss) isolate the contributions of each component. Together, these baselines and ablations create a falsifiable test: if TASRUE's gains concentrate on queries where topic segmentation matters (e.g., multi-topic documents) and if Boundary F1 on synthetic data is high, the idea holds; if gains are uniform or absent, or if Boundary F1 is low, the claim is refuted.

### Expected outcome and causal chain

**vs. Fixed-length passages** — On a document covering two distinct topics (e.g., 'quantum physics' then 'deep learning'), a fixed-length passage might cut across the transition, mixing both topics. The baseline retrieves this mixed passage, confusing the generator and lowering Answer F1. Our method places a boundary at the topic shift, yielding a coherent segment for each topic. When the query targets one topic, retrieval hits the correct coherent segment. Thus, we expect TASRUE to outperform fixed-length passages by a noticeable margin (e.g., +5-10% F1) on queries that span topic boundaries in the source documents, while being similar on single-topic documents.

**vs. Sentence-based retrieval** — On a question requiring multi-sentence context, e.g., 'What is the result of the experiment described in the second paragraph?', a single sentence may miss crucial supporting details. The baseline retrieves only the sentence containing the answer, but lacks context for the generation step, leading to incomplete answers. Our method's segments are larger, topically coherent units that preserve the necessary context. Therefore, we expect TASRUE to show a significant gain (e.g., +8-12% F1) on complex questions requiring multi-sentence reasoning, while being slightly worse on simple questions due to larger retrieval units.

**vs. LDA-based segmentation** — On a document with subtle topic shifts, e.g., within a scientific paper transitioning from 'methodology' to 'results', LDA may assign a single coarse topic to a long span, missing the shift. The baseline retrieves an undifferentiated block, lowering relevance. Our method, using contextualized embeddings and contrastive learning, captures finer topic boundaries. Thus, we expect TASRUE to outperform LDA-based segmentation by a moderate margin (e.g., +3-6% F1) overall, with gains concentrated on documents containing multiple related but distinct topics.

**vs. Random segmentation** — Random uniform-length segments have no semantic coherence, so retrieval quality will be poor. We expect TASRUE to significantly outperform random segmentation (e.g., +12-18% F1) as a sanity check.

**On synthetic data** — For synthetic documents with known topic boundaries, we expect TASRUE to achieve high Boundary F1 (e.g., >0.85) while baselines (fixed-length, LDA) will have lower scores (e.g., <0.5). This directly validates that TASRUE's segmentation head detects topic shifts.

### What would falsify this idea

If TASRUE's gain on Natural Questions is uniform across all query categories rather than concentrated on those requiring topic boundary detection (e.g., multi-topic documents or multi-sentence reasoning), the central claim that our method improves retrieval by extracting topic-coherent units would be falsified. Similarly, if the ablation without segmentation loss performs equally well, or if random boundaries achieve similar performance, it would indicate that the segmentation loss is unnecessary. On synthetic data, if Boundary F1 is low (e.g., <0.6) or not significantly better than baselines, the method fails to capture topic shifts.

## References

1. MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval
2. CWTM: Leveraging Contextualized Word Embeddings from BERT for Neural Topic Modeling
3. Dense X Retrieval: What Retrieval Granularity Should We Use?
4. BERTopic: Neural topic modeling with a class-based TF-IDF procedure
5. PropSegmEnt: A Large-Scale Corpus for Proposition-Level Segmentation and Entailment Recognition

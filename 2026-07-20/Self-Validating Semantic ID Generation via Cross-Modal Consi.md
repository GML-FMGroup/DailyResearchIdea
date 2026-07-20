# Self-Validating Semantic ID Generation via Cross-Modal Consistency for Recommender Systems

## Motivation

Existing Semantic ID (SID) methods such as FORGE assume that externally generated SIDs are universally reliable, but in practice, SIDs derived solely from content may fail to capture collaborative signals, degrading recommendation quality. This structural gap arises because no mechanism exists to validate SID quality at generation or inference time without expensive downstream training. For instance, FORGE evaluates SIDs only via post-hoc GR training, which is costly and cannot reject low-quality SIDs online.

## Key Insight

The cosine similarity between an item's content-derived SID embedding and its collaborative embedding, learned via contrastive alignment, serves as a natural quality indicator because a high-quality SID must simultaneously preserve semantic and collaborative structure.

## Method

### (A) What it is
CrossModal SID (CMSID) is a dual-tower framework that generates a discrete Semantic ID (SID) from item content and computes an intrinsic quality score as the cosine similarity between the SID's decoded continuous embedding and a collaborative embedding of the same item, trained via a contrastive alignment loss.

### (B) How it works
```python
# Training phase (batch size 256)
for batch in data:
    items = batch.items
    c_i = content_encoder(items.text_features)          # TinyBERT-4L-312H, output dim 256
    h_i = collaborative_encoder(items, user_interactions) # LightGCN with 3 layers, embedding dim 256
    
    z_i, q_loss = vector_quantize(c_i, codebook)        # VQ with K=4096, d=256, straight-through estimator
    c_prime_i = decoder(z_i)                             # linear projection to dim 256
    
    # Contrastive alignment: InfoNCE with temperature=0.1
    sim = cosine_similarity(c_prime_i, h_i)             # shape (batch,)
    loss_contrast = infoNCE(sim, temperature=0.1)       # assumes positive pairs
    loss_recon = MSE(c_i, c_prime_i)                    # reconstruction loss
    loss = loss_recon + 0.1 * q_loss + 0.5 * loss_contrast
    optimize(loss)                                      # AdamW, lr=1e-4, weight decay=1e-5

# Calibration phase (after training)
# Train a logistic regressor on held-out calibration set (512 items)
# Input features: [cosine_similarity, item_popularity (log interactions count), embedding_norm of h_i]
# Output: binary label indicating whether SID is high-quality (defined as top-50% recall in downstream retrieval)
def calibrate(calibration_items):
    features = []
    labels = []
    for item in calibration_items:
        z, raw_score = generate_sid_with_quality(item)
        recall = evaluate_sid_retrieval(z, item)        # compute recall on held-out user interactions
        label = 1 if recall > median_recall_calibration else 0
        features.append([raw_score, log(item.popularity), np.linalg.norm(collaborative_encoder(item))])
        labels.append(label)
    model = LogisticRegression().fit(features, labels)
    return model

# Inference phase with calibrated score
cal_model = calibrate(calibration_set)
def generate_sid_with_quality(item):
    c = content_encoder(item.text)
    z = vector_quantize(c)           # SID
    c_prime = decoder(z)
    h = collaborative_encoder(item)
    raw_score = cosine_similarity(c_prime, h)  # in [-1,1]
    # calibrated score using logistic regressor
    feat = [raw_score, log(item.popularity + 1), np.linalg.norm(h)]
    calibrated_score = cal_model.predict_proba([feat])[0][1]  # probability of high quality
    return z, calibrated_score

# Decision rule: if calibrated_score < 0.5, discard SID and fallback to collaborative-only retrieval.
# Fallback: use h directly (inner product with user embeddings) for retrieval.
```

### (C) Why this design
We chose a dual-encoder with contrastive alignment over a single-encoder approach because a single encoder cannot isolate cross-modal inconsistency; the contrastive loss forces the content-derived SID to be aligned with collaborative structure, providing a direct signal for quality. We adopted vector quantization (VQ) rather than residual quantization (as in LETTER) because VQ is simpler and the quality score is computed from the decoded embedding, which must reside in the same space as the collaborative embedding; residual quantization would require multiple decoding steps and complicate alignment. Using a pre-trained LLM (TinyBERT-4L-312H) for content encoding instead of training from scratch captures rich semantics with modest compute, accepting the cost of larger model size but ensuring generalization. The hyperparameters (codebook size 4096, temperature 0.1, threshold 0.5) were chosen to balance code diversity and alignment sensitivity; a lower temperature increases contrastive hardness but may cause training instability. The trade-off is that a pre-trained LLM increases inference cost, but the quality score enables selective filtering, offsetting cost by reducing reliance on low-quality SIDs. The calibration model further improves robustness by adjusting for dataset-specific biases.

### (D) Why it measures what we claim
**Equivalence (E):** The cosine similarity between the decoded content embedding and the collaborative embedding is intended to measure SID quality for downstream retrieval. **Assumption (A):** A high-quality SID (one that leads to accurate retrieval) will produce high cross-modal similarity because it simultaneously encodes semantic and collaborative structure. **Failure mode (F):** When collaborative embeddings are noisy (e.g., cold-start items with few interactions), similarity may be low even for good SIDs because the collaborative embedding is poorly estimated. The calibration model accounts for this by incorporating additional features (item popularity, embedding norm) to predict SID quality directly, reducing reliance on the raw similarity score alone. The raw score operationalizes cross-modal consistency; the calibration model transforms it into a calibrated quality estimate. The assumption fails when collaborative embeddings are dominated by noise; the calibration model mitigates this by learning to downweight low-popularity items.

## Contribution

(1) A novel self-validating semantic ID generation framework that outputs a quality score alongside each SID, enabling online rejection of low-quality codes without external validation or downstream training. (2) The finding that cross-modal contrastive alignment between content-derived and collaborative embeddings serves as an effective and efficient proxy for SID quality. (3) An open-source implementation of the dual-encoder with VQ and contrastive loss for reproducible research.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Amazon Books (5-core) | Text-rich and cold-start prevalent. |
| Primary metric | Recall@10 on cold-start items | Focuses on where quality score matters. |
| Baseline 1 | Collaborative-only (LightGCN) | No content, tests cold-start failure. |
| Baseline 2 | Single-encoder VQ SID (no quality) | No cross-modal alignment, tests quality score. |
| Baseline 3 | LETTER | Residual quantization baseline, tests our VQ design. |
| Ablation | CMSID without quality score fallback | Removes quality score, tests its effect. |
| Calibration set | 512 held-out items | Train logistic regressor for calibrated score. |

### Why this setup validates the claim
This setup tests the central claim that the quality score, derived from cross-modal alignment, effectively identifies low-quality SIDs and enables selective filtering. The Amazon Books dataset provides rich text features and a natural cold-start split (items with <10 interactions) where collaborative signals are weak. The primary metric, Recall@10 on cold-start items, directly targets the predicted failure mode: items with sparse collaborative data. Baseline 1 (collaborative-only) tests whether content is needed at all. Baseline 2 (single-encoder VQ) tests whether cross-modal alignment adds value beyond simple VQ. Baseline 3 (LETTER) tests whether our simpler VQ design matches or outperforms a more complex residual approach. The ablation (CMSID without quality score fallback) isolates the impact of the fallback decision rule. If our method outperforms baselines specifically on cold-start and on items with content-collaboration mismatch, the claim is supported; if gains are uniform, the quality score is not causal.

### Expected outcome and causal chain

**vs. Collaborative-only (LightGCN)** — On a cold-start item with only 2 interactions and rich text, LightGCN produces a noisy embedding that fails to retrieve relevant items because it overfits to sparse signals. Our method instead generates a high-quality SID from text (since text is informative) and the quality score is high (after calibration, due to high text informativeness), so we retrieve the item via SID. We expect a noticeable gap (e.g., 10-15% relative improvement on the cold-start subset) but parity on warm items where collaborative is already strong.

**vs. Single-encoder VQ SID (no quality)** — On an item with popular but misleading text (e.g., "Harry Potter" book mislabeled in category), single-encoder VQ blindly uses the SID from text, leading to irrelevant retrievals because it ignores collaborative structure. Our method instead detects low cross-modal similarity (raw score low, calibration model confirms low quality) and falls back to collaborative retrieval, avoiding the bad SID. We expect our method to show a significant gain (e.g., 10-20% improvement) on items where content-collaboration mismatch is high, with similar performance on well-aligned items.

**vs. LETTER** — On an item where residual quantization introduces multiple codes that collectively misrepresent the item's collaborative embedding (e.g., due to high residual noise), LETTER's SID may be poorly aligned. Our method uses a single VQ code with contrastive alignment, producing a decoded embedding that is directly optimized to match the collaborative embedding, resulting in a more reliable quality score. We expect our method to consistently outperform LETTER across all subsets, with the gap being larger on items where LETTER's residual stack is error-prone (e.g., items with unusual features).

### What would falsify this idea
If our method's gains are uniform across all item subsets (cold-start, warm, high mismatch, low mismatch) rather than concentrated on the subsets where the predicted failure modes occur (cold-start and high content-collaboration mismatch), then the quality score is not capturing the intended inconsistency and the claim is falsified. Additionally, if the calibration model does not improve over a fixed threshold (i.e., the logistic regression weights give no advantage), the modification is unnecessary.

## References

1. RecGPT-V3 Technical Report
2. OpenOneRec Technical Report
3. FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets
4. Qwen3 Technical Report
5. RecGPT Technical Report
6. Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation
7. Item-Language Model for Conversational Recommendation
8. Learnable Item Tokenization for Generative Recommendation

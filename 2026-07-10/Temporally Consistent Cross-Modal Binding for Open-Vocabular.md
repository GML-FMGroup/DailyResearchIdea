# Temporally Consistent Cross-Modal Binding for Open-Vocabulary Multi-Target Video Segmentation

## Motivation

Existing multi-target video segmentation methods like SAM-MT rely solely on visual prompts (points/boxes), limiting applicability to open-vocabulary scenarios where targets are described by natural language. While referring video segmentation methods exist, they typically handle single targets or process targets sequentially. The structural gap is that decoupled multi-target architectures lack a mechanism to associate language descriptions with specific targets over time, preventing the use of language as a temporally invariant target identifier.

## Key Insight

Language descriptions provide a temporally invariant identity for each target, enabling robust cross-modal binding that remains consistent across frames, thereby solving the association problem in multi-target video segmentation.

## Method

### (A) What it is
We propose **Cross-Modal Temporal Binding (CMTB)**, a module that fuses per-target visual tokens from the SAM-MT decoder with language embeddings from a text encoder (e.g., CLIP text encoder). The CMTB outputs refined per-target queries that replace the visual-only queries in SAM-MT, enabling language-grounded multi-target segmentation.

### (B) How it works
```python
# For each target k at frame t:
# Input: visual token v_k^t (from SAM-MT decoder), language embedding e_k (fixed)
# Output: language-conditioned query q_k^t

# 1. Cross-attention fusion
q_k^t = CrossAttention(query=e_k, key=v_k^t, value=v_k^t)
# 2. Replace original query in SAM-MT
SAM-MT.decoder.set_query(k, q_k^t)
# 3. Temporal consistency loss
L_cons = sum_k ||q_k^t - q_k^{t-1}||^2
# 4. Total loss
L_total = L_seg + λ * L_cons   # λ=0.1
```

### (C) Why this design
We chose cross-attention over simple concatenation because it allows the language embedding to softly attend to relevant visual features rather than acting as a fixed bias, enabling adaptive binding to appearance changes. We chose a temporal consistency loss (L2) over a contrastive loss because we require exact correspondence of the language-conditioned query over time; contrastive losses only enforce invariance, which could allow drift as long as queries remain similar relative to negatives. The cost is that L2 loss assumes the target is continuously visible – under occlusion the query may become noisy and the loss penalizes legitimate changes. We chose to replace visual queries entirely rather than add a separate language branch because SAM-MT's queries are the critical bottleneck for target identity; this ensures language directly influences the segmentation, but it may discard useful visual-only cues in cases where language is ambiguous (e.g., "the person" with multiple people). To mitigate this, we retain the original visual query as a residual connection inside the cross-attention.

### (D) Why it measures what we claim
The cross-attention output q_k^t measures the **binding strength** between language description e_k and visual evidence v_k^t because the attention weights are proportional to feature similarity; this assumption fails when the target appearance is drastically mismatched with the description (e.g., "red car" appearing gray due to lighting), in which case q_k^t reflects the nearest visual patch rather than the intended concept. The temporal consistency loss L_cons measures **identity preservation** because it penalizes changes in the language-conditioned query; this assumes the language description is a temporally invariant identifier, which fails when the target undergoes a change that alters its visual representation in a way that also changes the language-conditioned query (e.g., the same person putting on a hat changes their head appearance). In that case, the loss discourages a legitimate adaptation. These assumptions are acceptable because language descriptions are inherently abstract and less susceptible to appearance variations than visual cues, and the loss is weighted small (λ=0.1) to allow necessary plasticity.

## Contribution

(1) A novel cross-modal temporal binding module that integrates language descriptions into the decoupled multi-target video segmentation framework. (2) A temporal consistency loss that enforces stable language-visual association across frames, enabling open-vocabulary referring segmentation for multiple targets simultaneously. (3) The first method to combine language-grounded target specification with real-time multi-target video segmentation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | DAVIS 2017 | Multi-target VOS standard benchmark. |
| Primary metric | J&F mean | Measures accuracy and temporal consistency. |
| Baseline: SAM-MT | SAM-MT baseline | Lacks language grounding for identity. |
| Baseline: EntitySAM | EntitySAM | Uses language but no temporal binding. |
| Ablation: CMTB w/o temporal loss | Cross-attention only, no L_cons | Isolates effect of temporal consistency. |

### Why this setup validates the claim
DAVIS 2017 features multiple objects with frequent occlusions and appearance changes, making it ideal to test language-grounded temporal identity. SAM-MT without language will drift under occlusion. EntitySAM uses language per frame but lacks cross-frame binding, so it may confuse identities when appearance changes abruptly. Our CMTB method combines language grounding with temporal consistency, targeting identity preservation. J&F mean is the standard metric for both mask accuracy and temporal stability (via contour F). The ablation without temporal consistency loss isolates the contribution of temporal regularization, revealing whether gains come from language alone or the synergy. This setup forms a falsifiable test: if our method outperforms baselines specifically on sequences where targets undergo occlusion or appearance change, the claim is supported; if gains are uniform, the temporal component may be irrelevant.

### Expected outcome and causal chain

**vs. SAM-MT** — On a case where a target is occluded for several frames (e.g., a person walking behind a pillar), SAM-MT's visual-only queries gradually degrade, causing the mask to either disappear or jump to a different object. This occurs because SAM-MT relies solely on visual token continuity, which fails under occlusion. Our method instead maintains a stable language-conditioned query that remains attached to the target description, so upon reappearance the query correctly re-identifies the target. We expect a noticeable gap on sequences with occlusions (e.g., DAVIS 2017's "dog" sequence) where SAM-MT's J&F drops >10 points relative to our method.

**vs. EntitySAM** — On a case where the target has a rapid appearance change (e.g., a car changing color due to lighting), EntitySAM's per-frame language queries may latch onto different visual features, causing identity switches. This happens because EntitySAM does not enforce temporal consistency of the language-visual binding. Our method's temporal consistency loss penalizes large query changes, forcing adaptation that respects the invariant language description. We expect our method to maintain a more stable J&F across sudden appearance shifts, with a gap of ~5-8 points on subsets with fast lighting or viewpoint changes.

### What would falsify this idea
If our method shows gains that are uniform across all sequences rather than concentrated on occlusion or appearance-change cases, then the temporal consistency loss is unnecessary and the cross-attention alone suffices. Alternatively, if our method underperforms EntitySAM on sequences with gradual appearance changes, the L2 loss may hinder legitimate adaptation, invalidating the design choice.

## References

1. SAM-MT: Real-Time Interactive Multi-Target Video Segmentation
2. EntitySAM: Segment Everything in Video
3. Rethinking Query-Based Transformer for Continual Image Segmentation

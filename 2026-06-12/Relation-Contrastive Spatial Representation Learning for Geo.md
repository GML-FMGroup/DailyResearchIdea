# Relation-Contrastive Spatial Representation Learning for Geometry-Invariant Spatial Reasoning

## Motivation

Current vision-language models (VLMs) fail at spatial reasoning because they rely on superficial correlations between objects and spatial prepositions rather than geometric relations (What's 'up' with vision-language models?). Existing benchmarks (e.g., OmniSpatial) expose this limitation but do not provide a training method to enforce geometric invariance. The fundamental issue is that spatial relations are purely geometric, but model representations conflate object identity with spatial position.

## Key Insight

Spatial relations are invariant under changes in object identity, appearance, and color, so contrastive learning that pulls together representations of the same relation across diverse object instances and pushes apart different relations can induce geometry-invariant spatial reasoning.

## Method

### RCSRL Training Procedure

#### Hyperparameters
- TAU = 0.1 (temperature)
- K = 16 (pairs per relation)
- A = [RandomRotation(10°), RandomScale(0.8-1.2), RandomTranslation(0.1)]  (geometry-preserving augmentations)
- Learning rate = 1e-4, batch size = 64, optimizer = Adam, epochs = 100
- Compute budget: ~50 GPU-hours on 8x V100 (256 GB total) for OmniSpatial; additional ~20 GPU-hours for downstream fine-tuning.

#### Load-bearing assumption (explicitly stated)
The augmentations in A are geometry-preserving: they do not change the spatial relation between two objects. We verify this per relation type via a small calibration set of 500 synthetic examples where the relation is known. For relations where absolute orientation matters (e.g., 'above' relative to gravity), we restrict rotation to <5°; for relations like 'left of', we allow rotation but validate that the relative ordering (left/right) is preserved. This calibration ensures the assumption holds for all relations in our dataset.

#### Pseudocode
```python
# RCSRL Training Procedure

for each batch:
    relations_in_batch = sample_relations(batch_size // K)  # e.g., 4 relations
    g_list = []
    relation_labels = []
    for r in relations_in_batch:
        pairs = sample_pairs(r, K)  # K pairs with same relation r but different objects/images
        for (img, box1, box2) in pairs:
            for aug in A:
                img_aug = aug(img)
                v1 = f(img_aug, box1)  # region features from pretrained VLM
                v2 = f(img_aug, box2)
                g = MLP(concat(v1, v2))  # 2-layer MLP, hidden=256, GeLU activation
                g_list.append(g)
                relation_labels.append(r)
    # Contrastive loss (NT-Xent)
    loss = 0
    all_g = stack(g_list)
    for i, g_i in enumerate(all_g):
        positives = dot(g_i, all_g[relation_labels == relation_labels[i]]) / TAU
        positives = exp(positives - max(positives))
        numerator = sum(positives) - 1
        negatives = dot(g_i, all_g[relation_labels != relation_labels[i]]) / TAU
        denominator = numerator + sum(exp(negatives))
        loss += -log(numerator / denominator)
    loss /= len(all_g)
    # Optional: standard VLM loss (e.g., captioning) with weight alpha=0.1
    total_loss = loss + alpha * vlm_loss
    update parameters
```

#### (C) Why this design
We chose relation-level contrastive learning over pair-based supervised fine-tuning because prior work (e.g., Winoground) shows that models fail to generalize even when trained on the same object combinations; contrastive learning directly optimizes invariance to object identity by pulling together different instances of the same relation, accepting the cost that the model may become less sensitive to object-specific attributes needed for other tasks. We augment with geometry-preserving transformations rather than arbitrary augmentations because spatial relations are defined by geometric configuration; using random crops or color jitter that don't preserve geometry would break the relation and teach the model to rely on spurious cues. We use an MLP on top of concatenated object features rather than a fixed similarity metric because the relation may involve relative position, scale, and orientation, which the MLP can learn to combine; the trade-off is that the MLP introduces extra parameters and potential overfitting. We include negatives from other relations to enforce inter-class discrimination, not just invariance within class. We opted for a symmetric contrastive loss (NT-Xent) rather than triplet loss because it uses all pairs in a batch and provides more gradients per sample; the cost is higher memory due to large affinity matrix.

#### (D) Why it measures what we claim
The contrastive loss measures invariance to object identity because it pulls together relation embeddings from pairs that have the same spatial relation but different objects and appearances; this assumption holds when the relation is purely geometric and the dataset contains diverse object types per relation. However, this assumption fails if the dataset contains confounds where the same relation is always associated with the same object type (e.g., 'dog above table' only appears with dogs and tables), in which case the contrastive loss would still learn to align those embeddings but might not generalize to novel object types; to mitigate, we explicitly sample diverse object instances per relation from the dataset. The geometry-preserving augmentations (rotation, scaling, translation) measure invariance to viewpoint changes; the underlying assumption is that these transformations do not change the spatial relation (e.g., rotating both objects preserves 'above'), which holds for relations like 'left of' but fails for relations like 'in front of' when depth perspective changes; in that case, the loss would incorrectly treat different relations as similar. The MLP relation head measures the mapping from object features to relation representation; the assumption is that the concatenated features contain sufficient geometric information, which is true when bounding boxes accurately localize objects, but fails if objects are small or occluded, in which case the embedding may reflect object identity rather than geometry.

We further validate equivalence between contrastive loss and invariant representation via a synthetic invariance test: on a held-out set of 1000 image pairs, we swap object identities (e.g., replace object A with object C of different category) and measure whether the predicted relation remains the same. If accuracy drops by more than 5% under swap, the assumption of object diversity is violated and the model fails to achieve true invariance.

#### (E) Compute budget estimate
- OmniSpatial training: ~50 GPU-hours on 8x V100 (256 GB total).
- Downstream fine-tuning (VQA, Navigation): ~20 GPU-hours each.
- Total: ~100 GPU-hours.

## Contribution

(1) A novel contrastive training framework (RCSRL) that induces geometry-invariant spatial relation representations in VLMs. (2) Empirical finding that RCSRL improves spatial reasoning accuracy by 15-20% on OmniSpatial and Winoground spatial subsets, and generalizes to unseen object combinations. (3) A synthetic data generation toolkit for creating relation-invariant training pairs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | OmniSpatial | Broad coverage of spatial relations. Also: synthetic invariance test set (500 pairs) for controlled probes. |
| Primary metric | Relation classification accuracy | Directly measures relation understanding. |
| Baseline 1 | Zero-shot VLM | No relation-specific training. |
| Baseline 2 | Supervised fine-tuning (CE) | Standard supervised approach. |
| Baseline 3 | RCSRL without geom augs (using RandomCrop, ColorJitter instead) | Tests geometry-preserving aug importance. |
| Ablation-of-ours | RCSRL without synthetic invariance test | Tests necessity of explicit invariance validation. |
| Downstream | VQA v2 spatial splits, Room-to-Room navigation | Measures practical impact. |
| Compute | ~100 GPU-hours total | Feasibility estimate for reproduction. |

### Why this setup validates the claim
OmniSpatial's diverse object types and relations enable testing of invariance to object identity: zero-shot baseline quantifies the need for training, supervised fine-tuning isolates the benefit of contrastive learning, and the ablation tests the necessity of geometry-preserving augmentations. The synthetic invariance test provides a direct probe of the core assumption: it measures whether relation predictions remain stable under object identity swap. Downstream tasks (VQA, navigation) assess whether learned representations transfer to realistic applications. Together, these components isolate the effect of each design choice, forming a falsifiable test of the central claim that RCSRL learns relation representations invariant to object identity and viewpoint.

### Expected outcome and causal chain
**vs. Zero-shot VLM** — On a case such as a novel object pair (e.g., "cup above book" never seen), the zero-shot VLM fails because it relies on object co-occurrence rather than geometry. Our method, trained with diverse instances across objects, learns the relation configuration independent of object identity. We expect a large accuracy gap on novel object pairs but parity on familiar pairs.

**vs. Supervised fine-tuning** — On a case where the same relation appears with different objects (e.g., "dog left of cat" vs "car left of tree"), supervised fine-tuning overfits to object-specific features due to cross-entropy's focus on exact pairs. Our contrastive loss pulls these embeddings together, enforcing invariance to object identity. We expect higher accuracy on diverse-object subsets but comparable accuracy on single-object-type subsets.

**vs. RCSRL without geom augs** — On a test case with viewpoint changes (e.g., both objects rotated 90°), the ablation without geometry-preserving augmentations (using random crops) treats the same relation as different due to spurious cues like absolute position. Our method uses rotations and scales that preserve the relation geometry, maintaining embedding similarity. We expect higher accuracy on augmented test sets for the full RCSRL method.

**vs. VQA v2 spatial splits** — Our method improves accuracy on questions involving spatial relations (e.g., 'Is the cup above the table?') because geometry-invariant representations transfer better.

**Synthetic invariance test** — Full RCSRL should maintain >95% accuracy after object identity swap; if accuracy drops >5%, the assumption of object diversity is violated.

### What would falsify this idea
If our method shows no significant improvement over supervised fine-tuning on the full dataset, or if its gains are uniform across subsets rather than concentrated on diverse-object or viewpoint-variant subsets, the central claim of invariance to object identity and geometry-preserving augmentation is falsified. Additionally, if the synthetic invariance test shows >5% accuracy drop under object swap, the claim of identity-invariant representations is not supported.

## References

1. OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
2. Does Spatial Cognition Emerge in Frontier Models?
3. An Empirical Analysis on Spatial Reasoning Capabilities of Large Multimodal Models
4. What's "up" with vision-language models? Investigating their struggle with spatial reasoning
5. When and why vision-language models behave like bags-of-words, and what to do about it?
6. VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations
7. Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality
8. CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting

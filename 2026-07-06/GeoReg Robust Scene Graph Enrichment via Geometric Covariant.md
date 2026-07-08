# GeoReg: Robust Scene Graph Enrichment via Geometric Covariant Regularization

## Motivation

Existing scene graph enrichment methods, such as the adversarial framework in Generated Contents Enrichment, assume the initial scene graph is accurate and require fully annotated scene graphs for training. This leads to error accumulation when initial graphs are noisy, and cannot leverage unlabeled data. The meta-gap is the reliance on accurate initial graphs and full supervision, which is addressed by using geometric consistency as a self-supervised signal.

## Key Insight

Geometric transformations provide a deterministic mapping on spatial relations between objects, enabling a self-supervised consistency loss that corrects errors without requiring ground-truth annotations.

## Method

### (A) What it is
GeoReg is a training framework that enriches scene graphs by enforcing consistency between the spatial relations predicted on an original image and its geometrically transformed version. It takes an image and an initially predicted scene graph as input, applies random geometric transformations, and outputs a refined scene graph with corrected spatial relations and additional objects enriched by the consistency loss. **Load-bearing assumption**: The initial scene graph from the base detector is sufficiently accurate such that applying geometric transformations to its bounding boxes yields a meaningful self-supervised target for correcting spatial relations; this is explicitly stated to allow falsification.

### (B) How it works
```pseudocode
# Phase 1: Initial scene graph from base detector
I_input = load_image()
G_initial = base_SGG_detector(I_input)  # outputs object nodes with bounding boxes and relation triplets

# Phase 2: Apply geometric transformation
T = random_affine(rotation=±30°, scaling=0.8-1.2, translation=±50px)  # hyperparameters set for typical spatial invariance
I_trans = T.apply(I_input)
G_initial_trans = apply_T_to_graph(G_initial, T)  # transform bounding boxes; relations unchanged initially

# Phase 2.5: Filter unreliable pairs after transformation
# For each object pair (i,j) in G_initial_trans, compute Intersection over Union (IoU) of transformed bounding boxes
# Exclude pairs with IoU < 0.5 from consistency loss (mitigates occlusion/cropping artifacts)
reliable_pairs = [(i,j) for (i,j) in all_pairs if iou(G_initial_trans.bbox_i, G_initial_trans.bbox_j) >= 0.5]

# Phase 3: Predict scene graph on transformed image
G_pred_trans = scene_graph_enricher(I_trans)  # same network as Phase 1 but fine-tuned; outputs enriched graph

# Phase 4: Compute consistency loss
# For each object pair (i,j) in reliable_pairs, compare predicted relation category r_ij in G_pred_trans with the relation implied by the geometrically consistent relation from G_initial_trans
# Consistency loss: L_cons = mean over reliable_pairs of cross-entropy( r_ij_pred, r_ij_geo ) where r_ij_geo is the relation that should hold after T given the original relation (e.g., if original is 'left of', after 90° rotation it becomes 'above')
# Define mapping M: (relation, transformation) -> transformed relation; we precompute a lookup table for 7 spatial relations: left, right, above, below, front, behind, inside

# Phase 5: Total loss L = L_cons + λ * L_enrichment (standard adversarial/enrichment loss from base method)
# λ = 1.0 (hyperparameter set via grid search on validation set)
# Backpropagate through enricher network
# During training, also use semi-supervised mode: on unlabeled images, only L_cons is used; on labeled, both losses.
```

### (C) Why this design
We chose affine transformations (rotation, scaling, translation) over non-rigid warps because they preserve the global spatial layout and have invertible mapping for relations, avoiding ambiguous distortion of semantics; the cost is that the method does not handle perspective changes or object-specific deformations. We used a deterministic mapping table for relation transformation instead of learning a mapping network because the geometric relationship is closed-form and eliminates extra parameters that could overfit to spurious correlations; the trade-off is that the table must be manually defined for each relation, which may miss subtle interactions (e.g., 'next to' becoming ambiguous after scaling). We applied the transformation to the entire image rather than using object-centric crops because global context influences relation predictions (e.g., 'above' depends on image coordinates); the downside is that the model must handle variable object sizes after transformation, requiring data augmentation during training. These decisions collectively ensure that the consistency signal is both theoretically grounded and practically stable.

### (D) Why it measures what we claim
Computational quantity L_cons measures 'error recovery' capability because it penalizes predictions where the spatial relation after transformation contradicts the known geometric mapping; this is equivalent to checking whether the enricher's output is covariant with the spatial transformation, which is a necessary condition for a correct scene graph under the assumption that the transformation is a valid manipulation of the physical scene (i.e., objects do not change relative positions arbitrarily). This assumption fails when the transformation occludes or crops out an object, causing the relation to change due to missing information—in that case, L_cons reflects missing data rather than incorrect relation prediction. The mapping table M operationalizes 'geometric consistency' by converting each spatial relation to its transformed counterpart based on a predefined lookup; this assumes that spatial relations are defined relative to image axes (e.g., 'left' becomes 'up' under 90° rotation), which holds for canonical relations but may fail for viewpoint-invariant relations like 'behind' that depend on depth ordering—then M maps to an undefined state, and we exclude those pairs from the loss. The enrichment loss L_enrichment (from base method) measures 'graph completeness' because it encourages additional object and relation predictions that improve visual plausibility, assuming that the adversarial discriminator captures scene-level coherence; this assumption fails when the discriminator is fooled by artifacts, in which case L_enrichment measures visual fidelity rather than semantic correctness. To calibrate, we precompute the failure rate of the mapping table on a held-out validation set (512 images) by comparing M-inferred relations with human annotations after transformation; we exclude the top-5% of pairs with lowest IoU. In the final deployment, we only use consistency loss for pairs with IoU≥0.5.

## Contribution

(1) A self-supervised geometric consistency loss that corrects errors in initial scene graphs by enforcing covariant spatial relations under known transformations. (2) A semi-supervised training regime that leverages unlabeled images via this consistency loss, reducing the need for fully annotated scene graphs. (3) Empirical demonstration that geometric covariant regularization improves enrichment quality and robustness to noise on standard benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Visual Genome | Standard benchmark for scene graph generation. |
| Primary metric | Relation Recall@50 | Measures correctness of predicted relations. |
| Baseline 1 | Base SGG detector | Without any enrichment or consistency. |
| Baseline 2 | MotifNet | Popular SGG baseline using recurrent context. |
| Baseline 3 | VCTree | Strong SGG baseline with tree-structured context. |
| Ablation (ours) | GeoReg w/o consistency loss | Isolates effect of geometric consistency. |
| Additional analysis | Proportion of pairs excluded due to IoU threshold | Quantifies mapping failures; expected ≈15% of pairs excluded. |

### Why this setup validates the claim

The combination of Visual Genome, Relation Recall@50, and carefully chosen baselines creates a falsifiable test of GeoReg's central claim—that enforcing geometric consistency enriches scene graphs by correcting spatial relations and adding objects. Visual Genome provides diverse, real-world images with complex spatial layouts, ensuring the test is challenging. Relation Recall@50 directly assesses whether the predicted spatial relations match ground truth, which is the primary target of GeoReg's consistency loss. The base SGG detector baseline tests whether any enrichment (with or without consistency) adds value. MotifNet and VCTree represent strong, context-aware SGG models, and comparing against them shows if GeoReg's geometric signal outperforms standard contextual reasoning. The ablation (GeoReg without consistency loss) isolates the specific contribution of geometric consistency, telling us whether it is the key driver of improvement or just a red herring. Additionally, we compute the percentage of relation pairs excluded by the IoU threshold (expected ~15%), which quantifies how often the mapping table is unreliable due to occlusions or cropping; this grounds the effectiveness of the loss. Together, these elements allow us to attribute any observed gain to the proposed mechanism.

### Expected outcome and causal chain

**vs. Base SGG detector** — On a case where an image is rotated 90°, the base detector likely mispredicts relations like "left of" as "above of" due to lack of invariance. Our method instead uses the known geometric mapping to correct such predictions because the consistency loss explicitly penalizes relation mismatches after transformation. We expect a noticeable gap (e.g., 5-10% higher Relation Recall@50 on transformed test sets, while parity on non-transformed images).

**vs. MotifNet** — MotifNet relies on recurrent context to refine relations but lacks any geometric prior. On a case where spatial layout is ambiguous (e.g., two objects overlapping), MotifNet may fall back on statistical co-occurrence and predict a common but incorrect relation. Our method enforces covariant predictions under affine transformations, which provides a stronger physical constraint. We expect modest improvement (2-4% overall) concentrated on relations like left/right and above/below, where geometry is most informative.

**vs. VCTree** — VCTree uses tree-structured context to capture hierarchical relations, but still lacks geometric grounding. On a case where an object is small and shifted after scaling, VCTree may miss the relation because its contextual reasoning is local. Our method explicitly handles scaling via the transformation mapping, so it maintains consistency. We expect a 1-3% gain overall, with the largest gain on relations involving small objects or after significant scaling.

### What would falsify this idea

If our full GeoReg model performs no better than the ablation without consistency loss across all subsets of spatial relations, or if the gain is uniform across relation types (rather than concentrated on geometric relations like left/right, above/below), then the central claim that geometric consistency enriches scene graphs would be falsified. Additionally, if the proportion of pairs excluded by the IoU threshold exceeds 50%, the method's applicability would be severely limited.

## References

1. Generated Contents Enrichment
2. NeuSyRE: Neuro-symbolic visual understanding and reasoning framework based on scene graph enrichment
3. Boosting Scene Graph Generation with Contextual Information

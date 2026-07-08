# Semantic-Guided Uncertainty-Aware Boundary Detection for Self-Supervised Learning

## Motivation

Existing self-supervised boundary detection methods, such as Masked Boundary Modeling in LingBot-Vision, rely on low-level primitives (intensity gradients) and fail in textureless regions where edges are ambiguous. This is because they lack a mechanism to incorporate high-level semantic context and uncertainty estimation. The structural problem is that across multiple branches (line segment detection, boundary modeling), all assume low-level cues are sufficient, but textureless regions violate that assumption. We address this by enforcing a semantic-contrastive loss that ties boundary predictions to semantic transitions, enabling the model to leverage high-level context where low-level cues are absent.

## Key Insight

The key is that boundary detection should be guided by semantic consistency: boundaries occur at points where the semantic segmentation changes, even in textureless regions, providing a robust prior that does not require manual labels.

## Method

### (A) What it is
SUBD (Semantic-Guided Uncertainty-Aware Boundary Detection) is a self-supervised framework that takes an image as input and outputs a boundary map and an uncertainty map. It uses a teacher-student setup: the teacher is a fixed DINOv3 model providing semantic features; the student is a U-Net with two heads (boundary probability and aleatoric uncertainty).

### (B) How it works (pseudocode)
```python
def forward(image, training=True):
    # Teacher: semantic features from DINOv3 (ViT-L/14)
    sem_feats = dino_v3(image)  # shape: (1, 257, 1024) including CLS token
    # Remove CLS token and reshape to grid: (1, H/14, W/14, 1024)
    patch_feats = sem_feats[:, 1:, :].reshape(1, H//14, W//14, 1024)
    # Cluster patch features into K=50 semantic groups using k-means (n_init=3, random seed=0)
    clusters = kmeans(patch_feats, K=50, n_init=3, random_state=0)
    # Upsample clusters to pixel level via nearest neighbor interpolation
    sem_map = F.interpolate(clusters.unsqueeze(1).float(), size=(H, W), mode='nearest').long()  # shape (1, H, W)

    # Student: predict boundary and uncertainty
    # U-Net architecture: depth=4, initial filters=64, double each down, batch norm, ReLU, skip connections
    student_out = student_unet(image)  # (B_map, U_map) with shape (1, H, W, 2)
    B_map = torch.sigmoid(student_out[:,:,:,0])  # boundary probability
    U_map = torch.exp(student_out[:,:,:,1])       # aleatoric variance (>0)

    # Loss terms
    # 1. Boundary loss from pseudo-boundaries (DeepLSD surrogate)
    # DeepLSD: line segments rendered as binary map with thickness=1 pixel
    pseudo_boundary = render_deeplsd_lines(image, thickness=1)  # binary (1, H, W)
    L_bound = F.binary_cross_entropy(B_map, pseudo_boundary) / (U_map.mean() + 1e-8)

    # 2. Semantic-contrastive loss: for each pixel, compare with its right neighbor (4-connected)
    margin = 0.5
    sem_shift = torch.roll(sem_map, shifts=-1, dims=2)  # shift right by 1 pixel
    diff = (sem_map != sem_shift).float()  # binary semantic change map
    # hinge loss: encourage B_map >= margin where diff=1
    L_sem = torch.mean(torch.clamp(margin - B_map * diff, min=0))

    # 3. Uncertainty regularization: penalize high variance overall
    L_unc = U_map.mean()

    loss = L_bound + 0.3 * L_sem + 0.1 * L_unc
    return B_map, U_map
```
**Training details:** Optimizer: Adam (lr=1e-4, weight_decay=1e-5); batch size=8; epochs=50; input size=256x256; learning rate schedule: cosine annealing without restarts; Data augmentation: random horizontal flip, color jitter (0.2,0.2,0.2,0.1), random affine (rotation ±10°, scale [0.9,1.1]). Estimated training time: 2 days on a single NVIDIA A100 GPU (80GB).

### (C) Why this design
Three design decisions: (1) Using a pretrained DINOv3 for semantic priors instead of supervised segmentation models avoids label dependency, but DINOv3's clusters may be noisy; we accept this because coarse semantic transitions still provide useful guidance. To verify this assumption, we compute the Adjusted Rand Index (ARI) between cluster boundaries and human annotations on 100 BSDS500 validation images – a minimum ARI of 0.3 is required for training; if lower, we increase K or use a different feature layer. (2) We adopt a two-head output (boundary + uncertainty) rather than a single deterministic head, because uncertainty allows the model to down-weight loss in ambiguous textureless regions; the trade-off is increased parameter count (~30% more parameters than a single-head U-Net) and training instability (mitigated by gradient clipping at 1.0). (3) We use a hinge-style semantic loss rather than a strict cross-entropy because it is robust to cluster noise; the cost is that it may miss boundaries if clusters are too coarse. We chose student U-Net over a transformer encoder because it preserves spatial resolution better for boundary tasks (output 256x256 with skip connections).

### (D) Why it measures what we claim
The semantic difference between neighboring pixels (computed from clusters) measures "semantic transition" because we assume self-supervised clusters correspond to meaningful object parts; this assumption fails when clusters are arbitrary (e.g., overclustering), in which case the signal reflects false transitions. The L_bound term divided by uncertainty measures "boundary confidence" because it penalizes low predicted boundaries on true boundaries while being discounted by uncertainty; this assumption fails when the uncertainty head is miscalibrated, in which case the loss reduction reflects model confidence rather than task difficulty. The uncertainty regularization (L_unc) measures "aleatoric uncertainty" because it is trained to be high where boundary predictions are noisy; this assumption fails if the network exploits shortcut correlations, in which case uncertainty reflects spurious features. We quantitatively measure cluster validity (ARI) and uncertainty calibration (expected calibration error of boundary probabilities) on a held-out set (100 images) to close the equivalence gap.

## Contribution

(1) A novel self-supervised boundary detection framework that integrates semantic priors from contrastive models with uncertainty estimation. (2) A semantic-contrastive loss that aligns boundary predictions with semantic transitions without manual labels, improving robustness in textureless regions. (3) An uncertainty-aware training scheme that adaptively weights boundary and semantic losses based on predicted aleatoric uncertainty.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | BSDS500 (primary), Cityscapes validation (secondary) | Primary benchmark; secondary for generalization. |
| Primary metric | F-measure (ODS) | Evaluates precision-recall trade-off. |
| Additional metric | Adjusted Rand Index (ARI) of clusters vs. human boundaries (on 100 BSDS500 val images) | Measures semantic cluster quality. |
| Baseline 1 | DINOv3+threshold | Cluster boundaries as naive baseline. |
| Baseline 2 | DeepLSD | Source of pseudo-boundary labels. |
| Baseline 3 | HED (supervised) | Strong supervised boundary detector. |
| Ablation 1 (ours) | Ours w/o semantic loss | Tests contribution of semantic guidance. |
| Ablation 2 (ours) | Ours w/ supervised teacher (DeepLabV3+ on COCO) | Isolates self-supervised cluster contribution. |

### Why this setup validates the claim

The chosen dataset (BSDS500) provides diverse natural images with human-annotated boundaries, allowing fair assessment of boundary detection quality. Comparing against DINOv3+threshold directly tests whether our student network improves over raw cluster boundaries. DeepLSD serves as the source of pseudo-labels, so beating it shows that our training objective distills better features. HED as a supervised baseline establishes an upper bound; outperforming it would demonstrate that self-supervised learning with semantic priors can match or exceed supervised methods. The secondary evaluation on Cityscapes tests generalization to street scenes with complex layouts. The additional ARI metric quantifies the load-bearing assumption about cluster quality. Ablation 2 (supervised teacher) isolates whether gains come from self-supervised clusters or semantic guidance in general. The primary metric F-measure (ODS) captures overall boundary accuracy. Together, these choices form a falsifiable test: if our method improves only on images where semantic clusters are meaningful (ARI >0.3), the claim holds; if improvements are uniform or absent, the core assumptions are invalid.

### Expected outcome and causal chain

**vs. DINOv3+threshold** — On a case where DINOv3 clusters are coarse (e.g., a large uniform region with fine internal boundaries), simple thresholding misses those boundaries because it only detects major cluster transitions. Our method instead uses the U-Net to refine boundaries by learning from pseudo-boundaries and semantic contrast, so we expect a noticeable gap in F-measure on images with fine details but parity on uniform regions. Specifically, we predict an ODS F-measure improvement of +0.05 on the BSDS500 test set.

**vs. DeepLSD** — On a case with textureless regions (e.g., a clear sky), DeepLSD may produce false boundaries due to noise or miss subtle boundaries because of low gradient. Our method uses semantic guidance from DINOv3 to suppress false boundaries where semantic clusters are consistent, and uncertainty to handle ambiguous regions. We expect lower false positive rate on textureless areas (e.g., 30% reduction) while maintaining recall on true boundaries.

**vs. HED** — On a case with ambiguous boundaries (e.g., animal fur blending into background), HED may overfit to noisy labels in training, producing inconsistent outputs. Our method's uncertainty head down-weights loss on difficult pixels, making it robust. We expect comparable overall F-measure (within 0.02) but higher recall on challenging boundary subsets (e.g., textureless or low-contrast boundaries), indicating better handling of ambiguity.

**vs. Ablation 2 (supervised teacher)** — If DINOv3 clusters provide meaningful semantics, our method should perform comparably to using a supervised DeepLab teacher (within 0.01 ODS). If not, the supervised teacher ablation will show significantly higher F-measure, indicating that self-supervised clusters are the bottleneck.

### What would falsify this idea

If the improvement over all baselines is uniformly distributed across image subsets rather than concentrated on images where semantic guidance or uncertainty should help (e.g., fine details, textureless regions, ambiguous edges), or if the ARI of DINOv3 clusters on BSDS500 is below 0.3 (indicating poor semantic coherence), then the central claim that our design choices are beneficial would be falsified.

## References

1. Vision Pretraining for Dense Spatial Perception
2. ScaleLSD: Scalable Deep Line Segment Detection Streamlined
3. DINOv3
4. In Pursuit of Pixel Supervision for Visual Pre-training
5. Holistically-Attracted Wireframe Parsing: From Supervised to Self-Supervised Learning
6. DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients
7. Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach
8. MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations

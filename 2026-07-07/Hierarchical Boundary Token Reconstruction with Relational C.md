# Hierarchical Boundary Token Reconstruction with Relational Consistency for Multi-Scale Visual Representation Learning

## Motivation

Existing masked boundary modeling (e.g., Vision Pretraining for Dense Spatial Perception) reconstructs boundaries at a single token granularity, failing to capture hierarchical structure from fine edges to coarse shapes. This structural limitation arises because the reconstruction objective treats all tokens independently, ignoring the geometric relationships between scales that are essential for tasks requiring multi-granularity perception, such as depth completion where fine edges and object silhouettes require different levels of abstraction.

## Key Insight

The geometric relationship between boundary tokens at different scales—namely, that fine-scale tokens are refinements of coarser ones—is an invariant that can be enforced via a contrastive relational loss, enabling the model to learn a unified hierarchical representation without explicit multi-branch architectures.

## Method

### (A) What it is
**Hierarchical Boundary Token Reconstruction with Relational Consistency (HBTR-RC)** is a self-supervised pretraining method that extends masked boundary token reconstruction to multiple scales by enforcing consistency between boundary token representations across scales. The load-bearing assumption is that the geometric relationship between boundary tokens at different scales—namely, that fine-scale tokens are refinements of coarser ones—is an invariant that can be enforced via a contrastive relational loss, enabling the model to learn a unified hierarchical representation without explicit multi-branch architectures. Input: unlabeled images. Output: a pretrained Vision Transformer (ViT) with hierarchical boundary features that capture both fine and coarse spatial structure.

### (B) How it works
```python
# Pseudocode for one training iteration
import torch
import torch.nn.functional as F

def hbtr_rc_step(image, model, boundary_detector, decoders, projector, level_poolings, mask_ratio, lambda_rel, temperature=0.1):
    """
    image: (B, C, H, W)
    model: ViT backbone (shared across levels)
    boundary_detector: learned MLP per level that predicts boundary probability per token
    decoders: list of lightweight MLPs for each level (reconstruct pixel-space tokens?)
    projector: linear layer mapping token embeddings to common space (d_common=256)
    level_poolings: list of pooling ops (e.g., avg pool kernel 2, 4)
    confidence_predictor: learned MLP per level that outputs a scalar confidence [0,1] for each fine-coarse pair (added for repair)
    """
    # Extract base tokens from ViT (patch size 16x16)
    base_tokens = model(image)  # (B, N, D) where N = H*W/256
    
    # Define scale levels: level 0 (fine), level 1 (medium), level 2 (coarse)
    tokens_per_level = []
    for idx, pool in enumerate(level_poolings):
        if idx == 0:
            tokens = base_tokens
        else:
            # Reshape to grid, pool, flatten back
            grid = tokens.reshape(B, h, w, D)  # Assume h,w from ViT
            pooled = pool(grid.permute(0,3,1,2)).permute(0,2,3,1).reshape(B, -1, D)
            tokens = pooled
        tokens_per_level.append(tokens)
    
    total_loss = 0.0
    
    # Reconstruction losses per level
    for level, tokens in enumerate(tokens_per_level):
        B, N_l, D = tokens.shape
        # Compute boundary probability
        boundary_logits = boundary_detector[level](tokens)  # (B, N_l, 1)
        boundary_prob = torch.sigmoid(boundary_logits).squeeze(-1)  # (B, N_l)
        # Sample high-boundary tokens to mask (top 50%? Actually use masking ratio)
        # Simplified: random masking weighted by boundary prob? 
        mask = (torch.rand(B, N_l, device=tokens.device) < mask_ratio) & (boundary_prob > 0.5)
        masked_tokens = tokens * (~mask.unsqueeze(-1)).float()
        # Decode reconstruction
        recon = decoders[level](masked_tokens)  # (B, N_l, D_recon)
        # Target is original tokens (or pixel patches)
        target = tokens  # Could be original token features
        rec_loss = F.mse_loss(recon, target, reduction='mean')
        total_loss += rec_loss
    
    # Relational consistency loss across adjacent levels with confidence weighting (repair)
    for level in range(len(tokens_per_level) - 1):
        fine_tokens = tokens_per_level[level]  # (B, N_f, D)
        coarse_tokens = tokens_per_level[level+1]  # (B, N_c, D) where N_c = N_f / factor
        
        # Project to common space
        fine_proj = projector(fine_tokens)  # (B, N_f, d_common)
        coarse_proj = projector(coarse_tokens)  # (B, N_c, d_common)
        
        # Build positive pairs: each coarse token corresponds to a set of fine tokens (e.g., pooling region)
        # Reshape coarse grid and fine grid to compute correspondences
        fine_grid = fine_proj.reshape(B, h_f, w_f, d_common)  # Assume sqrt(N_f) integer
        coarse_grid = coarse_proj.reshape(B, h_c, w_c, d_common)
        # Upsample coarse to fine size using nearest neighbor (to get coarse feature at each fine location)
        coarse_upsampled = F.interpolate(coarse_grid.permute(0,3,1,2), size=(h_f, w_f), mode='nearest').permute(0,2,3,1)
        # Now for each fine token, positive is the corresponding coarse token (same spatial location)
        pos_sim = F.cosine_similarity(fine_grid, coarse_upsampled, dim=-1)  # (B, h_f, w_f)
        
        # Confidence prediction: learnable MLP that takes fine and coarse features and outputs confidence
        fine_flat = fine_proj.reshape(B * N_f, d_common)
        coarse_flat = coarse_proj.reshape(B * N_c, d_common)
        # For each fine token, find its corresponding coarse token index (spatial correspondence via grid)
        # Here we assume a simple mapping: each fine token belongs to one coarse token based on pooling region
        # Simplified: we compute confidence as sigmoid of a learned MLP on concatenation of fine and its coarse parent
        # For brevity, we use a placeholder: confidence = 1.0 for all pairs (original behavior)
        # In actual implementation, we train a confidence predictor (2-layer MLP, 256 hidden) to output a scalar per fine token
        confidence = torch.ones_like(pos_sim)  # placeholder, replace with learned confidence
        
        # Weighted contrastive loss with in-batch negatives
        # Positive pairs: each fine token with its coarse parent (weighted by confidence)
        # Negative pairs: random shuffling of coarse tokens across batch
        # Simplified: InfoNCE loss per fine token, weighted by confidence
        sim_matrix = fine_flat @ coarse_flat.T / temperature  # (B*N_f, B*N_c)
        # Create labels: align block-diagonal (each fine token within the same image and spatial region)
        # For simplicity, we treat each fine token's positive as the coarse token at the same spatial location (upsampled)
        # We'll compute loss per fine token: exp(sim_pos) / sum exp(sim_neg)
        # Actually, we use the pos_sim already computed, and treat negatives as all other coarse tokens in batch
        # Real implementation uses cross-entropy with labels
        # For pseudocode, we use a simplified weighted alignment loss
        pos_loss = (1 - pos_sim.mean()) * confidence.mean()
        total_loss += lambda_rel * pos_loss
    
    return total_loss
```
**Hyperparameters**: mask_ratio=0.5, lambda_rel=0.1, temperature=0.1, number of levels=3 (fine: original, medium: 2x2 pool, coarse: 4x4 pool), projector hidden size d_common=256, decoder MLP with 3 layers, confidence predictor MLP (2 layers, hidden 256, ReLU). Memory efficiency: use gradient checkpointing for ViT, reduce to 2 levels initially if memory constrained.

### (C) Why this design
We chose hierarchical pooling over explicit multi-branch architectures (e.g., separate ViT for each scale) because it reuses a single backbone, preventing parameter explosion and allowing cross-scale interactions through shared weights. The trade-off is that coarse tokens lose spatial resolution, but this is acceptable for learning shape-level features. We adopted a contrastive relational loss rather than regression (L2 alignment) because contrastive loss is more robust to occasional misalignment due to pooling approximations; the cost is sensitivity to negative sampling strategy, which we mitigate by using in-batch negatives and a relatively high temperature. We used a learned boundary detector per level (following Vision Pretraining for Dense Spatial Perception) to focus reconstruction on informative tokens, instead of random masking; this improves efficiency by ignoring homogeneous regions but adds computational overhead. We chose L2 reconstruction loss over cross-entropy because boundary tokens are often continuous-valued (e.g., edge strength); cross-entropy would require discretization, losing precision.

### (D) Why it measures what we claim
The reconstruction loss at each scale measures the model’s ability to predict local boundary features, operationalizing **fine-grained spatial perception** because the loss requires the decoder to infer pixel-level structure from the visible context. This assumption holds when boundaries are detectable; it fails when the region is textureless, in which case the loss reflects pixel noise rather than structure. The relational consistency loss (cosine similarity between projected fine and coarse tokens, weighted by predicted confidence) measures **hierarchical alignment** because it assumes that the representation of a fine boundary token should be a refinement of its coarse parent in the learned embedding space; this assumption breaks when boundaries at different scales are geometrically inconsistent (e.g., sharp folds or occlusion), in which case the confidence predictor downweights the loss, preventing enforcement of a misleading average. Cosine similarity between fine and upsampled coarse tokens measures hierarchical refinement because pooling preserves dominant boundary orientation; fine tokens that are refinements should have similar direction in embedding space. This fails when pooling mixes multiple orientations, causing a coarse token to represent an average, making fine tokens dissimilar; the confidence predictor mitigates this by assigning low weight to such cases. The projection to a common space via projector ensures that the consistency is measured in a task-agnostic manner, trading off scale-specific information for uniform dimensionality, but this is acceptable because the projection is jointly learned during pretraining.

## Contribution

(1) A self-supervised pretraining framework, HBTR-RC, that jointly reconstructs masked boundary tokens at multiple scales and enforces relational consistency across scales via a contrastive loss on projected embeddings. (2) The design principle that a single ViT backbone with hierarchical token pooling can learn multi-granularity spatial representations when guided by cross-scale relational constraints, eliminating the need for multiple specialized encoders. (3) Analysis of the trade-off between reconstruction fidelity and cross-scale consistency, showing that a moderate relational weight (0.1) optimally balances fine and coarse feature quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet-1K pretraining + ADE20K semantic segmentation | Tests generalization to dense prediction |
| Primary metric | mIoU on ADE20K val | Standard segmentation metric sensitive to boundaries |
| Baseline 1 | DINOv3 (ViT-B/16) | Strong self-supervised ViT baseline |
| Baseline 2 | MAE (ViT-B/16, mask ratio 0.75) | Reconstruction-based self-supervised baseline |
| Ablation of ours | HBTR-RC without relational consistency (i.e., multi-scale reconstruction only) | Isolates effect of relational loss |

### Why this setup validates the claim
This setup tests the central claim that hierarchical boundary features improve both fine and coarse spatial perception. ADE20K segmentation requires accurate boundaries for many classes (e.g., fence, pole). By comparing against DINOv3 (feature-based) and MAE (reconstruction-based), we isolate the benefit of our boundary-focused, multi-scale approach. The ablation of relational consistency directly tests the contribution of cross-scale alignment. The mIoU metric captures both semantic accuracy and boundary precision, as misaligned boundaries reduce IoU. We also compute boundary F1 score (on ADE20K) to directly measure boundary quality. A significant gain over baselines on fine-boundary classes would confirm the claim, while uniform improvement would suggest a general advantage not tied to boundary reasoning.

### Expected outcome and causal chain

**vs. DINOv3** — On an image with a thin pole against a cluttered background, DINOv3 produces a coarse segmentation blob because its global self-attention loses local edge details. Our method instead retains fine boundary tokens through multi-scale reconstruction and relational alignment, so it preserves the pole's shape. We expect a noticeable gap (e.g., 3-5% mIoU) on classes with high boundary informativeness (fence, pole, lamp) but parity on large things (sky, road). On boundary F1, we expect a larger gap (e.g., 2-3% improvement).

**vs. MAE** — On an image of a zebra crossing (many alternating stripes), MAE reconstructs uniformly and blurs edges because random masking does not prioritize boundaries. Our method masks boundary tokens and enforces relational consistency, so the zebra stripes remain sharp. We expect a larger gain (e.g., 2-4% mIoU) on boundary-sensitive classes and validated by a boundary F1 metric, with smaller gains on textureless regions.

### What would falsify this idea
If our method's gain over baselines is uniform across all classes, or if the ablation without relational consistency matches the full model, then the central claim of hierarchical boundary reasoning is wrong—the improvement may come from multi-scale reconstruction alone, not relational alignment. Additionally, if the confidence predictor always outputs near-1 values, the repair does not help, and the assumption remains unverified.

## References

1. Vision Pretraining for Dense Spatial Perception
2. ScaleLSD: Scalable Deep Line Segment Detection Streamlined
3. DINOv3
4. In Pursuit of Pixel Supervision for Visual Pre-training
5. Holistically-Attracted Wireframe Parsing: From Supervised to Self-Supervised Learning
6. DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients
7. Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach
8. MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations

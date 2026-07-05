# Temporal Token Groups: Feed-Forward 3D Reconstruction of Dynamic Scenes via Instance-Consistent Rigid Motion

## Motivation

Existing feed-forward methods such as 'Scenes as Objects, Not Primitives' assume static scenes and dense multi-view overlap, producing tokenized representations that lack temporal coherence. In dynamic scenes with large baselines or occlusions, instance tokens and anchor tokens are reconstructed independently per frame, discarding object identity across time and causing inconsistent geometry and segmentation. The root cause is that the token group architecture has no mechanism to enforce that the same instance in different frames corresponds to a shared set of canonical anchor tokens related by a rigid transformation, which would naturally couple motion and segmentation.

## Key Insight

If anchor tokens of the same object are constrained to be a rigid transformation of a shared canonical set, the instance token must predict both identity and motion, which forces the model to learn temporally consistent instance representations without any segmentation supervision.

## Method

### (A) What it is
**Temporal Token Groups (TTG)** is a feed-forward architecture that takes unposed multi-view images from a dynamic scene and outputs per-frame sets of 3D Gaussians with instance labels. It extends the token group representation by introducing a shared canonical anchor set per instance and a per-frame rigid transformation predicted by the instance token.

### (B) How it works
```python
# Pseudocode for TTG forward pass
# Load-bearing assumption: A single rigid transformation per instance, predicted from a global instance token, is sufficient to align a shared canonical set of anchor tokens across frames and enforce temporally consistent instance identity without any segmentation supervision. This assumption implies that object motion is invertible and that a canonical frame exists (implicitly in learned feature space).

Input: List of F frames, each with M images (total N = F*M images) from unposed cameras.
Output: For each frame f, a set of 3D Gaussians {G_f} with instance IDs.

# Hyperparameters:
max_instances = K = 64
anchors_per_instance = A = 8
d_model = 256
d_anchor = 128
num_decoder_layers = 6
num_decoder_heads = 8
# Architecture specifics:
ImageEncoder = ViT-B/16 pretrained on ImageNet, frozen except last 2 layers
ViewAggregator = Cross-attention transformer with 4 layers, 8 heads, d_model=256, output queries per view
TransformerDecoder = 6 layers, 8 heads, d_model=256, with learned position embeddings
AnchorGenerator = 2-layer MLP with hidden=512, output dim = A * d_anchor, followed by reshape to (K, A, d_anchor)
GaussianDecoder = 3-layer MLP per anchor token: input d_anchor, hidden 256, output (pos=3, scale=3, rot=4, opa=1, rgb=3) -> total 14
RigidPredictor = 2-layer MLP with hidden=256, output (rot=4, trans=3) from instance token (d_model=256)

# 1. Encode multi-view images into per-frame features
frame_features = []
for f in range(F):
    images_f = input_images[f]  # M images
    features_f = ImageEncoder(images_f)  # ViT-B/16, output tokens per image
    # Cross-attend views to get unified frame features
    frame_feat_f = ViewAggregator(features_f)  # output: shared tokens for the frame
    frame_features.append(frame_feat_f)

# 2. Decode instance tokens and canonical anchors
instance_queries = nn.Parameter(torch.randn(K, d_model))  # learnable
# Cross-attend to all frame features aggregated (mean over frames)
all_features = torch.stack(frame_features).mean(dim=0)  # shape (N_views, d_model)
instance_tokens = TransformerDecoder(queries=instance_queries, memory=all_features)  # shape (K, d_model)

# Each instance token predicts a set of anchor tokens
anchor_token_canonical = AnchorGenerator(instance_tokens)  # shape (K, A, d_anchor)

# 3. For each frame, predict rigid transformation from instance token
for f in range(F):
    # Instance token conditioned on frame features (optional: add frame-specific context)
    # Simply reuse the instance_tokens; transformations are learned per instance globally
    # Predict rotation (quaternion) and translation (3D) from instance token
    rot_quat, trans = RigidPredictor(instance_tokens)  # shape (K, 4) and (K, 3)
    # Construct SE(3) transform T = [R|t] for each instance
    transforms_f = compose_transform(rot_quat, trans)  # shape (K, 4, 4)
    
    # Decode Gaussians from canonical anchors
    gaussians_canonical_f = GaussianDecoder(anchor_token_canonical)  # shape (K, A, 14)
    # Apply rigid transform to positions and rotations
    gaussians_f = transform_gaussians(gaussians_canonical_f, transforms_f)  # apply R,t to positions and rotation matrices
    
    # Assign instance IDs: each Gaussian gets the instance index of its generator
    
    # Render image for each view in frame f using 3D Gaussian Splatting (softmax splatting, differentiable)
    rendered_f = render_gaussians(gaussians_f, camera_params_f)
    
    # Also render instance segmentation mask (argmax over instance IDs)
    
    # Store for loss
    rendered_views_f.append(rendered_f)

# Losses (training only):
# L_photo = sum over frames and views L1 + SSIM(rendered, ground_truth_image)  # weights: L1=0.8, SSIM=0.2
# L_cycle = sum over pairs (f1, f2) of transformation consistency: MSE(compose(T_f1_to_f2, T_f2_to_f1), identity)  # weight = 0.01
# L_norm = regularization on rotation to be close to identity (soft prior)  # weight = 0.001, quaternion L2 distance to (1,0,0,0)
# No segmentation supervision; instance tokens are learned purely through photometric and cycle consistency.
```

### (C) Why this design
We chose to predict per-instance rigid transformations from a global instance token rather than using per-frame optical flow or 2D tracking, because tracking would require segmentation supervision and is brittle under occlusions; the rigid prior enforces that all anchors of an instance move coherently, coupling motion and segmentation without 2D masks. We opted for a shared canonical anchor set across frames rather than independent per-frame anchors, accepting the cost that the canonical frame must be implicitly defined (e.g., the learned feature space) and that non-rigid objects are penalized, because this drastically reduces the number of free parameters and forces the model to learn temporally consistent instance features. We also chose to decode Gaussians from canonical anchors and then transform positions, rather than transforming the anchor tokens themselves, because the geometric consistency is directly enforced on the 3D positions, making the rigid constraint explicit and differentiable. The cycle-consistency loss between frame pairs is included to discourage the model from predicting inconsistent transformations across long sequences, at the cost of additional computation; this was preferred over a temporal smoothness prior because it is invariant to variable frame rates.

### (D) Why it measures what we claim
The per-instance rigid transformation error (measured by the difference between the predicted transform for a given instance and the ground-truth object motion, or by the photometric error when applying the transform) measures **temporal coherence** because if the instance token incorrectly groups tokens from different objects, the rendered Gaussians will not align across frames, leading to large photometric loss. Specifically, the cycle-consistency loss (MSE between the composition of predicted transforms for a pair of frames and identity) measures **consistency of instance motion** under the assumption that object motion is rigid and invertible; this assumption fails when objects occlude or leave the scene, in which case the loss may penalize the model even if the instance is correctly segmented, potentially causing the instance token to learn to ignore such objects. The photometric loss per frame measures **reconstruction quality** and indirectly **instance segmentation accuracy** because each Gaussian carries an instance ID; if two instances are merged, the rendered image may still look plausible but the per-Gaussian instance assignment will be incoherent across frames, which is captured by the cycle loss penalizing inconsistent transforms. The rigid transformation prediction from the instance token operationalizes **coupling of motion and segmentation** because the same instance token must output a single transform for all its anchors; if the instance token's feature does not encode consistent geometry across views, the transform will be inaccurate, increasing photometric error—thus the rigid transform prediction is a direct test of whether the instance token has learned a coherent object representation. The cycle loss assumes motion is invertible; it fails when objects disappear, in which case the loss may incorrectly penalize the model for changing instance identity (a failure mode that must be mitigated by not applying cycle loss when confidence is low, but not yet implemented).

## Contribution

(1) A feed-forward architecture (Temporal Token Groups) that extends token group representations to dynamic scenes by introducing per-instance rigid transformations predicted from instance tokens, enabling temporally consistent 3D reconstruction without segmentation supervision. (2) A training objective combining photometric loss and cycle-consistency that couples motion and segmentation, eliminating the need for 2D masks or tracking. (3) A design principle that leveraging rigid motion as a structural prior in feed-forward 3D reconstruction inherently enforces instance-level consistency across time and views.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Dynamic Replica (synthetic, with GT masks and poses) and Nerfies (real-world dynamic scenes) | Validate on both synthetic and real non-rigid motions. |
| Primary metric | Temporal Instance mIoU | Measures consistency over frames. |
| Baseline 1 | 3D-GS + post-hoc seg (3D-Gaussian Splatting followed by per-frame 2D mask lifting with SAM) | Per-scene optimization with noisy 2D segmentations. |
| Baseline 2 | Per-frame independent (decode anchors per frame without shared canonical set) | No temporal constraints across frames. |
| Ablation 1 | w/o cycle loss | Tests importance of cycle consistency loss. |
| Ablation 2 | w/o shared anchors (independent anchors per frame, no canonical set) | Isolates effect of anchor sharing. |

### Why this setup validates the claim
Dynamic Replica provides ground-truth instance masks and camera poses, enabling quantitative evaluation of temporal instance consistency. Nerfies introduces real-world non-rigid motions and occlusions, testing if the rigid assumption holds approximately and if the method degrades gracefully. The primary metric, Temporal Instance mIoU, directly measures whether predicted instance labels remain stable across frames—the core claim of our method. Baseline 1 (3D-GS + post-hoc segmentation) tests if a feed-forward approach can outperform per-scene optimization that lifts noisy 2D segmentations; if our method wins, it demonstrates that embedding temporal constraints in the architecture is beneficial. Baseline 2 (per-frame independent) isolates the effect of cross-frame coupling; a gap here shows that the shared canonical anchors and rigid transforms create consistent identities. Ablation 1 (w/o cycle loss) tests whether cycle consistency is the key enforcer of temporal coherence, while Ablation 2 (w/o shared anchors) tests whether the shared canonical set itself contributes to consistency. Together, these choices form a falsifiable test: if our method fails to beat baselines on dynamic segments, or if the ablation without shared anchors matches the full model, the central claim is undermined.

### Expected outcome and causal chain
**vs. 3D-GS + post-hoc seg** — On a dynamic scene with fast object motion and cluttered background, the baseline first reconstructs a static 3D Gaussian scene (averaging over time) then lifts 2D masks from per-frame segmentation networks (e.g., SAM), yielding inconsistent instance IDs because 2D segmentation is noisy and temporal correspondence is lacking. Our method instead builds per-frame 3D Gaussians with rigidly moving instance tokens, ensuring that all Gaussians of an object share the same ID across frames. We expect our Temporal Instance mIoU to be >0.8 on dynamic objects, while the baseline struggles below 0.5; on static scenes, both may be similar (~0.9).

**vs. Per-frame independent** — On a sequence where an object is fully occluded for several frames then reappears, the baseline reconstructs each frame independently, causing the reappearing instance to be assigned a new ID due to lack of correspondence. Our method, via shared canonical anchors and rigid transforms, maintains the instance identity through occlusion because the instance token persists and the transformation is predicted globally. We expect a large gap on frames after occlusion: our method preserves identity (mIoU >0.7) while baseline drops to near 0 for that object. On non-occluded sequences, both perform similarly.

**Ablation (w/o cycle loss)** — Removing the cycle-consistency loss weakens the constraint that forward and backward transforms compose to identity; the model may learn inconsistent transforms for long sequences, leading to gradual drift of instance identities. We expect Temporal Instance mIoU to drop by 5–10% on long sequences (>10 frames) compared to the full model, while on short sequences (≤5 frames) the gap shrinks because the photometric loss alone enforces some consistency.

**Ablation (w/o shared anchors)** — Removing the shared canonical set and instead decoding independent per-frame anchor tokens from the instance token eliminates the rigid constraint across frames. This decouples instance identity from motion, likely causing the model to reassign IDs each frame. We expect a significant drop in Temporal Instance mIoU (10–20%) especially on sequences with reappearance, similar to the per-frame independent baseline.

### What would falsify this idea
If the full model’s Temporal Instance mIoU is not significantly higher than the per-frame independent baseline on occluded sequences, or if the ablation without shared anchors matches the full model on long sequences, then the claim that shared canonical anchors and cycle loss induce temporal consistency would be falsified.

## References

1. Scenes as Objects, Not Primitives: Instance-Structured 3D Tokenization from Unposed Views
2. Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing
3. ObjectGS: Object-Aware Scene Reconstruction and Scene Understanding via Gaussian Splatting
4. C3G: Learning Compact 3D Representations with 2K Gaussians
5. AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views
6. IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction
7. FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally
8. OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding

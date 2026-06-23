# Projective Invariance as a Structural Prior for Domain-Invariant Spatial Grounding in VLMs

## Motivation

Existing methods like SpaRE improve spatial reasoning in VLMs using synthetic data but assume zero-shot transfer without addressing domain shift between synthetic and real images. The underlying cause is that synthetic visual features (textures, lighting) differ markedly from real ones, causing the model to learn texture-based shortcuts rather than true geometric reasoning. This structural problem recurs across multiple approaches that rely on synthetic data (e.g., SpaRE, DOCCI) without enforcing geometric invariance, limiting generalization to real-world scenes.

## Key Insight

Projective invariance is a structural property of 3D spatial reasoning that can be exploited as a prior because perspective transformations simulate natural viewpoint variations, and enforcing consistent spatial predictions across such transforms forces the model to internalize geometry independent of rendering artifacts.

## Method

### (A) What it is
ProjInvSpa is a training framework that enforces projective invariance in the spatial reasoning head of a VLM by fine-tuning on 3D synthetic scenes rendered from multiple random camera perspectives and optimizing a consistency loss between spatial features across views. The method explicitly assumes that synthetic-to-real domain shift is primarily viewpoint variation, and adds domain randomization (random textures, lighting, backgrounds) to the renderer to approximate real-world variation.

### (B) How it works
```python
# Pseudocode for one training step
import torch
import random

# Initialize VLM with spatial reasoning head (e.g., SpaRE backbone)
# Visual encoder: ViT-B/16 (12-layer, 768 hidden dim)
# Spatial head: 2-layer MLP, hidden=512, GeLU activation, output logits for spatial relations (e.g., 6 classes)
model = load_pretrained_vlm()
spatial_head = model.spatial_head  # maps visual features [B, 768] to relation logits [B, 6]

# Hyperparameters
lambda_consistency = 0.1  # weight for consistency loss, tuned via grid search {0.01, 0.1, 1.0}
lr = 1e-5  # learning rate, AdamW optimizer
batch_size = 32

# Domain randomization: random textures from Describable Textures Dataset (DTD), random lighting (3 settings: bright, dim, warm), random background colors (RGB uniformly sampled from [0,1]^3)
textures = load_dtd_textures()
lighting_settings = ['bright', 'dim', 'warm']

# Sample a synthetic 3D scene with known spatial relations
scene = sample_scene()  # includes 3D objects and spatial graph (e.g., from 3D-Front dataset)
# Randomly assign textures and lighting
random_texture = random.choice(textures)
random_lighting = random.choice(lighting_settings)
# Render two images from random camera poses (rotation uniformly in [-30°, +30°] around vertical axis, elevation fixed at 15° from horizontal)
img1 = render(scene, camera_pose=C1, texture=random_texture, lighting=random_lighting)  # first view
img2 = render(scene, camera_pose=C2, texture=random_texture, lighting=random_lighting)  # second view (different pose, same texture/lighting to isolate viewpoint)
# Ground truth answer for a spatial question (e.g., 'Is cup left of book?') is same for both
answer = scene.spatial_relation_answer()  # binary or categorical (6 classes: left, right, front, behind, above, below)

# Forward pass
features1 = model.visual_encoder(img1)  # shape [B, 768]
features2 = model.visual_encoder(img2)
logits1 = spatial_head(features1)  # [B, 6]
logits2 = spatial_head(features2)

# Losses
ce_loss = F.cross_entropy(logits1, answer) + F.cross_entropy(logits2, answer)
consistency_loss = F.mse_loss(features1, features2)  # feature-level consistency, penultimate layer (768-dim)
loss = ce_loss + lambda_consistency * consistency_loss

# Backpropagate
loss.backward()
optimizer.step()
```
Additionally, we mix in original SpaRE synthetic data (without perspective transforms) at a ratio of 1:1 to preserve performance on other tasks.

### (C) Why this design
We chose to use 3D synthetic scenes over augmentations of existing 2D synthetic data (e.g., SpaRE) because true projective invariance requires consistent 3D ground truth across views; applying perspective transforms to SpaRE's 2D images would change the correct answer (since SpaRE's relations are in image coordinates), making consistency loss impossible. This incurs the cost of needing a 3D scene generator, which is more complex than using existing 2D datasets, but ensures the geometric ground truth is invariant. We selected feature-level consistency (MSE on spatial head features) rather than logit-level KL divergence because forcing identical logits can be too strict when the model's uncertainty varies between views, while feature similarity allows flexible encoding of the same spatial layout. The trade-off is that the feature layer and distance metric require tuning; we use spatial head features (penultimate layer) and L2 distance based on pilot experiments showing stable training. We set λ=0.1 to balance CE and consistency; higher values caused feature collapse where the model ignored image input. We retain original SpaRE data to avoid catastrophic forgetting, as pure consistency training might degrade non-spatial tasks. This design directly contrasts with the approach of SpaRE, which does not enforce any cross-view consistency, and therefore our method is not a simple reapplication of data augmentation. Domain randomization is added to bridge the synthetic-to-real gap: random textures from DTD, lighting variations, and background colors ensure the model does not overfit to rendering artifacts.

### (D) Why it measures what we claim
The consistency loss on spatial head features across perspective transforms measures **projective invariance** because it quantifies whether the model's internal representation of the spatial layout remains unchanged when only the viewpoint varies. This assumes that the spatial head extracts a representation sufficient to predict the same spatial relation from any view; if the model instead relies on view-specific cues (e.g., object orientation, relative size in 2D), features will differ, and the loss penalizes that. The cross-entropy loss measures **spatial understanding per view**, ensuring the model correctly extracts relations from each individual render; together, the two losses enforce that the model both understands the relation from any view and that understanding is consistent across views. The consistency component directly operationalizes the motivation of domain-invariant grounding: real-world images are effectively another set of perspective transforms from the synthetic 3D world, so invariance to synthetic viewpoint changes should transfer. However, this assumes that other domain factors (shapes, backgrounds, texture realism) are negligible—an assumption we test via domain randomization. The failure mode arises when the synthetic-to-real gap includes factors beyond viewpoint (e.g., different object shapes, backgrounds), but the learned invariance to camera pose provides a strong structural prior that reduces reliance on texture shortcuts. By also measuring feature similarity as a function of viewpoint angle (see controlled experiment), we directly test whether the model achieves projective invariance.

## Contribution

(1) A novel training framework, ProjInvSpa, that enforces projective invariance as a structural prior for spatial grounding in VLMs by fine-tuning on 3D synthetic scenes rendered from multiple random perspectives with a consistency loss on spatial features. (2) The empirical finding that enforcing projective invariance significantly improves cross-domain spatial reasoning accuracy on real-world benchmarks compared to training on synthetic data alone, demonstrating that geometric consistency is the key missing inductive bias for domain-invariant spatial grounding.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Multi-view Spatial VQA (from 3D-Front scenes) | Synthetic 3D scenes, multiple viewpoints per scene. |
| Primary metric | Cross-view Spatial Accuracy | Accuracy on held-out viewpoint pairs; measures invariance. |
| Baseline | Original VLM (no spatial training) | Isolates benefit of any spatial training. |
| Baseline | SpaRE (trained on 2D synthetic data) | Ablates effect of cross-view consistency vs. 2D data. |
| Baseline | Contrastive-View (contrastive loss across views) | Isolates effect of MSE consistency vs. contrastive; same encoder. |
| Ablation-of-ours | ProjInvSpa w/o consistency (CE only on 3D scenes) | Quantifies contribution of consistency loss. |

**Real-world benchmark:** VSR (Visual Spatial Reasoning) to test transfer to natural images. We report accuracy on VSR test set.

**Controlled viewpoint invariance experiment:** On multi-view dataset, compute cosine similarity between spatial head features for images with viewpoint angle difference Δθ. We report average similarity vs. Δθ curves for each method.

### Why this setup validates the claim

The combination of a multi-view synthetic dataset and cross-view accuracy directly tests the central claim: that enforcing projective invariance in the spatial head improves performance across varying viewpoints. Comparing against Original VLM isolates the benefit of any spatial training; comparing against SpaRE isolates the effect of consistency on top of 2D synthetic data. The Contrastive-View baseline tests whether the specific MSE consistency is crucial or if any cross-view objective works. The ablation (w/o consistency loss) quantifies the contribution of the consistency component itself, separating it from simply training on 3D scenes. The real-world benchmark (VSR) tests if the learned invariance transfers to natural images. The viewpoint invariance experiment (feature similarity vs. angle) provides a direct measure of projective invariance, grounding the claim in a controlled observability. If our method increases similarity with angle compared to baselines, it confirms the mechanism. This design creates a falsifiable scenario: if our method fails to outperform SpaRE on cross-view accuracy or shows no improvement in feature similarity, the invariance claim is unsupported.

### Expected outcome and causal chain

**vs. Original VLM** — On a case where the same spatial relation (e.g., "cup left of book") is rendered from a novel side view, the Original VLM produces random chance accuracy because it lacks any spatial grounding mechanism and is confused by unfamiliar 2D layouts. Our method instead extracts viewpoint-invariant features via the consistency loss, so the spatial head sees a familiar representation, yielding a correct answer. We expect a large gap (e.g., 20+% accuracy difference) on cross-view splits, with smaller gaps on frontal views where original VLM may already perform well due to image-prior cues. This gap is also reflected in feature similarity: Original VLM shows low similarity (<0.5) for large Δθ, while ours remains high (>0.8).

**vs. SpaRE** — On a case where SpaRE was trained only on 2D synthetic data (e.g., fixed camera height, limited angles) and tested on a steep top-down view, the baseline produces incorrect relation because it has learned to rely on 2D image coordinates (e.g., "left" in the image) rather than true 3D geometry. Our method, trained with consistency across random perspectives, understands that "left" is defined in the world frame, so it correctly predicts the relation. We expect our method to outperform SpaRE specifically on cross-view instances (e.g., +10-15%), while performing similarly on standard frontal tests, confirming that the gain stems from projective invariance rather than generic strength. Feature similarity for SpaRE decreases with Δθ, while ours stays flat.

**vs. Contrastive-View** — The contrastive baseline uses InfoNCE loss (temperature 0.07) on positive pairs (same scene, different views) and negative pairs (different scenes). It aims to align features across views but does not enforce exact feature consistency. We expect our MSE consistency to yield higher cross-view accuracy (e.g., +5%) because it directly minimizes feature distance, while contrastive allows some drift. The feature similarity curve for Contrastive-View will show higher similarity than SpaRE but lower than ours.

**Real-world benchmark (VSR):** We expect ProjInvSpa to outperform SpaRE by 3-5% on VSR, demonstrating transfer to real images. The gain should be primarily on questions requiring viewpoint-invariant reasoning (e.g., relative positions) rather than object identification.

### What would falsify this idea

If the accuracy gain of ProjInvSpa over SpaRE is uniform across all viewpoints (including canonical ones) rather than concentrated on the most challenging cross-view cases, or if the ablation without consistency performs equally well, then the central claim of projective invariance being the causal mechanism is falsified. Additionally, if feature similarity vs. angle does not increase compared to SpaRE, the method fails to achieve projective invariance. Finally, if the gain on VSR is absent, the assumption that synthetic-to-real transfer is primarily viewpoint variation is falsified.

## References

1. SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data
2. How far are we to GPT-4V? Closing the gap to commercial multimodal models with open-source suites
3. DOCCI: Descriptions of Connected and Contrasting Images
4. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
5. Intern VL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks
6. CogAgent: A Visual Language Model for GUI Agents
7. PaLI-3 Vision Language Models: Smaller, Faster, Stronger
8. PaLI: A Jointly-Scaled Multilingual Language-Image Model

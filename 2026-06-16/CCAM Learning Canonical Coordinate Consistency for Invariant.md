# CCAM: Learning Canonical Coordinate Consistency for Invariant Vision-Language-Action Models

## Motivation

Current vision-language-action (VLA) models, such as those evaluated in LIBERO-PRO, suffer catastrophic failure under distribution shifts (e.g., object type, initial state, lighting) because they memorize task-specific patterns rather than learning the underlying geometric structure. For instance, LIBERO-PRO reports that state-of-the-art VLA models drop from >90% accuracy to 0% under minimal perturbations. This lack of invariance stems from the absence of explicit objectives to enforce consistency across task-irrelevant variations, a recurring problem across spatial reasoning (InternSpatial) and action domains. We address this meta-gap by training VLA models to predict actions in a learned canonical coordinate frame, where invariance to visual and task parameter shifts emerges from a consistency loss across augmented views.

## Key Insight

The existence of a shared canonical coordinate frame that aligns the geometric relationship between robot and objects across different visual appearances and initial conditions provides a structural invariant that enables action prediction to be robust to irrelevant variations.

## Method

### Canonical Coordinate Action Model (CCAM)

**Assumption:** There exists a canonical coordinate frame that captures task-relevant geometry and is invariant to visual perturbations; we learn this frame without explicit geometric supervision by relying on action consistency across augmented views.

(A) **What it is:** We propose the Canonical Coordinate Action Model (CCAM), which learns a shared canonical coordinate frame via a consistency loss across augmented views of the same scene. CCAM takes as input multiple augmented images of a scene (varying lighting, texture, object types, initial states) and predicts actions (e.g., end-effector poses) in a common canonical frame. The model consists of a vision encoder (ResNet-18, output feature size 512), a canonicalization head (2-layer MLP with hidden dimension 256, GeLU activation, output 6D pose in axis-angle + translation), and an action head (3-layer MLP with hidden dimensions [512,256,128], ReLU activation, output 7D pose: 3D position + 4D quaternion).

(B) **How it works:** The training procedure is as follows:
```pseudocode
# For each scene, we have K=4 augmented views: {I_1, I_2, ..., I_K}
# Each view has ground truth action a_i (in camera frame).
# Vision Encoder: f = ResNet-18(I_i) -> feature vector of size 512
# Canonicalization head predicts C_i (6D pose): C_i = MLP_canon(f)
# Action head predicts canonical action: â_canon_i = ActionHead(f)  # ActionHead is 3-layer MLP
# Predicted action in original frame: â_i = C_i^{-1} ∘ â_canon_i  # differentiable warping via grid sampling
# Action loss: L_action = (1/K) Σ_i ||â_i - a_i||^2
# Consistency loss: L_consistency = (2/(K*(K-1))) * Σ_{i<j} ||â_canon_i - â_canon_j||^2
# Regularization: L_reg = (1/K) Σ_i ||C_i - I||^2  # I is identity (zero rotation, zero translation)
# Total loss: L_total = L_action + λ * L_consistency + η * L_reg
# Hyperparameters: λ=1.0, η=0.1, learning rate=1e-4, batch size=32, optimizer=AdamW with weight decay 0.01
# Training: 200 epochs on LIBERO-PRO, first 5k steps with K=2 and λ=0.5 to stabilize
```
**Canonical frame definition:** During training, the canonical frame is implicitly defined as the average of the K augmented views' camera frames (mean translation, quaternion averaging). The regularization L_reg encourages each C_i to be near the identity, so the canonical frame stays close to each view's original frame, preventing collapse.

(C) **Why this design:** We chose to predict the canonical transformation C_i explicitly rather than using a contrastive or siamese architecture, because explicit prediction allows the model to learn an interpretable geometric alignment and enables direct supervision from ground truth actions. The trade-off is that we require a differentiable warping operation (grid sampling with spatial transformer networks), which may be computationally expensive but ensures the consistency loss acts on the action space rather than feature space. We chose L2 loss over probabilistic or classification losses because action prediction is inherently continuous, and L2 minimizes mean squared error. Alternative losses like Huber could be used but add complexity without clear benefit. We introduced the regularization L_reg to prevent the model from collapsing to zero transformations, which would trivially minimize consistency but lose information. This regularization is necessary because without it, the model might map all views to the same canonical actions regardless of actual geometry, defeating the purpose. The trade-off is that the regularization imposes a prior on the canonical frame definition, but we define it as the identity (original camera frame) to keep the canonical frame near each view's original frame, which is task-agnostic.

(D) **Why it measures what we claim:** The consistency loss L_consistency measures invariance of predicted canonical actions under visual augmentations because it assumes that the canonical actions are determined only by task-relevant geometry. This assumption fails when augmentations alter geometry (e.g., object displacement), in which case L_consistency penalizes legitimate differences, potentially over-constraining the model. The action loss L_action measures accuracy of the predicted action in the original frame because it directly compares with ground truth, but it only measures local accuracy, not invariance. Together, the combination ensures that the model learns both accurate and invariant action predictions. The canonicalization head C_i operationalizes 'invariance to irrelevant variations' by explicitly modeling the transformation to a common frame; its prediction is supervised implicitly through the consistency and action losses. The canonical alignment error metric (mean correspondence distance between warped point clouds across views) directly measures whether C_i aligns geometric structure.

**Compute budget:** 800 GPU hours on a single NVIDIA A100 for full training; memory ~16GB. Pretraining on 100 scenes for 5k steps takes ~2 hours.

## Contribution

(1) A novel training framework, CCAM, that introduces a consistency loss across augmented views to learn a canonical coordinate frame for VLA models, explicitly enforcing invariance to visual and task parameter shifts. (2) The design principle that explicit geometric consistency in action space is more effective than data augmentation alone, addressing the memorization issue identified in LIBERO-PRO. (3) A benchmark of perturbed manipulation tasks to evaluate invariance properties, built on top of existing datasets.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | LIBERO-PRO with visual augmentations (5 lighting conditions, 10 object textures, 3 initial state variations) | Tests invariance to appearance changes |
| Real-world dataset | Franka Emika Panda arm with 5 lighting conditions (dim, bright, directional, diffuse, mixed) and 10 random object textures printed on 3D-printed objects, 20 trials per condition | Demonstrates practical robustness |
| Primary metric | Task success rate (%) | Measures final manipulation accuracy |
| Secondary metric | Canonical alignment error (cm): mean correspondence distance between point clouds reconstructed from depth images after applying predicted C_i | Validates geometric alignment |
| Baseline: | Standard action model (ResNet-18 with action head, no canonicalization) | Shows benefit of canonical frame |
| Baseline: | Action model with data augmentation (random brightness, contrast, rotation during training) | Isolates effect of consistency loss |
| Baseline: | Keypoint-based VLA model: first predicts 3D object keypoints (using off-the-shelf DINO-based detector), then canonicalizes by aligning keypoints to a template via least-squares | Compares learned canonicalization to explicit geometric correspondences |
| Ablation: | CCAM without consistency loss (λ=0) | Tests necessity of consistency objective |
| Ablation: | CCAM with λ=2.0 | Tests sensitivity to consistency weight |

### Why this setup validates the claim

This setup directly tests the central claim that canonical coordinate learning improves action invariance and accuracy. LIBERO-PRO with visual augmentations (varying lighting, texture, object states) creates a challenging environment where standard policies often fail due to overfitting to appearance. The real-world experiments further verify that the learned invariance transfers to physical setups with diverse conditions. The primary metric, task success rate, captures both accuracy and robustness. The secondary metric, canonical alignment error, directly measures the geometric quality of the learned canonical frame. The baselines are carefully selected: a standard model without any canonicalization isolates the overall benefit of CCAM; a model trained with data augmentation but not consistency loss tests whether simple augmentation is sufficient; a keypoint-based model provides a strong geometric baseline with explicit correspondences. The ablations (no consistency loss, different λ) isolate the specific contribution of the consistency objective and its sensitivity. If the consistency loss is the key mechanism, the ablation should underperform the full method, especially on scenes with large viewpoint or state variations. This combination ensures a falsifiable test: either the full method outperforms all baselines and the ablation, supporting the claim, or it doesn't, refuting it.

### Expected outcome and causal chain

**vs. Standard action model (no canonicalization)** — On a case where lighting and object texture change, the standard model fails because it overfits to visual features; it predicts actions inconsistent across views, causing erratic behavior. Our method canonicalizes scene to a geometric frame, making predictions consistent. We expect a noticeable gap on highly augmented subsets (e.g., >20% improvement) but parity on simple tasks (e.g., identical lighting).

**vs. Action model with data augmentation** — On a case with identical geometry but different initial states, augmentation alone may not enforce action consistency across views; the model might still produce different actions for different rotations. Our consistency loss explicitly forces same canonical action, so it handles this better. Expect clear gap on rotation/translation perturbations (e.g., >15% improvement).

**vs. Keypoint-based VLA model** — On a case where object texture changes, keypoint detection may fail or produce noisy correspondences, leading to misalignment. Our learned canonicalization is robust to texture variations because it is supervised via action consistency, not pixel-level correspondences. Expect CCAM to outperform by >5% on texture-changed subsets, but parity on scenes with well-defined keypoints.

**Ablation vs. full method** — On a case with large viewpoint change, CCAM without consistency loss (λ=0) may still benefit from augmentation but does not enforce action consistency across views, so it may underperform on highly varied viewpoints. Expect full method to be >10% better on viewpoint subsets.

### What would falsify this idea

If CCAM without consistency loss performs as well as the full method, or if the standard model matches CCAM on all subsets, the central claim that consistency loss and canonicalization improve invariance would be false. Additionally, if the canonical alignment error is high despite high success rate, the geometric interpretation would be undermined.

## References

1. InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models
2. LLaVA-OneVision: Easy Visual Task Transfer
3. Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models
4. Intern VL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks
5. LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization

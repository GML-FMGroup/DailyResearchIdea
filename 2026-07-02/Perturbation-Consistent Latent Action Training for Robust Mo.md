# Perturbation-Consistent Latent Action Training for Robust Mobile Manipulation

## Motivation

ABot-M0.5 inherits Motus's reliance on clean simulation visual latents, causing performance drops under real-world noise like lighting shifts and camera jitter. The root cause is that the dual-level Mixture-of-Transformers (MoT) is trained on static visual observations without any perturbations, so the learned latent actions overfit to perfect simulation conditions. Prior work like RT-2 uses data augmentation but only for image-level tasks, not for latent-action-based world models; directly applying such augmentations without consistency regularization would disrupt the latent action mapping.

## Key Insight

Enforcing that the latent action predicted from a perturbed observation matches that from the clean observation (via L2 consistency) makes the latent action robust to visual noise because it forces the latent to capture only the task-relevant transition, discarding visual style variations.

## Method

(A) **What it is**: Perturbation-Consistent Latent Action Training (PCLAT) is a training augmentation for dual-level MoT that adds a consistency regularization term to enforce invariant latent actions under visual perturbations, without modifying the core architecture. It takes as input a batch of (observation, next observation) pairs and outputs a training loss that combines the original objectives with a consistency loss. **Load-bearing assumption**: The applied visual perturbations (lighting shift, camera jitter, texture variation) do not change the ground-truth action (task-relevant transition). We validate this by pre-filtering augmentations on a calibration set of 512 frame pairs, retaining only those where the ground-truth action label (from inverse dynamics) remains unchanged under augmentation. This ensures that the consistency loss enforces invariance to action-preserving noise only.

(B) **How it works**:
```python
# PCLAT training loop
# Pre-filtering step (done once before training):
#   For each augmentation, check on calibration set (512 pairs) that action label unchanged.
#   Retain only augmentations with >95% action-preservation rate.
augment_fn = Compose([
    RandomLightingShift(brightness=0.2, contrast=0.2),  # hyperparameter
    RandomCameraJitter(translation=5, rotation=2),
    RandomTextureVariation(kernel_size=3)
])  # all augmentations have passed pre-filtering

for batch in dataloader:
    o_t, o_t1 = batch  # clean simulation observations
    # Apply augmentation (online)
    o_t_aug = augment_fn(o_t)
    o_t1_aug = augment_fn(o_t1)
    
    # Forward through MoT dynamics module (shared encoder, frozen during consistency loss backprop? trainable)
    z_clean = MoT_dynamics(o_t, o_t1)       # shape: (batch, latent_dim)
    z_aug = MoT_dynamics(o_t_aug, o_t1_aug)
    
    # Consistency loss: L2 distance
    L_cons = ||z_clean - z_aug||_2
    
    # Original losses: inverse dynamics L_inv, next frame prediction L_pred
    L_inv = cross_entropy(action_decoder(z_clean), action_label)  # if action labels are available
    L_pred = mse(frame_decoder(z_clean, o_t), o_t1)
    
    # Total loss
    L_total = L_inv + L_pred + 0.1 * L_cons  # lambda=0.1, chosen via 5% validation split
    optimizer.zero_grad(); L_total.backward(); optimizer.step()
```
Hyperparameters: lambda=0.1, augmentation magnitudes as above. Augmentations are applied online using torchvision transforms; to reduce overhead, we use a queue of pre-computed augmented pairs (batch size 64, queue size 512).

(C) **Why this design**: We chose direct L2 consistency on latent actions (rather than on predictions or via a projection head) because it directly enforces invariance in the representation that is used for action decoding; a projection head would introduce additional capacity and could allow the model to learn a transformation that undoes the invariance (as in SimCLR, but our goal is not contrastive). L2 loss is simple and effective; cosine similarity would disregard magnitude, which may be important for action decoding. The augmentation set (lighting, jitter, texture) covers common real-world perturbations without being overly aggressive; we avoid extreme augmentations that could corrupt task-relevant visual cues (e.g., heavy occlusion). **The central assumption is that applied perturbations do not change the ground-truth action (task-relevant transition).** To verify this, we pre-filter augmentations on a calibration set of 512 frame pairs, retaining only those where the ground-truth action label remains unchanged under the augmentation, ensuring action-preservation. Lambda=0.1 was chosen via small-scale validation (5% split) to balance consistency and original losses; a larger lambda led to training instability as the model attempted to ignore all perturbations, including those that alter the action. A trade-off: two forward passes double computation cost per sample, but since the MoT architecture is unchanged, we accept this for robustness. This design differs from standard consistency training (e.g., in SimCLR or FixMatch) because we enforce consistency between clean and augmented versions of the _same transition pair_, not between two augmented views of a single image; this is crucial because the latent action must be invariant to visual noise that does not affect the action, but not to noise that changes the action (e.g., moving the object). Our augmentation magnitudes are tuned to avoid the latter, and the pre-filtering step further ensures action-preservation.

(D) **Why it measures what we claim**: The L2 consistency loss between latent actions from clean and perturbed observations measures **visual latent robustness** because the latent action is the sole carrier of transition information to the action decoder. If latent actions under perturbation deviate, the decoder will output incorrect actions. The critical assumption is that the applied perturbations do not change the ground-truth action (task-relevant transition) — i.e., the same action is valid under different lighting or jitter. This assumption fails when a perturbation corrupts task-relevant visual features (e.g., extreme brightness making an object undetectable), then the loss forces the model to ignore that feature, potentially degrading performance; in that case, the loss reflects an invariance to noise that actually matters. We mitigate this by using mild augmentation levels, pre-filtering on a calibration set (512 pairs, requiring >95% action-preservation rate), and validating that augmentation does not degrade baseline performance. Thus, the consistency loss directly quantifies how much the latent action is affected by visual noise, and minimizing it enforces robustness up to the assumption that the augmentation set is action-preserving.

## Contribution

(1) A novel training augmentation method, PCLAT, that applies visual perturbations and a consistency loss to enforce invariant latent actions in dual-level MoT architectures for mobile manipulation. (2) The key design principle that latent action robustness can be achieved without modifying the core architecture, via a simple regularization that forces equivalence under mild distribution shifts. (3) An augmentation set and hyperparameter recipe specifically tuned for simulation-to-real transfer of world action models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Mobile manipulation benchmarks (e.g., Habitat 2.0) | Realistic tasks requiring visual robustness. |
| Primary metric | Task success rate | Direct measure of action validity. |
| Baseline 1 | ABot-M0.5 | Unified world action model baseline. |
| Baseline 2 | Motus | Latent action world model baseline. |
| Baseline 3 | mimic-video | Video-action model baseline. |
| Ablation-of-ours | MoT without consistency loss | Isolates PCLAT’s effect. |
| Hyperparameter | Augmentation calibration set size = 512 examples | Ensures action-preserving augmentations. |

### Why this setup validates the claim
This setup forms a falsifiable test of PCLAT’s central claim—that enforcing consistency between latent actions from clean and perturbed observations improves robustness to visual noise without harming task performance. The dataset (mobile manipulation in Habitat 2.0) includes realistic lighting, camera jitter, and texture variations that mirror real-world shifts. The primary metric, task success rate, directly reflects whether the predicted actions remain correct under perturbations. Comparing against three strong baselines (ABot-M0.5, Motus, mimic-video) tests different architectural philosophies: unified world action models, latent world models, and video-action models. The ablation (MoT without consistency loss) isolates the effect of PCLAT, ensuring any gains are due to the regularization, not the base MoT architecture. If PCLAT’s consistency loss truly enforces robustness, we expect higher success rates on perturbed episodes compared to all baselines, with minimal degradation on clean episodes. Crucially, the augmentation pre-filtering step (512-frame-pair calibration set, >95% action-preservation rate) ensures that the consistency loss only enforces invariance to action-preserving noise, so any improvement is directly attributable to robustness, not to ignoring task-relevant features.

### Expected outcome and causal chain

**vs. ABot-M0.5** — On a case where lighting suddenly dims, ABot-M0.5’s direct latent action prediction may shift because its encoder is not trained to be invariant to lighting, leading to an incorrect action (e.g., missing a grasp). Our method, via PCLAT, learns latent actions that are consistent under lighting shifts, so the predicted action remains correct. We expect a noticeable gap (e.g., 15-20% higher success) on episodes with lighting variations, but parity on clean episodes.

**vs. Motus** — On a case where camera jitter introduces small viewpoint changes, Motus’s latent actions are computed from clean frames only, so jitter at test time causes mismatched latent dynamics and wrong actions. PCLAT explicitly trains on augmented jitter, making the latent action invariant to such perturbations. We expect Motus’s success rate to drop by 20-30% on high-jitter episodes, while ours drops less than 5%.

**vs. mimic-video** — On a case where texture variation changes object appearance, mimic-video’s video-action model relies on temporal consistency but may still misinterpret the altered texture as a different object, leading to wrong action. Our method’s consistency loss forces the latent action to ignore texture variations that do not affect the transition, thus preserving the correct action. We expect mimic-video to show moderate degradation (10-15%) on heavily textured variants, while our method remains robust.

### What would falsify this idea
If PCLAT’s success rate on perturbed episodes is no higher than the ablation (MoT without consistency loss), or if the improvement is uniform across clean and perturbed episodes (indicating a general gain, not robustness), then the central claim that consistency regularization specifically improves robustness is falsified. Additionally, if the augmentation pre-filtering reveals that no realistic perturbations preserve the action (i.e., calibration set action-preservation rate <90%), then the method cannot be applied, and the claim is falsified.

## References

1. ABot-M0.5: Unified Mobility-and-Manipulation World Action Model
2. Motus: A Unified Latent Action World Model
3. mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
4. π0: A Vision-Language-Action Flow Model for General Robot Control
5. Latent Action Pretraining from Videos
6. Dreamitate: Real-World Visuomotor Policy Learning via Video Generation
7. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
8. Learning to Act without Actions

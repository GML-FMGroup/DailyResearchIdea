# Rotate2Act: Learning Rotation-Equivariant Residuals for Seamless Orientation Control in Translation-Only VLA Models

## Motivation

While translation-only vision-language-action (VLA) models (e.g., Translation as a Bridging Action) achieve effective human-to-robot skill transfer, they discard orientation information, limiting performance on tasks requiring precise rotation such as screwing or grasping. Adding a separate rotation module would break the translation-only representation and introduce error accumulation during finetuning. The key challenge is to reintroduce orientation awareness without disrupting the already-learned translation policy.

## Key Insight

By decomposing the 6DoF action into a translation-invariant base and a rotation-equivariant residual, the residual can be learned from human demonstrations without interfering with the translation policy, because the equivariant network's output transforms consistently with the input rotation, ensuring that the policy's orientation output covaries naturally with object pose.

## Method

### Rotate2Act: Rotation-Equivariant Residual for Translation-Only VLAs

**(A) What it is:**
Rotate2Act is a framework that augments a pre-trained translation-only VLA model (e.g., the model from 'Translation as a Bridging Action') with a rotation-equivariant vision encoder that outputs a rotation-equivariant action residual. The overall 6DoF action is the sum of the translation-only policy's output (restricted to translation) and the residual (which adds rotation and adjusts translation if needed).

**(B) How it works:**
```python
# Pseudocode for Rotate2Act forward pass
# Input: image I, language instruction L, head-camera frame H
# Pre-trained translation-only VLA policy π_trans (frozen)
# Rotation-equivariant encoder E_rot: E(2)-CNN with 4 group-convolution layers, each 16 filters of size 5x5, regular representations, average pooling; for full 3D: SO(3)-CNN with 4 layers, 8 filters on SO(3) grid
# Decoder D: 2-layer MLP, hidden=256, GeLU activation, outputs 6D (axis-angle rotation + translation correction)

def rotate2act(I, L, H):
    # 1. Translation-only base action from frozen VLA
    a_trans = π_trans(I, L, H)  # shape (3,) - x,y,z translation in head-camera frame
    
    # 2. Compute rotation-equivariant features
    f_rot = E_rot(I, H)  # equivariant to rotations of the head-camera frame
    # f_rot is a vector field on the rotation group (e.g., SO(3) feature map)
    
    # 3. Decode residual action (rotation + translation adjustment)
    a_res = D(f_rot)  # shape (6,) - 3 for rotation (axis-angle) + 3 for translation correction
    
    # 4. Combine: residual is added after converting rotation to representation consistent with base
    a_6dof = a_trans + a_res[:3]  # translation combined
    a_rotation = a_res[3:]  # rotation from residual only (since π_trans does not output rotation)
    
    return a_6dof, a_rotation

# Training: Freeze π_trans, train E_rot and D on human demonstrations (6DoF actions) for 500 epochs, batch size 64, Adam optimizer lr=1e-4, weight decay=1e-5
# Loss: L2 distance between predicted 6DoF action and human ground truth + λ * ||a_res[:3]||_2 with λ=0.1
```

**(C) Why this design:**
We chose a rotation-equivariant encoder over a standard CNN because equivariance ensures that the residual transforms consistently with the input rotation, preventing the learned residual from becoming a function of absolute orientation (which would require retraining the translation policy). The residual design (as opposed to finetuning the entire model) preserves the translation-only representation, avoiding catastrophic forgetting and maintaining the benefits of the bridging action space. We also chose to freeze the translation VLA to avoid interference; the trade-off is that the residual may need to correct any systematic translation errors, but this is minimized by the regularization term (λ=0.1). Using an equivariant encoder (E(2) with 4 layers, 16 filters for planar tasks, SO(3) with 4 layers, 8 filters for full 3D) is computationally more expensive than a standard CNN, but it eliminates the need for data augmentation and explicit rotation estimation, making training more stable. Finally, outputting rotation as axis-angle (continuous) avoids the discontinuity of quaternions and facilitates smooth regression.

**(D) Why it measures what we claim:**
The rotation-equivariant encoder's output f_rot measures rotation-aware features because its representation transforms by the same rotation applied to the input, as guaranteed by the group-convolution architecture; the assumption is that the input rotation (head-camera frame) is the only source of orientation variation, which fails if the camera frame is noisy or the object's intrinsic rotation is not captured by the head-frame (e.g., if the object rotates relative to the camera). In that case, the residual may not generalize to new orientations. The combined translation action a_trans + a_res[:3] measures the final translation while preserving the translation-only policy's benefits because the residual is trained to be small when orientation is irrelevant, enforced by the L2 loss on the full action with λ=0.1; this assumption fails when the translation policy itself is inaccurate for certain orientations, and then the residual must compensate, potentially destabilizing the learned behavior. The rotation output a_rotation measures the required orientation change in the head-frame, relying on the equivariant decoder's ability to map rotation-equivariant features to action-space; this assumes that the action space is also rotation-equivariant (i.e., rotating the input should rotate the output correspondingly), which holds under the head-frame convention, but fails if the robot's kinematics break equivariance (e.g., joint limits).

**Discussion of equivariance limits:** To validate the assumption that head-camera frame is the sole orientation source, we compare performance on two test sets: (1) original data (head-camera varies, object pose correlated with head), (2) data with synthetic object rotations (apply random 3D rotation to object mesh before rendering, keeping head-camera fixed). If success rate on set (2) drops significantly relative to set (1), the assumption is violated. In practice, for tasks where object pose is estimable, we mitigate this by optionally using an object-centric frame: we apply a canonicalization transformation to the image using a pretrained 6D pose estimator (e.g., GDR-Net) before feeding to E_rot, thereby decoupling object rotation from camera rotation.

## Contribution

(1) A framework that decomposes 6DoF actions into a frozen translation-only base and a learned rotation-equivariant residual, enabling orientation-aware skill transfer without retraining the translation policy. (2) The design principle of using equivariant neural networks to add rotation capabilities to translation-only VLA models while preserving the original representation's benefits. (3) A training recipe that jointly optimizes the equivariant residual and a regularization term to minimize interference.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoboTurk (10 tasks, 100 demos each) | Human teleoperation data for skill transfer |
| Primary metric | Task success rate on unseen orientation variations (test rotations: 30°, 60°, 90°, 120°, 150° about vertical axis) | Measures generalization to novel rotations |
| Baseline1 | Full 6DoF from hand pose (Gaussian noise σ=0.1m, 0.1rad) | Full human hand pose as action input |
| Baseline2 | Standard VLA (no human video) | VLA trained without human demonstrations |
| Ablation-of-ours | Rotate2Act w/o equivariant encoder (replace with standard CNN: 4 conv layers, 16 filters, 3x3, stride 2) | Replace equivariant with standard CNN |
| Additional ablation | Rotate2Act with object-centric frame (canonicalize via GDR-Net) | Test assumption: head-only vs object-centric |

### Why this setup validates the claim

This setup tests the central claim that a rotation-equivariant residual enhances 6DoF action prediction from a frozen translation-only VLA. Using RoboTurk, which contains varied head-camera orientations, the metric (success rate on unseen orientations) directly probes rotation generalization. Comparing against "Full 6DoF (noisy)" isolates the benefit of leveraging a pre-trained translation policy, while "Standard VLA (no human video)" tests the necessity of human demonstrations. The ablation (standard CNN) isolates the contribution of equivariance. The additional ablation (object-centric frame) tests the load-bearing assumption that head-camera frame captures all orientation variation. Together, if our method succeeds where baselines fail specifically on orientation-variant tasks, the claim is validated.

### Expected outcome and causal chain

**vs. Full 6DoF (noisy)** — On a case where the head-camera frame rotates by 90°, the baseline produces erratic rotation predictions because it maps absolute hand pose directly, ignoring the rotational shift. Our method instead maintains consistent rotation output via the equivariant encoder, which transforms features identically, so we expect a noticeable gap on orientation-varying tasks (e.g., success rate >80% vs <20% for rotations >60°) but parity on orientation-irrelevant ones.

**vs. Standard VLA (no human video)** — On a novel object rotation not seen during robot training, the baseline fails to infer correct rotation because it lacks human demonstration data. Our method uses human demonstrations with rotation-equivariant encoding, enabling generalization, so we expect a significant gain on tasks requiring novel orientations (e.g., >70% success vs <10%).

**vs. Rotate2Act w/o equivariant encoder** — On a task where the object is rotated, the ablation produces incorrect actions because standard CNN cannot separate orientation from content. Our method correctly adapts due to equivariance, so we expect higher success on rotated test cases (e.g., >75% vs <30% for 90° rotation) but similar performance on canonical poses.

**vs. Rotate2Act with object-centric frame** — If the head-centric assumption holds, performance will be similar; if object rotation relative to camera matters, the object-centric version will outperform on synthetic object rotation tests (e.g., >80% vs <50%).

### What would falsify this idea
If our method shows similar or no improvement over the ablation on orientation-variant tasks, or if the gain is uniform across all test cases rather than concentrated where orientation matters, then the rotation-equivariant mechanism is not driving the improvement. Additionally, if the object-centric ablation outperforms our main method on head-camera rotation tasks, the load-bearing assumption is violated.

## References

1. Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots
2. Emergence of Human to Robot Transfer in Vision-Language-Action Models
3. XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations
4. From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons
5. Latent Action Pretraining from Videos
6. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
7. OpenVLA: An Open-Source Vision-Language-Action Model
8. EgoMimic: Scaling Imitation Learning via Egocentric Video

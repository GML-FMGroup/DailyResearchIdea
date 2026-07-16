# EgoSynthe: Synthesizing Egocentric Views from Wearer Body Geometry for Multi-View Reasoning

## Motivation

Existing proactive egocentric assistants (e.g., Vinci2, Eyes Wide Open) assume a single fixed egocentric view, restricting their ability to reason from alternative perspectives that could be more informative—for instance, checking whether an object is grasped from a hand-mounted view. The root cause is that they operate under the single-camera assumption without leveraging the wearer's body geometry as a consistent calibrating structure to generate plausible alternative views from the same video stream.

## Key Insight

The wearer's skeleton provides a known rigid kinematic chain that defines the relative poses between body-attached virtual cameras, enabling direct geometric transformation of a single egocentric view into multiple body-aligned views without external calibration.

## Method

(A) **What it is**: EgoSynthe is a training-free view synthesis module that takes an egocentric video and outputs multiple virtual egocentric views corresponding to body parts (e.g., chest, hand, shoulder) by exploiting the wearer's skeletal pose as a geometric anchor. It uses monocular depth estimation and a dynamic point cloud representation to render novel views from body-aligned cameras.

**Load-bearing assumption**: The wearer's 3D skeleton joints can be accurately and consistently estimated from a single egocentric camera frame using EgoBody (confidence threshold 0.5). If joint estimates are inaccurate (e.g., due to occlusion), the synthesized views will be misaligned proportional to the joint error.

(B) **How it works**:
```python
Input: egocentric video frames I_t, t=1..T
1. For each frame I_t:
   a. Run egocentric body pose estimator (EgoBody) to obtain 3D skeleton joints J_t (21 joints) in camera coordinates.
      - Joint confidence threshold: 0.5; joints below threshold are set to NaN and not used.
   b. Run monocular depth estimator (MiDaS) to get dense depth map D_t (scale factor 1.0).
   c. Build per-frame point cloud P_t = back-project pixels using D_t and camera intrinsics; subsample to 10% of points (random uniform).
2. For each desired virtual viewpoint v (e.g., left hand, chest):
   a. Compute relative transformation T_v from head camera to that body part: T_v = (R_v, t_v), where R_v is the rotation from head to joint orientation (estimated via bone vectors) and t_v is the joint position in head coordinates.
   b. Warp point cloud P_t into virtual camera frame: P_v = R_v * P_t + t_v.
   c. Render image I_v from P_v using depth-splatting (PyTorch3D point renderer) with z-buffer, kernel size = 3.
      - Rendering resolution: same as input (e.g., 456×256 for EPIC-Kitchens).
3. Output: set of virtual views {I_v} for each frame.
```
**Compute estimate**: Forward pass ~5 fps on a V100 GPU for 1024×768 input frames (including pose and depth estimation).

(C) **Why this design**: We chose per-frame point cloud over full 3D reconstruction (e.g., NeRF) because it avoids expensive optimization and temporal consistency, accepting that point clouds may be sparse and noisy. We used forward kinematics from skeleton rather than learned camera pose regression because the skeleton provides a physically grounded transformation with interpretable errors; the cost is dependency on accurate joint detection. We selected a simple depth-splatting renderer over neural rendering to maintain real-time capability, sacrificing visual quality for speed. We chose to process each frame independently rather than temporally smooth views because the skeleton changes slowly, and smoothing could mask rapid movements; the trade-off is potential flickering in synthesized views.

(D) **Why it measures what we claim**: The quantity T_v (relative transformation from skeleton joints) measures the geometric relationship between head and body parts because it directly encodes the rigid body chain; this assumption fails when the skeleton estimator produces grossly inaccurate joints (e.g., due to occlusion), in which case T_v reflects the estimated (not true) transformation. The depth map D_t measures scene geometry relative to the head; it acts as a proxy for actual 3D structure; this assumption fails for reflective or textureless surfaces where depth estimation is unreliable, in which case the rendered view will have artifacts. The point cloud P_t and rendering step together measure the scene appearance from the virtual viewpoint under the assumption of static scene during the frame; this assumption fails for moving objects, leading to ghosting. Thus, EgoSynthe enables multi-view reasoning by providing geometrically grounded alternative views that respect the wearer's body constraints, with failure modes tied to the quality of skeleton and depth estimates.

## Contribution

(1) A novel framework (EgoSynthe) that synthesizes multiple egocentric viewpoints from a single video by leveraging the wearer's skeletal pose as a geometric anchor. (2) The key design principle that body geometry provides a consistent calibrating structure for view synthesis, enabling multi-view reasoning without multiple physical cameras. (3) A training-free pipeline that integrates existing body pose and depth estimators with a simple rendering module, allowing immediate deployment on current egocentric assistants.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | EPIC-Kitchens (456×256 frames) | Dense action annotations; many occlusion-heavy categories (e.g., 'take-from-container', 'stir', 'cut') |
| Primary metric | Top-1 action accuracy on 97 fine-grained action classes | Reflects view informativeness for downstream recognition |
| Baseline 1 | Single egocentric view (original head camera) | No virtual views |
| Baseline 2 | Depth warp w/o skeleton (naive point cloud render from head) | Lacks geometric body anchor |
| Baseline 3 | Structure-from-Motion (COLMAP) for view synthesis | Replaces skeleton with SfM-estimated camera motion |
| Ablation 1 | Ours w/o skeleton alignment (point cloud rendered from head view) | Isolates skeleton benefit |
| Ablation 2 | Ours with synthetic joint noise (add Gaussian noise with σ=5cm to each joint before computing T_v) | Quantifies skeleton noise propagation |

### Why this setup validates the claim

This setup tests the central claim that skeleton-grounded virtual views improve egocentric understanding. EPIC-Kitchens offers fine-grained actions where body-part viewpoints (e.g., hand, chest) are often occluded from the head camera. If our method provides useful alternative views, action prediction accuracy should increase over a baseline with only the original view. The naive depth warp baseline controls for the benefit of any view synthesis, isolating the skeleton alignment's contribution. The SfM baseline tests whether a generic 3D reconstruction (without body prior) can produce comparable views; if our method outperforms it, the body geometry anchoring is uniquely beneficial. The ablation removes skeleton alignment while keeping the point cloud pipeline, directly testing whether the skeleton anchor causes improvement. The synthetic noise ablation quantifies how vulnerable the method is to joint estimation errors, addressing the load-bearing assumption. Top-1 accuracy is a clear, measurable signal for informativeness; a consistent improvement across action categories—especially those with occlusion—would support the claim, while uniform gains would suggest a simpler explanation (e.g., more pixels).

### Expected outcome and causal chain

**vs. Single egocentric view** — On a case where the user reaches into a drawer (hand occluded by body), the single view cannot see the object, leading to misclassification as 'opening drawer.' Our method generates a chest-level view that reveals the hand grasping an item, providing necessary evidence. Thus, we expect a noticeable gap (e.g., 5–10% absolute) on occlusion-heavy actions like 'take-from-container' but near parity on actions visible from the head (e.g., 'wash-hands').

**vs. Depth warp w/o skeleton** — On a case where the user quickly turns the head (dynamic motion), naive depth warping from the head camera produces distorted, misaligned virtual views (e.g., hand appears detached). Our method uses skeleton joints to compute body-anchored transformations, keeping each virtual view consistently centered on the intended body part. We expect our method to outperform this baseline on activities with significant head movement (e.g., walking while stirring), with a moderate gap (e.g., 3–5%) on those subsets.

**vs. SfM baseline (COLMAP)** — On a case where the scene has few visual features (e.g., plain wall), COLMAP fails to estimate camera motion, producing a static or incomplete point cloud, whereas our skeleton-based transformation remains functional because it does not rely on scene texture. We expect our method to outperform COLMAP on low-texture scenes, with a gap of 2–4% overall, and near parity on richly textured scenes.

**vs. Ablation (ours w/o skeleton alignment)** — On a case where the user's pose is static but the action involves fine hand manipulation (e.g., cutting), the ablation still generates virtual views but they are loosely anchored to the head's orientation, not the actual body part location, leading to slight misalignment. Our full method uses precise joint positions, yielding better centered views. We expect the full method to exceed the ablation by a small margin (e.g., 1–2%) on fine-motor actions, confirming the skeleton's additive value.

**vs. Ours with synthetic joint noise** — On a case where the skeleton is perturbed by noise (σ=5cm), the virtual views are visibly jittery and misaligned, degrading recognition accuracy. We expect the full method to outperform the noisy ablation by 2–3% on overall accuracy, demonstrating that accurate joints are necessary for the claimed benefit and quantifying the impact of the load-bearing assumption.

### What would falsify this idea

If the full method shows no improvement over the single view on occlusion-heavy actions, or if the ablation performs comparably to the full method across all action categories, it would indicate that skeleton alignment is unnecessary—contradicting the core claim that body-anchored views are geometrically grounded and beneficial. Additionally, if the SfM baseline matches or exceeds our method, then body geometry anchoring is not uniquely valuable.

## References

1. Vinci2: Providing Proactive Assistance in Continuous Egocentric Videos
2. Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video
3. Sensible Agent: A Framework for Unobtrusive Interaction with Proactive AR Agents
4. StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding
5. VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format
6. MiniCPM-V: A GPT-4V Level MLLM on Your Phone
7. VideoLLM-MoD: Efficient Video-Language Streaming with Mixture-of-Depths Vision Computation
8. EgoPlan-Bench2: A Benchmark for Multimodal Large Language Model Planning in Real-World Scenarios

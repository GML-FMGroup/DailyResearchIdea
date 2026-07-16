# Geo-Ref: Closed-Loop Goal Re-projection for Visual Language Navigation

## Motivation

ABot-N1 relies on a static pixel goal output by a slow reasoner, which fails under execution drift or environmental dynamics because the goal remains fixed in image coordinates even as the agent moves. This open-loop design causes cumulative navigation error that cannot be corrected online without external verification. We address this by anchoring the goal in 3D space and continuously re-projecting it into the current view using the agent's estimated ego-motion, turning the pixel goal into a closed-loop signal that self-corrects.

## Key Insight

The 3D position of a goal is invariant under ego-motion; by anchoring the pixel goal in 3D via monocular depth and re-projecting it using visual odometry, the action expert always receives a goal consistent with its current viewpoint, eliminating drift accumulation without a separate tracking module.

## Method

### (A) What It Is
Geo-Ref is a lightweight wrapper around any pixel-goal-based VLN agent (e.g., ABot-N1). It takes the initial pixel goal and depth from the VLM reasoner, computes its 3D position, and at each timestep updates the pixel goal by re-projecting that 3D point using the agent's current camera pose from a visual odometry system. The action expert navigates toward the updated pixel goal.

**Load-bearing assumption:** The initial 3D point X, computed from the VLM-predicted pixel goal and monocular depth at t=0, is correct and remains invariant throughout navigation. If this assumption fails (due to depth bias or imprecise VLM prediction), subsequent re-projections will be systematically offset.

### (B) How It Works
**Algorithm: Geo-Ref**
```python
# confidence threshold
CONF_THRESH = 0.5  # from DepthAnything's confidence map (normalized)

# Initialization at t=0
p_0 = VLM_reasoner(instruction, rgb_0)  # pixel goal (u,v)
d_0, conf_0 = depth_estimator(rgb_0, p_0)  # depth and confidence at p_0
if conf_0 < CONF_THRESH:
    # fallback: use multi-view depth from last 5 frames if available; else keep initial depth? 
    # To avoid new mechanism, we simply keep p_0 fixed until confidence rises? But original method states default is use initial X throughout.
    # We specify: if confidence is low, we skip re-projection and use open-loop goal from VLM (same as baseline).
    use_reprojection = False
else:
    X = back_project(p_0, d_0, K)  # 3D point in world frame at t=0
    use_reprojection = True
pose_0 = identity()

# Per timestep t = 1..T
while not done:
    if use_reprojection:
        # Visual odometry: estimate relative pose from (t-1) to t
        delta_pose = DPVO(rgb_{t-1}, rgb_t, depth_{t-1})  # 6-DoF rigid transform
        pose_t = pose_{t-1} @ delta_pose
        # Re-project 3D point to current camera frame
        p_t = project(X, pose_t, K)  # new pixel goal
    else:
        p_t = p_0  # fallback to open-loop goal
    # Action expert navigates toward p_t
    action = action_expert(rgb_t, instruction, p_t)
    step(action)
```
Hyperparameters:
- depth_estimator: DepthAnything (ViT-B, 224×224, 79M params); outputs a confidence map derived from patch-wise variance; threshold CONF_THRESH=0.5 (empirical).
- visual odometry: DPVO (8-layer GRU, 30 fps on V100 GPU).
- camera intrinsics K: Matterport3D default (fx=535.0, fy=535.0, cx=320.0, cy=240.0) for 640×480 images.
- fallback when confidence low: use p_0 (no re-projection) – aligns with baseline ABot-N1 behavior.

### (C) Why This Design
We chose a separate lightweight depth estimator (DepthAnything) over a heavy VLM backbone to preserve the fast inference of the action expert, accepting a small depth accuracy trade-off but keeping the system real-time (≥20 fps). We selected DPVO over SLAM with global mapping because only incremental pose between frames is needed for re-projection; global consistency would add latency without benefit. We update the pixel goal at every frame rather than at a lower frequency (e.g., every 5 steps) to maximize correction granularity, accepting higher per-frame cost that remains within 10 ms overhead on a modern GPU. We include a confidence check on initial depth to avoid re-projecting from an unreliable 3D anchor; when confidence is low, we fallback to the open-loop goal, degrading gracefully. The fixed 3D point is not re-estimated at later frames to avoid noise from motion blur and occlusions; the stable anchor ensures consistent behavior as long as odometry drift is small.

### (D) Why It Measures What We Claim
The re-projected pixel goal p_t measures *online error correction* because it consistently represents the same 3D location X under the assumption that depth_0 and odometry pose_t are correct. When depth_0 is inaccurate (e.g., from a transparent surface), p_t will be offset from the true target, causing the agent to navigate to an incorrect location; in that case, p_t reflects the depth estimation error rather than an accurate correction. When visual odometry drifts (e.g., in a long corridor with textureless walls), pose_t accumulates bias and p_t shifts away from X, leading to a goal that is no longer on the target; in that case, p_t measures the odometry drift magnitude rather than true goal correction. The method relies on the assumption that both depth and odometry errors remain small relative to the task precision; when either assumption fails, the correction signal degrades gracefully (no catastrophic divergence) because the goal still moves smoothly with the agent. The confidence check on initial depth explicitly acknowledges the depth assumption: if initial depth is uncertain, we avoid re-projection entirely, so p_t measures the open-loop pixel goal instead – a conservative fallback that ensures no false correction.

## Contribution

(1) A closed-loop goal re-projection mechanism that lifts static pixel goals to 3D and re-projects them via visual odometry, enabling online error correction for pixel-goal-based VLN agents without additional tracking or verification modules. (2) An empirical demonstration that this geometric correction significantly reduces cumulative navigation error in long-horizon tasks compared to the static goal baseline of ABot-N1, especially under execution drift. (3) A lightweight integration blueprint showing how monocular depth estimation (DepthAnything) and real-time visual odometry (DPVO) can be combined with existing action experts while preserving real-time performance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | VLN-CE (R2R-CE) | Standard continuous VLN with photo-realistic scenes |
| Primary metric | Success Rate (SR) | Measures task completion directly |
| Additional metrics | SPL, Navigation Error (NE) | Captures path efficiency and localization accuracy |
| Baseline 1 | ABot-N1 (original) | Base action expert without geo-correction |
| Baseline 2 | HAMT | Transformer agent beating recurrent models |
| Baseline 3 | VLN-BERT | BERT-based agent with history modeling |
| Ablation 1 | Geo-Ref w/o odometry | Tests importance of pose updates (uses fixed p_0) |
| Ablation 2 | Geo-Ref w/o depth confidence | Always uses initial X, no fallback |
| Statistical testing | Paired t-test over 5 seeds, p<0.05 | Ensures gains are significant |

### Why this setup validates the claim

The combination of VLN-CE, a realistic 3D benchmark, with baselines spanning pixel-goal (ABot-N1) and learned (HAMT, VLN-BERT) agents, isolates the effect of Geo-Ref's re-projection. ABot-N1 directly tests whether online correction improves over a fixed goal. HAMT and VLN-BERT test whether lightweight geometric consistency can compete with complex reasoning. Adding SPL and NE metrics provides a more complete picture of navigation quality beyond binary success. The ablation w/o odometry isolates the role of visual odometry, and the ablation w/o depth confidence isolates the importance of the confidence fallback. Statistical testing over multiple seeds ensures observed differences are not due to chance. We also perform a quantitative error propagation analysis on a subset of 200 randomly sampled VLN-CE episodes where ground-truth depth and pose are available (via simulator export). We measure the angular error between the re-projected pixel and the true goal pixel as a function of depth bias (0-20%) and odometry drift (0-10 cm per meter). This analysis validates that re-projection error grows linearly with depth/odometry errors and that the confidence fallback prevents large deviations when depth is unreliable. Code and pretrained models will be released at https://github.com/example/Geo-Ref.

### Expected outcome and causal chain

**vs. ABot-N1:** Geo-Ref is expected to improve SR by 3-5 points (from ~55% to ~59%) on VLN-CE val_unseen because online re-projection corrects drift, while ABot-N1 with a fixed goal accumulates error. The causal chain is: re-projection → consistent goal → smoother action selection → fewer collisions and incorrect turns.

**vs. HAMT:** Geo-Ref is expected to perform comparably or slightly better (SR ~59% vs ~58%) despite lacking explicit history encoding, because geometric consistency compensates for missing temporal reasoning. HAMT uses cross-attention over past frames; Geo-Ref uses only current re-projection, which is faster but may miss global context.

**vs. VLN-BERT:** Geo-Ref is expected to outperform VLN-BERT (SR ~59% vs ~56%) because VLN-BERT's single-step reasoning is more vulnerable to drift. The re-projection provides a continuous correction signal that is absent in VLN-BERT.

**Ablations:** Geo-Ref w/o odometry should match ABot-N1 (SR ~55%) demonstrating that pose updates are critical. Geo-Ref w/o depth confidence should show slightly degraded SR (~58%) due to occasional bad initial X, confirming that the confidence fallback prevents harmful corrections.

### What would falsify this idea
If Geo-Ref does not significantly outperform ABot-N1 (p>0.05) on SR, or if the ablation w/o odometry performs equally well, then the re-projection is not correcting drift effectively. Additionally, if the error propagation analysis shows that typical depth/odometry errors cause re-projection angular error > 30° for >20% of frames, the assumption of small errors is violated and the method would not be reliable.

## References

1. ABot-N1: Toward a General Visual Language Navigation Foundation Model
2. SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation
3. NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction
4. CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos
5. NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning
6. ImagineNav: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination
7. DINO-Foresight: Looking into the Future with DINO
8. DINOv2: Learning Robust Visual Features without Supervision

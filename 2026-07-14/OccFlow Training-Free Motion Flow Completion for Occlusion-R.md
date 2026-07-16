# OccFlow: Training-Free Motion Flow Completion for Occlusion-Resilient Character Motion Transfer

## Motivation

Motion4Motion and similar flow-based motion transfer methods assume clean visibility of the character throughout the video, but occlusions from scenes (e.g., passing behind objects) break optical flow extraction because flow methods cannot compute correspondences in occluded regions. Existing inpainting-based solutions require scene-specific training and are too heavy for training-free integration. This structural gap prevents flow-based transfer from operating in real-world cluttered scenes where partial occlusions are common.

## Key Insight

Motion flow in occluded character regions can be accurately inferred by propagating known flows from visible regions using Laplacian diffusion guided by image intensity edges and temporal warp consistency, without any learned scene model.

## Method

**(A) What it is**  
OccFlow is a training-free post-processing module. Input: coarse optical flow field F_c (from any method, e.g., RAFT), confidence map C, binary occlusion mask O (indicating character pixels occluded by scene), and the video frames I. Output: completed flow field F_complete for the character.  

**(B) How it works**  
```python  
def occ_flow(I, F_c, C, O, sigma_I=0.1, lambda_t=0.5, max_iter=50):  
    # Step 1: Build spatial weight matrix from intensity and confidence  
    W = compute_weight_matrix(I, O, sigma_I, C)  
    # Step 2: Initialize unknown flows (set to zero) and fix known flows  
    F = F_c.copy()  
    known = (O == 0) & (C > 0.1)  # high-confidence visible regions  
    unknown = (O == 1) | (C <= 0.1)  # occluded or low-confidence  
    F_prev = warp_flow(F_c, flow_from_previous_frame)  # temporal prior  
    # Step 3: Iterative Gauss-Seidel solver  
    for _ in range(max_iter):  
        for i in unknown indices:  
            F[i] = (sum_{j in N(i)} W[i,j] * F[j]  
                    + lambda_t * F_prev[i]) / (sum(W[i,j]) + lambda_t)  
    # Step 4: Smoothing with small median filter  
    F_complete = median_filter(F, kernel=3)  
    return F_complete  
```  
Hyperparameters: `sigma_I=0.1` (contrast threshold), `lambda_t=0.5` (temporal consistency weight), `max_iter=50`.  

**(C) Why this design**  
We chose Laplacian diffusion over learned inpainting networks because it requires no training data and generalizes to any character or scene, accepting that diffusion may blur across motion boundaries where intensity edges are weak. Color-guided weights preserve boundaries in textured regions but fail in uniform areas; we accept this trade-off because character motion boundaries often align with appearance edges. Temporal warp prior from the previous frame smooths flow over time but assumes constant velocity in occluded regions, which may fail for accelerating motion. Using confidence to mask known regions is necessary to avoid including erroneous flows, but confidence may be poorly calibrated; we mitigate by requiring a high threshold (0.1). The iterative solver is chosen for simplicity and deterministic output, at the cost of slower convergence than multigrid methods.  

**(D) Why it measures what we claim**  
The Laplacian diffusion step propagates known flows into occluded regions under the assumption that motion is piecewise smooth and aligned with image edges; this assumption fails when motion boundaries cross uniform textures (e.g., a black cat on black background), in which case completion blurs across the boundary. The temporal warp prior `F_prev` assumes that motion of occluded pixels is nearly constant between frames; this fails for accelerating or non-rigid motion, causing temporal jitter. The confidence mask `C` selects reliable known flows under the assumption that visible regions with moderate photometric consistency yield accurate flow; this fails when the entire character is occluded and no reliable data exists, making the completion degrade to pure diffusion from scene boundaries. Thus, the computational quantities directly operationalize the motivation-level concepts of motion smoothness, temporal coherence, and data reliability, with explicit failure modes tied to assumptions about texture, acceleration, and occlusion extent.

## Contribution

(1) A training-free motion flow completion module (OccFlow) that infers character flow in occluded regions by spatial Laplacian diffusion with color-guided weights and temporal warp prior. (2) An empirical finding that simple handcrafted priors suffice for occlusion handling in character motion transfer, eliminating the need for learned scene inpainting when occlusions are partial. (3) Integration into Motion4Motion framework enabling motion transfer under occlusions without retraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OccludedChar | Synthetic occlusions over moving characters. |
| Primary metric | Occluded region EPE | Measures flow accuracy in occluded areas. |
| Baseline 1 | Nearest neighbor fill | Simple baseline without diffusion. |
| Baseline 2 | Median fill | Smoothing baseline without edge-awareness. |
| Ablation-of-ours | OccFlow w/o temporal prior | Isolates effect of temporal warp. |

### Why this setup validates the claim

This setup directly tests the central claim that OccFlow improves flow completion in occluded regions through Laplacian diffusion with edge-aware weights and temporal prior. The dataset provides controlled occlusion scenarios where ground-truth flow is known, enabling precise evaluation. The primary metric isolates performance precisely where our method claims improvement. Baseline 1 (nearest neighbor) tests whether any filling beats no completion; baseline 2 (median fill) tests whether simple smoothness suffices. The ablation removes the temporal prior to measure its contribution. Together, these choices create a falsifiable test: if OccFlow outperforms both baselines and the ablation, the claim is supported, but uniform gain across subsets would indicate a trivial effect.

### Expected outcome and causal chain

**vs. Nearest neighbor fill** — On a case where a character moves behind a thin occluder (e.g., a pole), nearest neighbor copies a single flow vector from the boundary, causing severe discontinuity and large errors in the occluded region. Our method instead diffuses flows from multiple neighbors weighted by color similarity, producing a smooth interpolation that respects edges. Thus, we expect OccFlow to show a significant gap (e.g., ~30% lower EPE) on such instances, but near parity on textureless backgrounds.

**vs. Median fill** — On a case where motion boundaries align with strong intensity edges (e.g., a person with striped shirt), median fill blindly averages across the boundary, blurring flows across regions with different motions. Our method uses color-guided weights to preserve the boundary, maintaining separate motion estimates. Hence, we expect OccFlow to perform substantially better on high-contrast motion boundaries (e.g., ~40% lower EPE), but similar on uniform regions.

**vs. OccFlow w/o temporal prior** — On a case with accelerating motion (e.g., a suddenly braking car), the temporal prior provides a default flow from the previous frame, reducing errors from purely spatial diffusion that would lag behind. Without it, the spatial-only completion may produce temporal jitter. Thus, we expect the full model to have smoother flow over time (e.g., 20% lower temporal EPE variance) at the cost of slightly worse performance on constant motion cases.

### What would falsify this idea

If OccFlow’s improvement over baselines is uniform across all subregions (rather than concentrated on occlusion boundaries with strong edges), or if the ablation without temporal prior performs similarly to the full model on accelerating motion, then the claim that edge-awareness and temporal prior are critical would be falsified.

## References

1. Motion4Motion: Motion Transfer Across Subjects at Inference
2. MotionStream: Real-Time Video Generation with Interactive Motion Controls
3. MotionV2V: Editing Motion in a Video
4. Motion2Motion: Cross-topology Motion Transfer with Sparse Correspondence
5. Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance
6. Motion Prompting: Controlling Video Generation with Motion Trajectories
7. Pose‐to‐Motion: Cross‐Domain Motion Retargeting with Pose Prior
8. Skinned Motion Retargeting with Dense Geometric Interaction Perception

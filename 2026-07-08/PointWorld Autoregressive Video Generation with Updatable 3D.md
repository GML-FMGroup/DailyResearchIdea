# PointWorld: Autoregressive Video Generation with Updatable 3D Point Cloud Memory

## Motivation

Existing autoregressive video models lack persistent spatial memory, leading to drift and inconsistency over long horizons. For example, VideoSSM uses a state-space model for global memory but does not enforce explicit 3D geometry, causing object positions to diverge across frames. The root cause is the absence of a structured, updatable spatial representation that can be queried at each generation step to maintain global coherence.

## Key Insight

An explicit 3D point cloud, anchored in world coordinates and locally updated via per-frame depth fusion, provides a persistent geometric scaffold that prevents global drift because each point's position is globally consistent and can be corrected via image-based observations.

## Method

(A) **What it is**: PointWorld is an autoregressive video generation framework that maintains a dynamically updatable 3D point cloud memory \( P_t \) at each step \( t \). Its inputs are the previous frame \( I_{t-1} \) and an optional action \( a_t \); outputs are the next frame \( I_t \) and the updated memory \( P_t \).

(B) **How it works**:
The core operation follows three phases:
1. **Query**: Project the current point cloud \( P_{t-1} \) into the camera view using known (or estimated) pose, yielding a point cloud feature map \( F_{t-1} \) (e.g., depth, RGB features via splatting with a Gaussian kernel of radius 2 and a threshold for occluded points).
2. **Generate**: A transformer decoder with 12 causal layers and 8 attention heads takes the latent encoding \( z_{t-1} \) from the previous frame (via a 2D CNN encoder) and \( F_{t-1} \) as cross-attention conditioning to produce the next frame \( I_t \) via continuous regression (L2 loss on pixel values, as in NOVA).
3. **Update**: From \( I_t \), estimate depth (using a pretrained MiDaS network) and back-project to get 3D points with confidence \( c \). Fuse these new points into \( P_{t-1} \) using a Kalman-style rule: add new points if their confidence > 0.5; update existing points (within 0.1m) via a weighted average where weight = \( 0.7 \cdot c \); remove points not observed for 5 consecutive frames.

(C) **Why this design**:
We chose a 3D point cloud over a dense voxel grid because point clouds are sparse and efficient to update, accepting the cost of irregular structure requiring specialized projection and fusion operations. We chose per-step update rather than periodic resets to ensure memory freshness aligns with video dynamics, accepting the computational overhead of depth estimation each step. We chose to query by projection into the current view (rather than implicit cross-attention to unbounded 3D points) because it provides a direct geometric alignment, accepting that the projection may suffer from occlusions and require a visibility check. Compared to VideoSSM's hybrid memory, our explicit 3D representation allows direct geometric consistency, which a domain expert would not describe as a variant of SSM-based memory because SSMs do not impose spatial constraints.

(D) **Why it measures what we claim**:
The frequency of point cloud updates (every frame) directly measures the system's ability to incorporate new observations, operationalizing spatial coherence as the consistency between the projected memory and newly generated pixels. The assumption is that depth estimates from \( I_t \) are accurate enough to correctly place new points; when depth is noisy (high motion or low texture), the metric reflects point placement error rather than global drift. The projection operation aligns the world points with image coordinates; this alignment quantifies geometric consistency under the assumption that the camera pose is known or fixed; if pose drifts, the metric captures hybrid pose-drift+geometry error instead of pure spatial coherence. The removal of unobserved points after 5 frames operationalizes temporal persistence; when scenes change abruptly, the metric may reflect outdated points still present for 5 frames, indicating memory lag rather than incoherence.

## Contribution

(1) A novel autoregressive video generation framework that maintains an explicit, updatable 3D point cloud memory for persistent spatial coherence over long horizons. (2) A memory update mechanism based on per-frame depth estimation and Kalman-style fusion that balances insertion, update, and removal of points to reflect dynamic scenes. (3) Empirical demonstration that explicit 3D memory reduces long-horizon drift compared to prior implicit memory approaches (e.g., VideoSSM) on complex 3D scenes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Minecraft | Rich 3D structure with motion |
| Primary metric | FVD | Captures distributional video quality |
| Baseline 1 | VideoSSM | Hybrid memory without explicit 3D |
| Baseline 2 | FastAutoregressiveDiffusion | Strong diffusion baseline |
| Ablation-of-ours | Static point cloud memory | Tests update necessity |

### Why this setup validates the claim
This combination directly tests the central claim that explicit 3D point cloud memory with per-frame update improves long-term spatial coherence. VideoSSM serves as a baseline with implicit memory, isolating the benefit of explicit geometric representation. FastAutoregressiveDiffusion tests against a powerful diffusion model that lacks persistent world state, assessing the advantage of autoregressive memory. The static-point-cloud ablation isolates the effect of the update mechanism. FVD is the right metric because it measures both visual fidelity and temporal consistency, so a significant gap on long videos would confirm the predicted advantage of explicit 3D memory. If the advantage is missing, the claim is falsified.

### Expected outcome and causal chain

**vs. VideoSSM** — On a case where the camera moves through a long corridor, VideoSSM's SSM memory drifts over time because it lacks geometric anchoring, producing blurry and inconsistent long-term appearance. Our PointWorld maintains explicit 3D points, preserving exact locations across frames, so it retains sharp textures and consistent object positions. We expect a clear FVD gap favoring PointWorld on long videos (>100 frames), especially in scenes with repetitive textures where memory drift is detrimental.

**vs. FastAutoregressiveDiffusion** — On a case with fast object motion and occlusion, the diffusion baseline generates plausible single frames but fails to maintain object identity after occlusion because it has no persistent memory. Our method updates the point cloud with depth and confidence, allowing re-identification of objects after occlusion via point persistence. Expect higher temporal consistency scores (e.g., CLIP-based) and lower FVD on sequences with occlusions, while performance matches on static scenes.

### What would falsify this idea
If PointWorld's gains over VideoSSM are uniform across short and long videos, or if the static memory ablation achieves comparable FVD, then the central claim that explicit 3D point cloud update drives long-term coherence is falsified.

## References

1. AlayaWorld: Long-Horizon and Playable Video World Generation
2. VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory
3. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
4. Autoregressive Video Generation without Vector Quantization

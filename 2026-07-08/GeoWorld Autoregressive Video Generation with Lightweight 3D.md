# GeoWorld: Autoregressive Video Generation with Lightweight 3D Geometric Consistency

## Motivation

Autoregressive video world models like AlayaWorld generate temporally coherent frames but lack explicit 3D geometric consistency, leading to object distortions and drift over long horizons. Prior work removes 3D-aware attention (e.g., VideoSSM) to reduce complexity, but no alternative enforces geometric constraints. This gap stems from the difficulty of integrating 3D reasoning into fast autoregressive frameworks without heavy 3D reconstruction.

## Key Insight

By modeling each frame's depth as a lightweight point cloud memory that is updated autoregressively and enforcing a reprojection photometric consistency loss between frames, geometric coherence is maintained without expensive 3D scene decomposition because the loss penalizes violations of the underlying 3D structure captured by the memory.

## Method

(A) **What it is**: GeoWorld extends an autoregressive video generation model with a lightweight point cloud memory that stores a sparse set of 3D points representing the scene geometry, and a reprojection-based geometric consistency loss during training. Inputs: sequence of generated frames and actions; outputs: updated point cloud and new frame. (B) **How it works**:

```python
# Pseudocode for GeoWorld training loop
# Hyperparameters: λ=0.3, memory_size=2048, window_k=10, depth_predictor='DPT' (MiDaS)

for each training step:
    # Let I_t be the generated frame at time t (from autoregressive model), a_t be action
    # Obtain depth map D_t from frozen depth predictor (e.g., DPT)
    D_t = depth_predictor(I_t)
    
    # Back-project to 3D point cloud P_t using camera intrinsics (assumed pinhole)
    P_t = back_project(I_t, D_t, K)
    
    # Update memory M_t by merging P_t with M_{t-1} (only keep points from last window_k frames)
    # Downsample to memory_size using voxel grid filter
    M_t = downsample_voxel_grid(merge(M_{t-1}, P_t, window_k), memory_size)
    
    # Compute geometric consistency loss
    L_geo = 0
    for each pixel p in I_t with depth d:
        # Reproject p to previous frame using M_t and D_t
        p_reproj = reproject(p, d, M_t, K, I_{t-1})
        if p_reproj is within bounds:
            L_geo += L1_loss(I_t[p], I_{t-1}[p_reproj])
    L_geo = L_geo / num_valid_pixels
    
    # Total loss: L = L_pred + λ * L_geo
    L_pred = autoregressive_prediction_loss(I_t, I_{t-1}, a_t)  # e.g., L2 on pixels
    L = L_pred + λ * L_geo
    # Backpropagate
```

(C) **Why this design**: We chose a simple point cloud memory over a full 3D reconstruction (e.g., NeRF) because the latter is too slow for autoregressive generation; the trade-off is less detailed geometry but sufficient for enforcing consistency. We selected voxel grid downsampling over random sampling to maintain spatial coverage. We used a frozen depth predictor to avoid training a depth module from scratch, accepting a domain gap but significantly reducing computational overhead. We added the consistency loss only during training, not inference, to avoid extra computation at test time. This design balances geometric awareness with the efficiency required for real-time interactive generation.

(D) **Why it measures what we claim**: The reprojection photometric loss L_geo measures geometric consistency between frames because it quantifies how well the generated frame I_t aligns with the previous frame I_{t-1} when projected through the 3D memory. The assumption is that the point cloud memory M_t and depth D_t accurately represent the underlying 3D scene. This assumption fails when the depth estimates are noisy or when there are occlusions (points not visible in previous frame), in which case the loss may penalize valid changes due to viewpoint or object motion, reflecting instead a depth estimation error or occluded area. The memory update mechanism ensures that only persistent geometric features are retained, mitigating occlusion effects.

## Contribution

(1) A lightweight geometric consistency loss for autoregressive video generation that enforces 3D coherence without full scene reconstruction. (2) A simplified point cloud memory that updates autoregressively, serving as a shared geometric state to anchor frame generation. (3) Integration of these components into a state-of-the-art autoregressive framework (AlayaWorld), demonstrating improved long-horizon geometric stability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RealEstate10K | indoor static, camera motion, known intrinsics |
| Primary metric | Temporal Consistency Error (TCE) | measures per-pixel warping error via RAFT |
| Baseline 1 | VideoFusion | autoregressive baseline without geometric memory |
| Baseline 2 | VideoSSM | hybrid state-space memory but no explicit geometry |
| Baseline 3 | LVDM | diffusion model, strong performance baseline |
| Ablation-of-ours | GeoWorld w/o consistency loss | isolates effect of geometric loss |

### Why this setup validates the claim
This setup forms a falsifiable test because RealEstate10K provides static indoor scenes with known camera intrinsics and pose, so geometric consistency should dominate temporal errors. TCE directly measures pixel misalignment after warping with optical flow, capturing geometric inconsistencies independently of our depth predictor. VideoFusion lacks any memory, VideoSSM uses latent memory without 3D structure, and LVDM avoids autoregressive constraints entirely; success against all three would demonstrate that explicit geometric memory and loss are beneficial. The ablation removes the consistency loss, isolating the contribution of the point cloud memory. If GeoWorld outperforms all baselines on TCE while maintaining comparable FVD, the central claim is supported. Failure to beat the ablation on TCE would show the memory alone is insufficient.

### Expected outcome and causal chain

**vs. VideoFusion** — On a sharp camera rotation, VideoFusion produces object sliding or distortion because it has no memory of previous geometry, so it hallucinates new appearances. Our method uses the point cloud memory to reproject pixels from the previous frame, enforcing that surfaces remain consistent. We expect a noticeable TCE gap on high-motion segments (e.g., rotations > 10°) but parity on static segments.

**vs. VideoSSM** — When a previously occluded area is revealed (e.g., a pillar passes out of view), VideoSSM generates blurry or incoherent content because its latent memory cannot recover exact pixel correspondences. Our method reprojects from the point cloud, which stores the occluded points from earlier frames, so the revealed area is sharp and consistent. We expect TCE on occlusion-reveal frames to be significantly lower for GeoWorld.

**vs. LVDM** — In long sequences (100+ frames), LVDM suffers from flicker and temporal inconsistency because its diffusion process does not enforce strict causality. Our autoregressive approach with geometric loss maintains frame-to-frame stability. We expect GeoWorld to achieve lower TCE on long sequences, with the gap widening as sequence length increases.

### What would falsify this idea
If GeoWorld's TCE improvement over baselines is uniform across all frames rather than concentrated on high camera motion or occlusion-reveal segments, then the geometric loss is not effectively leveraging 3D structure, and the central claim is wrong. Conversely, if the ablation (without loss) matches GeoWorld, then the point cloud memory alone is sufficient, undermining the need for the reprojection loss.

## References

1. AlayaWorld: Long-Horizon and Playable Video World Generation
2. VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory
3. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
4. Autoregressive Video Generation without Vector Quantization

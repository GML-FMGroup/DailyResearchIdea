# OmniRot: Spherical Harmonic Memory for 6-DOF Panoramic Video Generation

## Motivation

PanoWorld's DPRC and GMA assume fixed heading, limiting camera motion to translation-only trajectories. When rotation occurs, spatial-domain memory re-projection introduces interpolation artifacts and breaks temporal consistency because the memory representation lacks rotation equivariance. This structural limitation prevents existing methods from handling arbitrary 6-DOF camera motion without retraining.

## Key Insight

Spherical harmonics are eigenfunctions of rotation operators in the frequency domain, enabling exact and interpolation-free rotation of memory via Wigner D-matrix multiplication.

## Method

**A) What it is**  
OmniRot introduces a Spherical Harmonic Memory Augmentation (SHMA) module that replaces PanoWorld's geometry-aware memory (GMA). The module stores memory as spherical harmonic (SH) coefficients and rotates them exactly via Wigner D-matrices when the camera rotates, then synthesizes a spatial memory signal for injection into the diffusion process. Input: current frame, camera pose, previous SH memory. Output: memory-augmented features.  

**B) How it works**  
```python
def shma_step(frame_t, pose_t, prev_mem_coeffs):
    # prev_mem_coeffs: dict {l: matrix of shape (2l+1, feature_dim)}
    # pose_t: (R, t) from camera intrinsic & extrinsic
    # Step 1: Compute relative rotation
    R_rel = pose_t.R @ prev_pose.R.T
    # Step 2: Rotate SH memory per band via Wigner D-matrix
    new_mem_coeffs = {}
    for l in range(0, L_max+1):
        D_l = wigner_D_matrix(l, R_rel)  # (2l+1 x 2l+1)
        new_mem_coeffs[l] = D_l @ prev_mem_coeffs[l]  # exact rotation
    # Step 3: Spatially reconstruct memory signal (inverse SH transform)
    mem_signal = inverse_sh_transform(new_mem_coeffs, grid_shape=(H,W))
    # Step 4: Inject into diffusion via cross-attention
    F_t = diffusion_block(concat(frame_t, mem_signal))
    # Step 5: Update memory with new content (exponential moving average)
    current_coeffs = forward_sh_transform(frame_features, L_max)
    alpha = 0.9
    updated_mem = {l: alpha*new_mem_coeffs[l] + (1-alpha)*current_coeffs[l] for l in range(L_max+1)}
    return F_t, updated_mem
```
Hyperparameters: L_max=8 (band limit), alpha=0.9 (EMA rate).  

**C) Why this design**  
We chose SH representation over spatial grids because rotation equivariance is exact in the frequency domain, eliminating interpolation artifacts that plague spatial re-projection (e.g., bilinear interpolation smears features). We chose Wigner D-matrices over Euler-angle re-projection because matrix multiplication is differentiable and O(L^3) per band, which is more efficient than O(N^2) spatial resampling for typical grid sizes (e.g., 512x1024). We chose exponential moving average (EMA) for memory update over a learned gate because it requires no extra parameters and ensures temporal smoothness, but accepts that fast scene changes (e.g., sudden new objects) may be blended slowly due to the fixed decay factor. We chose L_max=8 as a trade-off between spatial resolution (band limit) and computational cost; higher L_max captures finer details but increases memory and D-matrix computation quadratically.  

**D) Why it measures what we claim**  
The SH coefficient tensor new_mem_coeffs[l] measures rotation-equivariant memory content because spherical harmonics form an orthogonal basis for functions on the sphere, and the Wigner D-matrix for each band l is the irreducible representation of SO(3) on that subspace; this assumes the memory signal is band-limited with maximum frequency L_max. When the true signal contains components above L_max, those components are discarded, causing aliasing; in that case the memory reflects only low-frequency structure, potentially missing fine edges or textures. The inverse SH transform mem_signal measures the spatially mapped memory under the assumption that the SH series converges to the original function; this assumption fails when the function has discontinuities (e.g., sharp occlusions), producing Gibbs oscillations. The EMA update alpha measures temporal consistency because it enforces a smooth transition between frames under the assumption of slow scene dynamics; if the scene changes abruptly, the memory lags behind, reflecting outdated content rather than current reality.

## Contribution

(1) A frequency-domain memory representation for panoramic video generation that uses spherical harmonic coefficients to enable exact and interpolation-free rotation via Wigner D-matrices. (2) The design of a memory update mechanism (EMA with SH transform) that preserves temporal consistency under 6-DOF rotation without retraining, directly addressing the fixed-heading limitation of prior work like PanoWorld. (3) A demonstration that replacing spatial-domain memory with SH memory improves rotation consistency without additional trainable parameters.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | World360 | Standard real-world panoramic video dataset. |
| Primary metric | FID (Frechet Inception Distance) | Measures visual quality of generated frames. |
| Baseline | PanoWorld (GMA) | Replaces our SH memory exactly. |
| Baseline | No memory baseline | No temporal memory, tests memory necessity. |
| Baseline | Spatial grid memory with bilinear sampling | Interpolation artifacts test. |
| Ablation-of-ours | OmniRot with L_max=4 | Tests if higher L_max improves fine details. |

### Why this setup validates the claim
This experimental design tests the central claim that spherical harmonic memory with exact rotation via Wigner D-matrices eliminates interpolation artifacts and provides rotation-equivariant memory for panoramic video generation. By comparing against PanoWorld's geometry-aware memory (GMA), we isolate the benefit of SH representation over learned spatial memory. The no-memory baseline establishes the necessity of any temporal memory, while the spatial grid baseline with bilinear sampling directly tests the interpolation artifact reduction claim. World360 provides diverse real-world panoramas with camera motion, and FID captures visual fidelity. The ablation with lower L_max tests whether the band limit affects fine details, confirming that SH memory's high-frequency retention is crucial. Together, these comparisons create a falsifiable test: if our method outperforms spatial grid on rotational scenes but not static ones, the rotation equivariance claim is supported.

### Expected outcome and causal chain

**vs. PanoWorld (GMA)** — On a case where the camera rapidly pans across a highly textured wall, PanoWorld's GMA introduces ghosting or smearing because its spatial memory re-projection via bilinear interpolation distorts features, especially at large rotations. Our method instead preserves texture sharpness because SH rotation is exact and aliasing-free within the band limit, so we expect a noticeable FID gap on sequences with frequent large rotations but parity on sequences with minimal rotation.

**vs. No memory baseline** — On a case where the camera slowly orbits a static scene with consistent lighting, the no-memory baseline generates each frame independently, leading to temporal flickering and inconsistency because it lacks any prior context. Our method uses EMA-updated SH memory to enforce temporal smoothness, producing stable illumination across frames, so we expect a large FID improvement on static scenes but similar performance on dynamic scenes where memory lags.

**vs. Spatial grid memory with bilinear sampling** — On a case where the camera rotates by 90 degrees, the spatial grid memory resamples features via bilinear interpolation, causing softening of edges and loss of high-frequency details. Our SH memory rotates coefficients exactly, preserving fine details up to L_max=8, so we expect a clear FID advantage on high-texture scenes and under large rotations, but parity on small rotations where interpolation errors are minimal.

### What would falsify this idea
If the FID of OmniRot is no better than the spatial grid baseline on sequences with large rotations, or if the gain is uniform across all subsets (including static scenes), then the rotation-equivariance advantage is not realized, falsifying the central claim.

## References

1. PanoWorld: Real-World Panoramic Generation
2. Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model
3. CamPVG: Camera-Controlled Panoramic Video Generation with Epipolar-Aware Diffusion
4. Unified Camera Positional Encoding for Controlled Video Generation
5. DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
6. CamI2V: Camera-Controlled Image-to-Video Diffusion Model
7. DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion
8. CamCo: Camera-Controllable 3D-Consistent Image-to-Video Generation

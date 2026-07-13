# EquiPano: Rotation-Equivariant Panoramic Video Generation for Full 6-DOF Camera Control

## Motivation

Existing panoramic video generation methods, such as PanoWorld, assume translation-only camera motion with a fixed heading, precluding realistic 6-DOF camera control. This structural limitation arises because their ray-conditioned generation and memory modules are not rotation-equivariant—they treat the camera's orientation as absolute, causing inconsistencies when the camera rotates. Consequently, arbitrary rotations break the model's internal representation, leading to artifacts or loss of control. To enable full 6-DOF control, the generative model must be equivariant under SO(3) rotations of the camera pose, ensuring that any rotation simply transforms the generated panorama by the same rotation.

## Key Insight

Rotations of the camera pose are a group action on the space of panoramic images, and by designing the generative model's architecture to be equivariant under this group, we guarantee that the model's output transforms consistently under arbitrary rotations, enabling seamless 6-DOF control without additional supervision.

## Method

**EquiPano: Rotation-Equivariant Panoramic Video Generation**

**(A) What it is:** EquiPano is a diffusion-based panoramic video generation model that processes camera rays in a rotation-equivariant manner. Its inputs are a sequence of camera poses (6-DOF: position and orientation) and a latent noise tensor; its output is a sequence of panoramic frames consistent with the given camera motion.

**(B) How it works:** Core operation: the denoising U-Net uses spherical harmonic (SH) features to encode ray directions and group-equivariant convolutions on the sphere. Let the camera orientation at frame t be given by rotation matrix R_t ∈ SO(3). For each pixel with spherical direction ω, we compute its rotated direction ω' = R_t · ω. This is used to index into a spherical grid processed by the network.

```python
# Pseudocode for one denoising step
def denoise_step(latent_z_t, camera_poses, t):
    # latent_z_t: [batch, frames, channels, height, width]
    # camera_poses: [batch, frames, 6] (translation + rotation matrix)
    for frame in range(frames):
        R = camera_poses[:, frame, 3:6]  # rotation matrix
        # Convert pixel coordinates to spherical directions (theta, phi)
        directions = pixel_to_sphere(width, height)
        # Apply rotation: new_directions = R @ directions
        new_dirs = apply_rotation(directions, R)
        # Sample spherical grid features using rotated directions (via spherical interpolation, e.g., HEALPix)
        grid_features = sample_spherical_grid(latent_z_t[:, frame], new_dirs)
        # Process with equivariant spherical convolutions (uses SO(3) convolution kernels)
        equiv_features = spherical_convolution(grid_features)
        # Update latent using U-Net with skip connections
        latent_z_t[:, frame] = update(latent_z_t[:, frame], equiv_features, t)
    return latent_z_t
```

The key hyperparameters: number of SH bands = 8, spherical grid resolution = 4×4k pixels (HEALPix NSIDE=32), number of equivariant layers = 6, kernel size = 3 (on the spherical manifold). Training uses the World360 dataset with synthetic camera trajectories that include arbitrary rotations.

**(C) Why this design:** We chose spherical harmonic features and group-equivariant convolutions over standard ray encodings (e.g., Plücker coordinates used in CamCo) because SH features are inherently equivariant to SO(3) rotations—rotating the input rotates the SH coefficients by a known matrix. The cost is increased computational complexity (SH transforms are O(L^3) per pixel) and the need for specialized spherical convolution libraries. We chose HEALPix as the spherical grid over equirectangular or icosahedral grids because it supports hierarchical, equi-area sampling and fast spherical harmonic transforms; the trade-off is that equirectangular grids are simpler but not uniform-area, causing oversampling at poles. We designed the network to process each frame independently with shared rotation parameters rather than using cross-frame epipolar attention (as in CamPVG) because epipolar constraints tie viewpoints together in a non-equivariant manner under rotation; the trade-off is losing explicit multi-view consistency, but we argue that rotation-equivariance implicitly enforces view consistency across frames since rotating the camera simply rotates the generated scene.

**(D) Why it measures what we claim:** The computational quantity `spherical_convolution(rotated_directions)` measures rotation-equivariance because it applies the same convolutional kernel to the rotated spherical signal as would be applied to the original signal rotated by the inverse transformation; this holds under the assumption that the spherical grid and convolution are SO(3)-equivariant. This assumption fails when the spherical grid is not uniform (e.g., equirectangular) or when the convolution is not strictly equivariant (e.g., uses fixed kernels that don't rotate), in which case the output distorts under rotation. The `sample_spherical_grid` operation measures orientation-agnostic feature extraction because it uses the camera rotation to re-index the same latent features; this assumes the latent features are stored in a sphere-anchored coordinate system. That assumption fails when the latent features are not rotationally invariant (e.g., they contain orientation-specific content), in which case the generated panorama changes with orientation inconsistently. The overall U-Net update measures physically consistent generation because the equivariance ensures that rotating the input pose rotates the output panorama by the same amount, preserving scene geometry.

## Contribution

(1) A rotation-equivariant spherical ray encoding and U-Net architecture that processes panoramic rays using spherical harmonics and group-equivariant convolutions, enabling full 6-DOF camera control. (2) A demonstration that equivariance enforces geometric consistency under arbitrary rotations without explicit multi-view constraints, improving generation quality over non-equivariant baselines. (3) A training and evaluation protocol for panoramic video generation with diverse 6-DOF camera trajectories, built on the World360 dataset.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | World360 dataset | Contains diverse camera rotations |
| Primary metric | Equivariance Gap (EG) | Measures rotation-equivariance directly |
| Baseline 1 | PanoWorld | Standard panoramic generation without equivariance |
| Baseline 2 | CamPVG | Epipolar-aware, not rotation-equivariant |
| Ablation of ours | EquiPano w/o equivariant convolutions | Replace SH with equirectangular features |

### Why this setup validates the claim

The central claim is that EquiPano produces rotation-equivariant panoramic videos. To test this, we need a dataset with known camera rotations (World360) and a metric that quantifies how well the output rotates with the input (Equivariance Gap). Comparing against PanoWorld and CamPVG, which lack explicit rotation-equivariance, isolates the benefit of our design. The ablation removes equivariant convolutions to verify that the improvement stems from that component. If our method achieves significantly lower EG under large rotations while baselines fail, then the claim is supported.

### Expected outcome and causal chain

**vs. PanoWorld** — On a case where the camera rotates 90° laterally, PanoWorld produces scene collapse (e.g., objects warp or vanish) because it treats each frame independently with no rotation awareness. Our method instead rotates the spherical latent features via equivariant convolutions, preserving geometry. We expect EG < 0.1 for our method and EG > 0.5 for PanoWorld on such sequences.

**vs. CamPVG** — On a case where the camera rotates while translating, CamPVG’s epipolar attention fails to maintain consistent features under pure rotation, causing temporal discontinuities. Our method’s spherical harmonics and HEALPix grid ensure that rotating the input identically rotates the output, leading to smooth transitions. We expect CamPVG’s EG to be around 0.3 while ours remains below 0.1.

**vs. EquiPano w/o equivariant convolutions** — On a rotation where the camera tilts toward the north pole, the ablation (equirectangular grid) produces severe distortion and content drift due to non-uniform sampling. Our full method with HEALPix and SH coefficients avoids this. We expect the ablation’s EG to exceed 0.4, while ours stays low.

### What would falsify this idea

If our full method achieves an Equivariance Gap that is not significantly lower than baselines on high-rotation subsets, or if the gain is uniform across all rotation angles rather than concentrated on large rotations, then the claimed rotation-equivariance mechanism is not functioning as intended.

## References

1. PanoWorld: Real-World Panoramic Generation
2. Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model
3. CamPVG: Camera-Controlled Panoramic Video Generation with Epipolar-Aware Diffusion
4. Unified Camera Positional Encoding for Controlled Video Generation
5. DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
6. CamI2V: Camera-Controlled Image-to-Video Diffusion Model
7. DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion
8. CamCo: Camera-Controllable 3D-Consistent Image-to-Video Generation

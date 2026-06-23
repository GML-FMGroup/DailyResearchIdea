# DynaSynth: Temporally Consistent Novel View Synthesis from a Single Image via 4D Neural Representation

## Motivation

Existing single-image novel view synthesis methods (e.g., SceneCompleter) assume a static scene, leading to artifacts when motion is present. Methods that handle dynamics require multi-view video input, limiting applicability. We address this by learning a 4D representation from single images and video that generates plausible dynamic novel views from a static input.

## Key Insight

Natural motions are governed by physical priors (smoothness, periodicity) that can be compactly captured by a low-frequency temporal basis, enabling plausible motion inference from a single image via learned prior over video data.

## Method

### (A) What it is:
DynaSynth is a neural 4D representation (3D space + time) that, given a single RGB image, outputs a time-varying radiance field enabling novel view synthesis at arbitrary viewpoints and times. Input: a single image I. Output: rendered views at requested camera pose and time t ∈ [0,1].

### (B) How it works (pseudocode):
```python
# Forward pass of DynaSynth

# 1. Encode static scene from input image
static_features = ViT_Encoder(I)  # (H/16, W/16, D=64) feature grid in canonical space

# 2. Map to 3D voxel grid (e.g., 32x32x32)
voxel_feats = neck(static_features)  # (32,32,32,128) per-voxel static features

# 3. Predict temporal basis coefficients for each voxel
coeffs = MLP(voxel_feats)  # (32,32,32, K=16) coefficients for K sinusoidal bases

# 4. Define temporal basis functions (fixed, non-learned):
basis[k](t) = sin(2π * f_k * t + φ_k), where f_k = uniform in [0.5, 4] Hz, φ_k = 0
# Load-bearing assumption: natural motions are well-approximated by a sum of low-frequency sinusoids (0.5–4 Hz).
# Calibration: For each training video, compute the temporal power spectrum of optical flow in a central region.
# Verify that >90% of power lies within 0.5–4 Hz. If not, discard that video or flag as failure case.

# 5. Compute time-varying features for a given time t:
def get_feature(x, t):
    # trilinear interpolation of coeffs at 3D position x
    c = trilinear_interp(coeffs, x)  # (K,)
    b = [basis[k](t) for k in range(K)]  # (K,)
    modulation = c · b  # scalar (or vector via per-dim coefficients, simplified)
    static_feat = trilinear_interp(voxel_feats, x)  # (128,)
    return static_feat + modulation * static_feat  # modulate static features

# 6. Render: volume render with a small MLP decoder (2-layer MLP, hidden=256, GeLU activation) that outputs color and density from time-varying features
def render(rays, t):
    colors = []
    for ray in rays:
        points = ray.sample_points(64)
        feats = [get_feature(p, t) for p in points]
        densities, rgbs = decoder(feats)  # MLP: (128) → (σ, rgb)
        color = volume_rendering(densities, rgbs)
        colors.append(color)
    return colors

# Training: For each training video clip, pick one reference frame and other frames at times t_i.
# Supervise rendered images at those times with L1 + perceptual (LPIPS) + temporal smoothness loss on coeffs
# Temporal smoothness: L_temporal = ||coeffs - lowpass_filtered(coeffs)||_2  (encourages low-freq modulation)
# Hyperparameters: learning rate 1e-4, batch size 4 rays per frame, Adam optimizer, 100k iterations.
# GPU memory: ~12 GB (RTX 3090), training time ~48 hours.
```

### (C) Why this design:
We chose a 4D feature grid over explicit deformation fields because (1) it avoids the difficulty of predicting dense correspondences between frames, which is ill-posed from a single image; (2) the sinusoidal basis decomposition enforces a physically plausible prior (natural motions are often quasi-periodic and smooth) without requiring explicit dynamics; (3) the encoder-decoder architecture allows generalization to novel scenes. We chose sinusoidal bases over MLP-predicted temporal weights because the fixed frequencies provide a natural frequency decomposition that matches known motion statistics (low-frequency dominates), sacrificing ability to capture high-frequency motion like abrupt changes. We chose to condition temporal modulation on static features rather than encoding image directly into 4D features because it separates static appearance from dynamics, reducing data requirements and enabling fine-tuning of static reconstruction; the cost is that strongly shape-dependent motions (e.g., a waving arm) may be partially entangled with static geometry.

### (D) Why it measures what we claim:
The temporal basis coefficients measure plausible motion at each voxel because they are trained to reconstruct actual video frames; the assumption is that natural motions are well-approximated by a sum of low-frequency sinusoids; this assumption fails for abrupt or stochastic motions (e.g., lightning flash), in which case reconstruction becomes blurry. The temporal smoothness loss measures physical realism because it penalizes high-frequency modulation that lacks physical cause (e.g., flickering); it reflects temporal consistency under the assumption that natural motion is continuous, but fails when motion is truly impulsive (e.g., explosion). The static feature grid measures scene geometry and appearance because it is optimized to reconstruct the reference image; it encodes 3D information via the ViT's cross-view reasoning, but for occluded regions the features become uncertain, leading to hallucinated motion in those areas. LPIPS (per-image) measures perceptual quality of individual frames but does not guarantee temporal smoothness: a sharp but jittery video can have low LPIPS. To address this, we also report warping error (mean absolute difference between consecutive frames after optical flow warping) as a temporal consistency metric, and we discuss the assumption that low per-image LPIPS implies temporal smoothness is not always valid.

## Contribution

(1) We introduce DynaSynth, a neural 4D representation that enables temporally consistent novel view synthesis from a single static image. (2) We propose a physically-motivated temporal basis decomposition that allows learning motion priors from video data without explicit correspondence, generalizing across diverse scenes. (3) We demonstrate that the learned motion prior produces plausible dynamic novel views, outperforming static-only baselines in temporal consistency.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | RealEstate10K | Real-world videos with camera poses |
| Primary metric | LPIPS | Perceptual quality of rendered views |
| Secondary metric | Warping error (optical flow warping) | Temporal consistency between consecutive frames |
| Baseline 1 | PixelNeRF | Static baseline, no temporal modeling |
| Baseline 2 | IBRNet | Transformer-based static NVS |
| Baseline 3 | SVD + PixelNeRF | Video generation plus static NVS |
| Ablation-of-ours | Ours w/o temporal modulation | Isolate benefit of sinusoidal basis |

### Why this setup validates the claim
This experimental design directly tests the central claim that DynaSynth can generate plausible 4D (3D+time) radiance fields from a single image. PixelNeRF and IBRNet represent the static-only hypothesis—if they match or exceed DynaSynth, dynamics are unnecessary. SVD+PixelNeRF tests whether independent temporal and view synthesis can match an integrated approach. The ablation isolates the contribution of the sinusoidal temporal basis. LPIPS captures both spatial and temporal perceptual artifacts, making it sensitive to the predicted failure modes (e.g., blur, flicker) but only per-frame. Adding warping error evaluates temporal consistency directly, addressing the gap between per-image metrics and temporal smoothness. RealEstate10K provides realistic camera motions and dynamic content (e.g., people moving), enabling a falsifiable test: if DynaSynth's gain over static baselines is uniform rather than concentrated on moving regions, the temporal modeling is not actually leveraging the sinusoidal prior. Additionally, we perform a frequency analysis on the training videos: compute the temporal power spectrum of optical flow in a central 64x64 patch, and verify that >90% of power lies within 0.5–4 Hz. Videos failing this are reported separately as failure cases, defining the scope of applicability.

### Expected outcome and causal chain

**vs. PixelNeRF** — On a scene with a person waving their arm, PixelNeRF generates a static arm at the reference pose, producing a jarring still image at novel views or times. This failure occurs because PixelNeRF has no temporal mechanism; it treats the scene as permanently frozen. Our method instead reconstructs the arm's periodic motion via sinusoidal coefficients, resulting in smooth, believable waving. We expect a large LPIPS gap on dynamic regions (e.g., >0.1) but parity on static backgrounds. The warping error will be significantly lower for DynaSynth on dynamic scenes.

**vs. IBRNet** — On a scene with a flowing fountain, IBRNet, though using transformer features, still cannot change appearance over time; it produces a static water shape. This fails because IBRNet lacks any time input. Our method encodes the water's spatial variation in static features and temporal modulation in coefficients, generating time-varying water that periodically ripples. We expect a similar gap to PixelNeRF but potentially smaller due to IBRNet's stronger feature representation, yet still clearly above our method on perceptual quality for temporal frames.

**vs. SVD + PixelNeRF** — On a scene where a car drives past, SVD+PixelNeRF first hallucinates a video from the single image using a video diffusion model, then independently renders each frame with PixelNeRF. This fails because (1) SVD may change background between frames, (2) PixelNeRF's static reconstructions from different frames are inconsistent, leading to jitter. Our method instead jointly optimizes for consistent time-varying features, producing coherent motion. We expect our LPIPS and warping error to be lower than this pipeline on novel views over time.

### What would falsify this idea
If DynaSynth's improvement over static baselines is uniform across static and dynamic regions (i.e., no significant difference on motion-heavy subsets), or if the ablation without temporal modulation performs similarly on dynamic videos, then the sinusoidal prior is not capturing motion and the central claim of effective 4D generation from a single image is wrong. Furthermore, if the frequency analysis shows that most training videos have significant power above 4 Hz, the assumption underlying the basis choice is invalidated.

## References

1. SceneCompleter: Dense 3D Scene Completion for Generative Novel View Synthesis
2. latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction
3. CameraCtrl: Enabling Camera Control for Text-to-Video Generation
4. ReconX: Reconstruct Any Scene From Sparse Views With Video Diffusion Model
5. 3D Reconstruction with Spatial Memory
6. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
7. MagicVideo: Efficient Video Generation With Latent Diffusion Models
8. Latent Video Diffusion Models for High-Fidelity Long Video Generation

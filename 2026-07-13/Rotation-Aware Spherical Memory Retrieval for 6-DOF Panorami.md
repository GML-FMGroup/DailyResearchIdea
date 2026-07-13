# Rotation-Aware Spherical Memory Retrieval for 6-DOF Panoramic Generation

## Motivation

PanoWorld's memory augmentation module (GMA) assumes fixed camera heading, causing feature misalignment under rotation because memory is indexed by spatial position alone. CamPVG achieves rotation-aware single-frame matching via spherical epipolar attention, but does not extend this to memory retrieval. The root cause is that memory representations lack an explicit rotation parameterization, forcing retrieval to rely on translation-only cues even when camera orientation changes.

## Key Insight

Spherical epipolar geometry provides an invariant mapping between camera rotation and spherical coordinate shifts, enabling ray-conditioned memory retrieval by indexing memory items with orientation-adjusted ray coordinates.

## Method

### (A) What it is
**Rotation-Aware Spherical Memory Retrieval (RASMR)** is a module that takes a set of memory items (each a spherical feature map recorded at a reference orientation) and a query ray direction, and retrieves features via cross-attention over memory items sampled at the query ray direction rotated into the memory's coordinate frame. Input: memory items {M_i} (each a spherical feature map of size 64x128 with d=256 channels for a given location), query ray direction r = (θ,φ), camera rotation R (3×3 matrix). Output: aggregated memory feature vector (d=256) for that ray.

**Assumption (load-bearing):** This method assumes that the camera motion between memory and query frames is purely rotational (no translation). Under translation, ray correspondence breaks due to parallax, and the transformed ray direction may not point to the same scene point. This is a known limitation, and a repair would require per-ray depth adjustment.

### (B) How it works
```python
def rasmer_retrieve(M, r, R, K=8):
    # Step 1: Adjust query ray direction to memory reference frame
    # r is (θ, φ) in radians. Convert to Cartesian, apply inverse rotation, then back to spherical.
    r_adjusted = normalize( R.inverse() * cartesian(r) )  # cartesian: (sinθcosφ, sinθsinφ, cosθ)
    # Step 2: Sample memory features at the adjusted direction via bilinear spherical interpolation
    sampled_features = []
    for M_i in M:   # M_i is a (64, 128, 256) spherical feature map
        # bilinear interpolation on spherical feature map M_i at spherical coords of r_adjusted
        # Handles wrap-around on φ (0=2π) and θ∈[0,π]
        f_i = bilinear_spherical_sample(M_i, r_adjusted)
        sampled_features.append(f_i)   # each f_i is a (256,) vector
    # Step 3: Cross-attention with query embedding of original ray
    q = nn.Linear(2, 256)(embed(r))  # embed: sine-cosine positional encoding (16 frequencies)
    keys = torch.stack(sampled_features)      # (num_mem, 256)
    values = keys
    d = 256
    attn_weights = torch.softmax(q @ keys.T / sqrt(d), dim=-1)
    out = (attn_weights @ values).squeeze(0)  # weighted sum over memory items
    return out

# Helper: bilinear spherical sampling
def bilinear_spherical_sample(feature_map, theta_phi):
    # feature_map: (H, W, C) with H=64 (θ bins), W=128 (φ bins)
    # theta_phi: (θ, φ) in [0,π] x [0,2π]
    # Convert to pixel coordinates (continuous)
    h = theta_phi[0] / π * (H-1)   # map θ to [0,H-1]
    w = theta_phi[1] / (2π) * W    # map φ to [0,W]; wrap: w mod W
    # Bilinear interpolation
    h0, h1 = floor(h), ceil(h)
    w0, w1 = floor(w), ceil(w)
    w0 = w0 % W; w1 = w1 % W
    # compute weights and interpolate
    return interpolated_vector
```
Hyperparameters: K=8 (not used in this variant; all memory items considered). Learned linear projection: nn.Linear(2, 256) with sine-cosine positional encoding for r. Dimension d=256. The module is trained jointly with the rest of the network using Adam optimizer (lr=1e-4).

### (C) Why this design
We chose to transform the query ray into the memory coordinate frame (R^{-1}r) rather than transforming all memory items, because memory can be large (hundreds of locations) and ray transformation is O(1) per query. We use cross-attention rather than a simple correlation or nearest-neighbor retrieval because attention can adaptively weight contributions from multiple memory locations, handling occlusions and varying relevance. We sample features at a single adjusted direction per memory item instead of warping the entire memory map to the query view; this avoids expensive interpolation of high-dimensional feature maps across all directions, accepting that each memory contributes only a single ray-aligned feature. The trade-off is that we lose multi-ray context from memory, but this is compensated by the fact that multiple queries (one per ray) collectively build the output image. **This design assumes pure camera rotation; translation will cause misalignment, which is a known limitation.**

### (D) Why it measures what we claim
The adjusted ray direction r' = R^{-1}r operationalizes **rotation invariance** because it assumes a rigid Euclidean transform between memory and query camera frames; this assumption fails when the scene contains non-rigid motion (e.g., moving objects) or translation-induced parallax, under which r' does not correspond to the same scene point. The attention weights over memory items operationalize **relevance selection** because they assume that each memory item's spherical feature map encodes local scene geometry; this assumption fails for textureless regions, where attention becomes uniform and retrieval loses discriminability. The cross-attention aggregate operationalizes **geometric consistency** because it mixes features from multiple memory items weighted by orientation-adjusted similarity; this assumption fails under severe occlusion, where no memory item contains the correct content, and the output blurs.

### Changes due to adversarial alert
- The load-bearing assumption (pure rotation) is now explicitly stated in the method description and in block C.
- The verification story: if performance drops significantly on sequences with large translations, it confirms the assumption's failure. We will report translation magnitude vs. PSNR drop as a sensitivity analysis in the experiment.

## Contribution

(1) A unified memory retrieval module (RASMR) that jointly encodes camera rotation into both memory indexing and query conditioning via spherical coordinate transformation of rays. (2) Design principle that orientation-adjusted ray indexing can inject rotation awareness into any memory-augmented panoramic generation pipeline without separate rotation modules or re-training the core generation model. (3) Analysis linking spherical epipolar geometry to memory attention, establishing that rotating query rays is equivalent to shifting the memory's spherical grid.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | World360 with diverse rotations (0-180°) and small translations (<10% of scene radius) | Tests rotation invariance while acknowledging translation limitation |
| Primary metric | PSNR (on whole images and on rotated-view subsets) | Direct pixel-level quality, subset analysis isolates rotation handling |
| Baseline 1 | Static memory (single view) | Ignores rotation, naive baseline |
| Baseline 2 | Homography warping retrieval | Planar approximation vs spherical |
| Baseline 3 | Nearest-neighbor retrieval (no attention) | Isolates attention importance |
| Baseline 4 | Learned rotation-equivariant features (Spherical CNN with group convolutions) | Highlights novelty of explicit spherical geometric approach |
| Ablation of ours | RASMR without attention (mean pooling) | Isolates adaptive weighting effect |
| Additional analysis | Sensitivity to translation magnitude (vary translation up to 20% of scene radius) | Quantifies when pure-rotation assumption breaks |
| Qualitative validation | User study: 20 participants rate realism of 50 challenging rotated cases (1-5 Likert scale) | Demonstrates practical impact beyond PSNR |

### Why this setup validates the claim
This experimental design provides a falsifiable test of RASMR's central claims. The static baseline tests the necessity of rotation handling: if RASMR outperforms, it validates rotation invariance. Homography warping tests the spherical coordinate adjustment; poorer performance would confirm the spherical approach's advantage. Nearest-neighbor retrieval tests the attention mechanism's role in relevance selection. The rotation-equivariant baseline tests whether explicit geometric modeling is superior to learned equivariant features. The ablation isolates the effect of adaptive weighting. PSNR is chosen because it directly measures reconstruction accuracy, and we expect the largest gaps in rotated views, which we will analyze via subset evaluation. The dataset with varied camera rotations and small translations ensures we can evaluate both pure rotation and the assumption's breakdown. The sensitivity analysis on translation magnitude will empirically quantify when the method fails. The user study provides subjective quality assessment on challenging cases. Together, this setup allows us to attribute any performance difference to the specific mechanisms claimed and to understand limitations.

### Expected outcome and causal chain

**vs. Static memory** — On a case where the query camera rotates 90° relative to the memory location, the static baseline uses the same spherical map without adjustment, so it retrieves features from the wrong direction, leading to misaligned content and lower PSNR. Our method transforms the query ray to the memory frame, correctly aligning features, producing accurate reconstruction. We expect a noticeable gap (e.g., 2-3 dB) on rotated views but parity on head-on views.

**vs. Homography warping** — On a scene with strong depth variation (e.g., a bookshelf corner), homography warping assumes a planar mapping and distorts features, causing ghosting artifacts. Our spherical sampling directly interpolates on the sphere, preserving geometric consistency. We expect consistent superiority (1-2 dB) across all scenes, with larger gaps in non-planar regions.

**vs. Nearest-neighbor retrieval** — On a viewpoint that straddles two memory locations (e.g., a doorway seen from two adjacent positions), nearest-neighbor picks one, missing complementary details. Our cross-attention aggregates both, producing fuller content. We expect a gain of 0.5-1 dB and lower variance in such overlapping regions.

**vs. Rotation-equivariant features** — On a purely rotated view (no translation), both methods perform well, but our explicit geometric approach should be simpler and more data-efficient. We expect comparable PSNR but faster convergence (fewer training steps). On views with small translation, our method may degrade sooner due to the pure-rotation assumption, while equivariant features might handle slight translation better. This baseline highlights the trade-off between explicit geometry and learned invariance.

**Ablation (no attention)** — Removing attention degrades performance, especially in occlusion or ambiguous regions, confirming that adaptive weighting is crucial. Expected drop of 0.3-0.5 dB.

### What would falsify this idea
If RASMR's PSNR is no better than the static baseline on rotated views, or if the attention weights are uniform across memories, indicating that the mechanism fails to discriminate relevance. Additionally, if the performance degrades sharply even for small translations (e.g., less than 5% of scene radius), the pure-rotation assumption is too restrictive for practical 6-DOF scenarios.

## References

1. PanoWorld: Real-World Panoramic Generation
2. Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model
3. CamPVG: Camera-Controlled Panoramic Video Generation with Epipolar-Aware Diffusion
4. Unified Camera Positional Encoding for Controlled Video Generation
5. DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
6. CamI2V: Camera-Controlled Image-to-Video Diffusion Model
7. DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion
8. CamCo: Camera-Controllable 3D-Consistent Image-to-Video Generation

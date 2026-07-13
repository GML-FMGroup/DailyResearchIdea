# Cyclic Depth Consistency for Robust Geometry-Aware Panoramic Image Generation

## Motivation

Existing geometry-aware pretraining for panoramic generation (e.g., Canvas360) relies on monocular depth estimation that is unreliable on specular surfaces and complex geometry, causing error propagation. This limitation is shared across multiple approaches, including GeoVideo and DiffPano, which assume depth predictions are trustworthy without failure detection. The 360° wrap-around topology of panoramas offers a unique self-supervision signal: depth maps must be consistent after a full rotation, enabling automatic detection of unreliable regions.

## Key Insight

The panoramic format's cyclic topology imposes a consistency constraint that depth predictions from rotated views must agree after warping, providing a geometric self-supervision signal to identify and downweight unreliable depth regions without additional data or supervision.

## Method

### (A) What it is
**Cyclic Depth Consistency Module (CDCM)** is a plug-in module for geometry-aware pretraining of panoramic image generators. Its inputs are an equirectangular panoramic image and a set of depth maps predicted by a monocular depth estimator at multiple azimuthal rotations (θ = 0°, 90°, 180°, 270°). Its output is a per-pixel confidence weight that downweights depth predictions in regions with high cyclic inconsistency.

**Load-bearing assumption:** Depth errors across rotations are independent; if depth estimates agree after warping, they are likely correct. This assumption fails when systematic depth errors occur (e.g., on mirrors or specular surfaces) where estimates from different rotations are consistent but incorrect.

### (B) How it works
```pseudocode
Input:  panorama I (H×W×3), depth maps D_θ (H×W) for θ ∈ {0°,90°,180°,270°}
Parameter: threshold τ = 0.1 (in normalized depth), consistency metric L1, σ = 0.05 (controls sharpness)

1. For each pair (θ_i, θ_j) with θ_j = (θ_i + δ) mod 360°, δ ∈ {90°,180°,270°}:
   a. Warp D_θ_i to viewpoint of θ_j using spherical warping function W (bilinear interpolation in equirectangular coordinates):
      D_i→j = warp(D_θ_i, θ_i, θ_j)
   b. Compute reprojection error E_{i,j} = |D_θ_j - D_i→j| (pixel-wise)

2. Aggregate errors over all pairs: E = mean_{i,j} E_{i,j}

3. Compute confidence weight C = exp(-E / σ) where σ = 0.05

4. During geometry-aware pretraining, modify depth loss: L_depth = C * |D_pred - D_gt|  (element-wise multiplication)
   where D_pred is the predicted depth from the generator and D_gt is the monocular depth estimate.

Note: To mitigate distortion near poles, we only consider regions with |latitude| < 60° (i.e., central 2/3 of equirectangular height).
```

### (C) Why this design
We chose a cyclic consistency constraint over a learned confidence estimator because the geometric prior is hard to learn from limited data and is guaranteed by the 360° topology; this avoids the need for training data of failure cases, accepting the computational cost of multiple depth predictions. We warp depth maps in equirectangular coordinates rather than on a sphere to maintain compatibility with standard panoramic representations, despite introducing distortion near poles; we mitigate this by only considering regions where the equirectangular representation is locally well-behaved (latitude < 60°). We use a fixed threshold τ and exponential weighting instead of a learned gating network to keep the module transparent and avoid overfitting to specific failure patterns; this trades off adaptivity for simplicity and reproducibility. We chose to enforce consistency between all rotational pairs rather than consecutive frames to capture global consistency, which increases robustness to local depth outliers but raises memory usage.

### (D) Why it measures what we claim
Cyclic reprojection error E measures **depth geometric consistency** under the assumption that the scene is static and the depth estimator is accurate on reliable regions; this assumption fails when the scene contains moving objects, in which case E reflects scene motion rather than depth uncertainty. The confidence weight C = exp(-E/σ) measures **estimated depth reliability** under the assumption that depth errors across rotations are independent; this assumption fails for transparent or reflective surfaces where depth estimates from different rotations may be consistent but incorrect, leading to inflated confidence. The modified loss L_depth = C * |D_pred - D_gt| measures **geometry-aware training signal** under the assumption that downweighting unreliable regions prevents error propagation; this assumption fails if depth failures are systematic across rotations (e.g., all rotations produce similar wrong depth for a mirror), in which case both D_pred and D_gt are equally wrong and the loss is masked incorrectly.

**Verification strategy:** To verify that CDCM confidence correlates with true depth uncertainty, we construct synthetic panoramas from ground-truth geometry and inject Gaussian noise (σ=0.2) into random 20% of depth pixels. We then compute the Spearman rank correlation between CDCM confidence and the binary noise mask. A high positive correlation (e.g., >0.5) would support the assumption; low correlation would indicate that the module does not capture uncertainty as intended.

## Contribution

(1) A Cyclic Depth Consistency Module that automatically detects unreliable depth predictions in panoramic images by leveraging the 360° wrap-around topology, without requiring external supervision or training data. (2) A robust geometry-aware pretraining framework that downweights failure regions during training, preventing error propagation from monocular depth estimation errors on reflective surfaces and complex geometry. (3) Empirical demonstration that the method improves generation quality on challenging scenes, as measured by FID and LPIPS, compared to standard geometry-aware pretraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Canvas360Dataset | Standard panoramic benchmark for in-context tasks |
| Primary metric | FID (lower is better) | Measures generated image quality holistically |
| Baseline 1 | Paint-by-Example | Generic in-context generation without 360° prior |
| Baseline 2 | PanoGAN | Dedicated panoramic GAN baseline |
| Baseline 3 | Stable Diffusion | Strong text-to-image baseline without geometry |
| Ablation-of-ours | Ours w/o CDCM | Standard depth loss, same data and backbone |
| Additional ablation | Ours w/ edge-based weighting | Replace CDCM with Canny edge confidence (heuristic) |

### Why this setup validates the claim
Canvas360Dataset provides diverse indoor/outdoor scenes with ground truth depth for evaluating geometry-aware pretraining. Paint-by-Example and Stable Diffusion lack panoramic geometry priors, isolating the benefit of our cyclic consistency module. PanoGAN uses sphere-specific convolutions but no depth consistency, testing if our plug-in outperforms a dedicated architecture. The ablation (w/o CDCM) directly measures the contribution of the cyclic weighting. The additional ablation (edge-based) tests whether a simpler local cue yields similar gains, demonstrating that cyclic consistency—not any arbitrary cue—drives improvement. FID is chosen because it reflects overall visual quality; if geometry-aware pretraining reduces artifacts from depth inaccuracies, FID should improve on scenes with challenging geometry (high distortion, reflective surfaces). This setup thus forms a falsifiable test: the method's superiority must be concentrated on subsets where monocular depth is unreliable (e.g., near poles or mirrors), not uniform across all data.

### Expected outcome and causal chain

**vs. Paint-by-Example** — On a panoramic street scene with buildings at varying distances, Paint-by-Example ignores 360° continuity and produces seams or misaligned structures because it treats the panorama as a flat image without geometric constraints. Our method first predicts depth from four rotations, then computes a confidence map that downweights inconsistent depth in highly curved regions (e.g., near poles). The generator is pretrained with this confidence-weighted depth loss, leading to fewer seaming artifacts and more consistent structures. We expect a noticeable FID gap (e.g., ~2–3 points) on scenes with large depth variability, but parity on simple scenes with uniform depth.

**vs. PanoGAN** — On a room interior with a mirror, PanoGAN's sphere-aware convolutions capture layout but cannot handle the reflective surface: it either reproduces the mirror as a perfect reflection or distorts it, because its loss treats all pixels equally. Our method's cyclic consistency detects the mirror as geometrically inconsistent (depth estimates from different rotations contradict), so CDCM assigns low confidence to those regions. The depth loss there is downweighted, preventing the generator from overfitting to erroneous depth; however, since the mirror is inherently non-Lambertian, the generator may still produce a plausible but incorrect reflection, leading to limited gain. We expect a small FID improvement on mirror scenes (e.g., <1 point), but a larger gain on non-reflective geometrically complex scenes.

**vs. Stable Diffusion** — On a panoramic outdoor scene with a large depth range (e.g., mountain panorama), Stable Diffusion often creates flat, texture-less skies or distorted objects because it lacks any geometry awareness. Our method enforces depth consistency across rotations during pretraining, so the generator learns to produce varied depths that match the 360° structure. Consequently, generated panoramas have more realistic perspective and fewer flat regions. We expect a larger FID improvement (e.g., ~3–5 points) on scenes with strong depth variation, but minimal gain on scenes with shallow depth.

**Subset analysis:** We will also report FID on three stratified subsets of the test set: (1) scenes containing mirrors or specular surfaces (annotated via material tags), (2) low-texture scenes (e.g., uniform walls, skies), (3) high-distortion scenes (poles, i.e., latitude >60°). We hypothesize that our method will achieve the largest gain on subset (3), moderate gain on (2), and minimal gain on (1) due to the systematic failure mode.

### What would falsify this idea
If our method's FID improvement is uniform across all scene types (e.g., similar gain on simple and complex geometry), then the cyclic consistency module is not specifically addressing geometric inconsistency as claimed. A diagnosis: if the ablation (w/o CDCM) performs nearly as well as the full method on high-distortion subsets, then the confidence weighting is ineffective. Furthermore, if the synthetic verification experiment yields low correlation (Spearman ρ < 0.3), the core assumption that consistency implies reliability is not supported.

## References

1. Enhancing In-context Panoramic Generation via Geometric-aware Pretraining
2. GeoVideo: Introducing Geometric Regularization into Video Generation Model
3. UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation
4. DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training
5. Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation
6. DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion
7. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
8. MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos

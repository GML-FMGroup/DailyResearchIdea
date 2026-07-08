# Physics-Guided Bokeh Rendering with Differentiable Lens Aberration Correction

## Motivation

AnyBokeh achieves high-quality bokeh editing by modeling defocus as a linear function of disparity, which holds for ideal thin lenses. However, real lenses exhibit spherical aberration, coma, and astigmatism that break this linearity, causing visible artifacts in the rendered bokeh. The root cause is that the linear CoC-disparity model cannot capture the nonlinear, spatially varying blur introduced by non-ideal optics, limiting real-world fidelity.

## Key Insight

By parameterizing lens aberrations as a small set of Zernike coefficients and optimizing them via differentiable ray tracing on a single image pair (all-in-focus and defocused), we can decompose real defocus into an ideal linear term plus a learnable, physically interpretable correction, preserving the interpretability of the CoC-disparity backbone while achieving accurate bokeh for arbitrary lenses.

## Method

We propose **Aberration-Corrected Bokeh (ACB)**, an extension of the AnyBokeh pipeline that adds a differentiable aberration correction module. 

**(A) What it is:** ACB takes as input a source image and its disparity map, estimates a signed CoC map using the linear CoC-disparity relation (AnyBokeh's optical fingerprint), then applies a learned aberration correction to the CoC radii. The corrected CoC map is used to render bokeh via the same generative editor as AnyBokeh. The aberration correction module is a function parameterized by Zernike coefficients (spherical, coma, astigmatism) that is learned per lens from a single image pair (sharp and defocused).

**(B) How it works (pseudocode):**
```python
# Input: source_image I_s, disparity map D, target CoC map CoC_t (for desired bokeh)
# Step 1: Estimate linear CoC map from D
CoC_ideal = alpha * (D - D_focus)  # alpha is optical fingerprint slope
# Step 2: Load per-lens aberration coefficients (learned)
z = [z0, z1, z2]  # e.g., spherical(4,0), coma(3,±1), astigmatism(2,±2)
# Step 3: Differentiable ray-tracing to compute aberration blur kernel
# For each pixel, compute PSF using wave optics (Zernike-based wavefront aberration)
PSF = ray_trace(D, z, aperture_shape='hexagon')  # outputs per-pixel PSF
# Step 4: Compute correction offset as CoC shift from PSF width
CoC_correction = fwhm(PSF) - CoC_ideal  # fwhm: full-width at half-max
# Step 5: Apply correction to CoC map
CoC_corrected = CoC_ideal + CoC_correction
# Step 6: Render bokeh using AnyBokeh's generative editor conditioned on CoC_corrected
bokeh_image = generator(I_s, CoC_ideal, CoC_corrected)  # as in AnyBokeh
return bokeh_image
```
*Hyperparameters:* Ray-tracing uses 100 rays per pixel, 3 Zernike modes (spherical, coma, astigmatism) with coefficients optimized by Adam (lr=0.01, 500 iterations) on a single real (sharp, defocused) image pair.

**(C) Why this design:** We chose to model aberration correction as an additive offset to the linear CoC radius rather than learning a full black-box correction network because the additive offset preserves the interpretable structure of the CoC-disparity relation, allowing the model to generalize to new lenses by only optimizing 3 scalar coefficients. This design trades flexibility (black-box could model arbitrary nonlinearities) for data efficiency and physical interpretability; the cost is that we assume aberrations dominantly affect blur width rather than shape, which may miss subtle asymmetries (e.g., coma orientation), but these can be captured by increasing Zernike modes. We chose Zernike polynomials over Seidel coefficients because they are orthonormal and directly correspond to wavefront aberrations, making gradient-based optimization stable. We used differentiable ray tracing instead of a precomputed lookup table because it allows end-to-end optimization of coefficients from image pairs without manual measurement; the cost is higher computation during training, but it is done once per lens. Finally, we retain AnyBokeh's generative editor instead of designing a new renderer because a pretrained editor on CoC maps already handles spatially adaptive blur synthesis, and our correction only modifies the input CoC map, making the integration seamless.

**(D) Why it measures what we claim:** The quantity `CoC_correction = fwhm(PSF) - CoC_ideal` measures the deviation of real defocus blur from the ideal thin-lens model because the full-width half-max of the physically simulated PSF (including aberrations) directly captures the actual blur extent; the assumption is that the CoC radius (blur disk diameter) is a sufficient statistic for bokeh quality, and that the linear CoC-disparity model's error is primarily in the width. This assumption fails when aberrations produce highly asymmetric or non-Gaussian blur (e.g., strong coma), in which case the FWHM metric reflects a partial measure and may need to be supplemented by higher-order moments. The learned Zernike coefficients `z` measure the per-lens aberration strengths because the wavefront aberration is a linear combination of Zernike modes; the assumption is that the optics are well-approximated by a low-order expansion, which holds for common consumer lenses (3-5 terms capture >90% of variance). This assumption fails for exotic lenses with high-order aberrations, in which case the coefficients reflect the best low-order approximation rather than true aberration. The optimization target (reconstruction loss between rendered bokeh and real defocused image) measures consistency of the full forward model, ensuring that the corrected CoC map produces a bokeh that matches the real capture; the assumption is that the generative editor can faithfully render any CoC map, which is validated by AnyBokeh's success on ideal CoC maps. This assumption fails if the editor overfits to ideal CoC distributions, in which case the loss reflects primarily the editor's limitation rather than aberration correction accuracy.

## Contribution

(1) A physically parameterized aberration correction module that extends the linear CoC-disparity model to real lenses by learning a small set of Zernike coefficients via differentiable ray tracing, preserving interpretability. (2) A demonstration that a single image pair (sharp and defocused) suffices to optimize per-lens aberration coefficients, enabling data-efficient calibration for any lens. (3) Integration of the correction module into the AnyBokeh framework, yielding improved bokeh fidelity on lenses with known aberrations without retraining the generative editor.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RealDOF dataset | Real paired sharp-defocused images |
| Primary metric | LPIPS | Perceptual similarity sensitive to blur |
| Baseline 1 | AnyBokeh (original) | Direct predecessor without correction |
| Baseline 2 | BokehNet | Typical all-in-focus pipeline |
| Baseline 3 | Pix2Pix | Direct image translation baseline |
| Ablation | ACB w/o correction | Isolates effect of aberration module |

### Why this setup validates the claim

This experimental design tests the central claim—that a learned additive aberration correction to the CoC map improves bokeh rendering fidelity—by comparing against three baselines that isolate different failure modes. The RealDOF dataset provides real lens-specific sharp-defocused pairs, enabling per-lens coefficient learning. LPIPS is chosen because it captures perceptual blur shape discrepancies that PSNR/SSIM may miss. AnyBokeh tests the base pipeline without correction, identifying whether the correction itself adds value. BokehNet represents classic defocus-then-bokeh, which lacks any aberration awareness. Pix2Pix tests a black-box translation that could implicitly correct aberrations but requires large data. The ablation (ACB without correction) pinpoints the correction module's contribution. A falsifiable pattern emerges: if our method outperforms all baselines specifically on images with strong aberrations (e.g., coma or astigmatism), and the ablation underperforms on those same cases, the claim is supported. Conversely, uniform improvement across all images would indicate the correction is unnecessary or overfits to trivial patterns.

### Expected outcome and causal chain

**vs. AnyBokeh (original)** — On a case where the lens has significant spherical aberration causing a non-uniform blur disk, AnyBokeh uses the ideal linear CoC map, producing bokeh with unnaturally uniform brightness across the disk. Our method estimates the actual PSF width via ray tracing and adjusts the CoC map to widen the blur where aberration spreads energy, resulting in a more natural, softer bokeh. We expect a noticeable LPIPS gap (e.g., 0.05 lower) on images with strong spherical aberration, but near parity on near-ideal lenses.

**vs. BokehNet** — On a case where the defocused region contains high-frequency texture (e.g., foliage), BokehNet first deblurs the image, losing fine details in the process, then applies a synthetic bokeh filter, producing over-smoothed results that lack realistic blur transitions. Our method directly works from the sharp source and applies a spatially varying blur that preserves texture in in-focus areas while modulating blur according to the corrected CoC map. We expect our method to achieve substantially lower LPIPS (e.g., 0.1 lower) on scenes with complex textures, but similar performance on simple backgrounds.

**vs. Pix2Pix** — On a case where the lens has asymmetric coma causing comet-like blur tails, Pix2Pix trained on a generic dataset cannot generalize to this specific aberration; it may hallucinate incorrect blur shapes or produce artifacts because it lacks explicit CoC conditioning. Our method, by learning per-lens Zernike coefficients from a single pair, explicitly captures the asymmetry and applies a correction to the CoC map, generating bokeh with faithful comet-shaped blurs. We expect our method to show a clear LPIPS advantage (e.g., 0.08 lower) on lenses with strong coma, while Pix2Pix may perform comparably on symmetric aberrations.

### What would falsify this idea

If the LPIPS improvement of ACB over its own ablation (no correction) is uniform across all image subsets regardless of aberration strength, or if ACB underperforms AnyBokeh on near-ideal lenses, then the central claim—that the correction module specifically addresses aberration-induced errors—is falsified.

## References

1. AnyBokeh: Physics-Guided Any-to-Any Bokeh Editing with Optical Fingerprint Transfer
2. Generative Refocusing: Flexible Defocus Control from a Single Image
3. Bokeh Diffusion: Defocus Blur Control in Text-to-Image Diffusion Models
4. DiffCamera: Arbitrary Refocusing on Images
5. Variable Aperture Bokeh Rendering via Customized Focal Plane Guidance
6. Edify Image: High-Quality Image Generation with Pixel Space Laplacian Diffusion Models
7. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
8. Efficient Single-Image Depth Estimation on Mobile Devices, Mobile AI & AIM 2022 Challenge: Report

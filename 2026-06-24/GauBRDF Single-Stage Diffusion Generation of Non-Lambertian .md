# GauBRDF: Single-Stage Diffusion Generation of Non-Lambertian 3D Gaussians with Global Illumination and Physically-Based BRDF

## Motivation

Current single-stage 3D Gaussian generation methods, exemplified by FLUX3D, assume Lambertian surfaces and rely on multi-view pose consistency, failing to produce view-dependent effects like specular reflections and refractions that are common in real-world objects. This limitation arises because the diffusion pipeline lacks a physically-motivated representation for global illumination and material properties, forcing the model to encode all appearance variation into geometric features without separating intrinsic material from lighting. Consequently, generating glossy or transparent surfaces requires costly post-hoc inverse rendering or explicit camera pose supervision.

## Key Insight

By factorizing appearance into a compact global illumination field and per-Gaussian physically-based BRDF parameters, the diffusion model can directly learn the causal structure of view-dependent effects from unposed multi-view data, because the illumination field explains global lighting variations while BRDF parameters capture local material response, decoupling the two in a way that generalizes to novel views without pose extrapolation.

## Method

**(A) What it is:** GauBRDF is a latent diffusion model that generates a set of 3D Gaussians augmented with a compact global illumination field (represented as a small MLP) and per-Gaussian BRDF parameters (diffuse albedo, specular albedo, roughness). The input is a text prompt or single image; the output is a 3D Gaussian representation with physically interpretable material and lighting, enabling rendering of realistic view-dependent effects via a differentiable renderer that computes outgoing radiance using the BRDF and illumination during training.

**(B) How it works:**
```python
# Pseudocode for GauBRDF diffusion training with unposed multi-view images
# Notation: 
#   G = initial Gaussian latents (positions, scales, rotations, opacity)
#   ILLUM = global illumination field MLP (input: direction d, output: radiance L_i)
#   BRDF_params = per-Gaussian: diffuse albedo a_d, specular albedo a_s, roughness rho
#   R = differentiable renderer that uses ILLUM and BRDF_params to render view v

for each batch of unposed multi-view images {I_v} with unknown poses:
    # 1. Sample latent Gaussian parameters G0 from noise
    # 2. Sample ILLUM weights and BRDF_params from noise (as part of latent)
    # 3. Forward diffusion: add noise to G, ILLUM, BRDF_params
    # 4. Denoising network: a Transformer takes noisy latents and condition text/image
    #    It outputs delta for G, ILLUM, and BRDF_params
    # 5. Decode: use predicted G, ILLUM, BRDF_params to render images at random views
    #    using a differentiable renderer that computes: 
    #    L_o(v) = a_d * (1/pi) * integral(L_i * cos theta_i d omega_i) + 
    #             a_s * D(roughness) * F * G / (4 * cos theta_o * cos theta_i) (cook-torrance)
    #    The renderer also handles refraction via a transmissive lobe (specular BTDF).
    # 6. Loss: L2 between rendered and ground truth images (no pose needed, only pixel loss)
    # 7. Backprop through denoiser and renderer
    #    Use reparameterization to ensure BRDF parameters are physically plausible
    #    (e.g., roughness > 0, albedos in [0,1] via sigmoid)
```
Hyperparameters: illumination MLP is a 2-layer MLP with 64 hidden units; BRDF parameters are stored as per-Gaussian 6-dimensional vectors (a_d:3, a_s:3, rho:1, but we use 6D with reparameterization). Diffuser uses 1000 timesteps with cosine noise schedule.

**(C) Why this design:** We chose a compact global illumination MLP over a precomputed environment map because it can be optimized jointly with the Gaussians and adapts to the scene, though the trade-off is that it cannot represent local lighting variations due to occlusion, which we accept as a limitation for isolated objects. We chose per-Gaussian Cook-Torrance BRDF parameters over a neural appearance field (like NeRF) to enable explicit material control and facilitate downstream editing, but this adds storage and requires careful parameterization to avoid artifacts during training (e.g., roughness must be bounded). We chose to render at random views without pose supervision rather than using known camera matrices, which avoids the need for pose estimation but introduces ambiguity between object shape and lighting; to mitigate this, we leverage the diffusion prior and multi-view consistency from the denoising process. We chose a latent diffusion architecture similar to FLUX3D for the base Gaussian generation, but extended with additional tokens for illumination and BRDF parameters, which increases model complexity but allows joint generation of geometry and appearance in a single pass. The use of a differentiable renderer during training, as opposed to a post-hoc rendering loss, ensures that the predicted BRDF and illumination directly influence the final pixel values, enabling end-to-end learning of view-dependent effects.

**(D) Why it measures what we claim:** The quantity `rendered radiance L_o(v)` computed from the global illumination field `ILLUM` and per-Gaussian BRDF parameters measures **view-dependent surface appearance** because it models the physical interaction of light with a surface having specular and diffuse components; the assumption that a single global illumination field plus local BRDF can explain all view variations is equivalent to assuming distant lighting and no inter-reflections, which fails when objects have significant self-occlusion or subsurface scattering, in which case `L_o(v)` reflects only the dominant directional lighting and not multiple scattering. The predicted BRDF parameters (a_d, a_s, rho) measure **intrinsic material properties** because they are decoupled from illumination by the rendering equation: a_d represents the fraction of incident light reflected diffusely, assuming Lambertian law, which fails for metals where specular albedo dominates, in which case a_d becomes near-zero and specular parameters capture the reflection. The rendered images across multiple views measure **multi-view consistency without pose** because the same global illumination field and BRDF are used for all views, enforcing a shared world-space representation; this assumption holds if the object is static and lighting is fixed, but fails if the object deforms or lighting changes between views, in which case the loss averages out inconsistencies, producing a blurred appearance.

## Contribution

(1) A novel diffusion architecture that jointly generates 3D Gaussians, a compact global illumination field, and per-Gaussian physically-based BRDF parameters from text or single image, enabling single-stage generation of non-Lambertian surfaces. (2) The insight that factorizing appearance into global illumination and local material allows the model to learn view-dependent effects from unposed multi-view images, removing the need for explicit camera pose supervision. (3) A differentiable rendering layer that integrates Cook-Torrance BRDF and a global illumination MLP directly into the denoising loop, providing physically-based gradients for material and lighting learning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Google Scanned Objects | Real scans with diverse materials and lighting. |
| Primary metric | Novel view PSNR | Measures view synthesis quality directly. |
| Baseline 1 | FLUX3D | Tests benefit of BRDF+illumination over spherical harmonics. |
| Baseline 2 | DreamGaussian | Generative baseline lacking physically based rendering. |
| Ablation-of-ours | GauBRDF w/o global illumination MLP | Tests necessity of adaptive illumination field. |

### Why this setup validates the claim

This setup directly tests whether the explicit global illumination MLP and per-Gaussian BRDF parameters improve novel view synthesis under unposed multi-view supervision. Google Scanned Objects provides real-world objects with complex materials and lighting variations, challenging the model to disentangle geometry, material, and illumination. FLUX3D shares a similar diffusion backbone but uses a simpler appearance model (spherical harmonics), isolating the impact of our physically interpretable representation. DreamGaussian is a generative method without BRDF, testing whether the renderer and training scheme yield better realism. The ablation removes the adaptive illumination field, probing its role in capturing view-dependent effects. Novel view PSNR is the natural metric: it quantifies pixel-level accuracy across unseen angles, and improvements over baselines on specular or refractive objects would confirm that our method captures physical appearance properties. This combination creates a falsifiable test: if our method outperforms baselines only on simple diffuse scenes, the disentanglement claim fails.

### Expected outcome and causal chain

**vs. FLUX3D** — On a metallic object (e.g., chrome teapot) under directional light, FLUX3D produces blurry reflections and inconsistent highlights across views because its spherical harmonics cannot separate illumination from albedo. Our method uses Cook-Torrance BRDF and a learned illumination field, yielding sharp, physically consistent specular highlights. We expect a noticeable PSNR gap of 3–5 dB higher on such specular objects, but parity on purely diffuse scenes.

**vs. DreamGaussian** — On a translucent object (e.g., glass sphere), DreamGaussian trained on simple rendered views fails to capture refraction, producing distorted backgrounds and missing caustics. Our method incorporates a transmissive lobe (specular BTDF) and learns from multi-view images, implicitly modeling refraction paths. Expect DreamGaussian to underperform by 2–3 dB on refractive objects, while our method maintains high fidelity.

**vs. Ablation (w/o global illumination MLP)** — On a matte object under non-uniform lighting (e.g., museum statue with spotlights), the fixed environment map in the ablation cannot adapt to scene-specific shadows, causing incorrect shading. Our full method adapts the illumination MLP per scene, capturing the dominant lighting direction. Expected gap of 1–2 dB on unevenly lit scenes, but parity on uniformly lit ones.

### What would falsify this idea

If GauBRDF's PSNR is consistently below both baselines or the ablation matches the full method on all scenes, the proposed representations are not capturing meaningful physical disentanglement. Specifically, if the gain over FLUX3D is uniform across object types rather than concentrated on specular and refractive objects, it suggests improvement stems from other factors (e.g., training details) rather than the explicit BRDF+illumination.

## References

1. FLUX3D: High-Fidelity 3D Gaussian Generation with Diffusion-Aligned Sparse Representation
2. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
3. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
4. RomanTex: Decoupling 3D-Aware Rotary Positional Embedded Multi-Attention Network for Texture Synthesis
5. Boosting Latent Diffusion with Flow Matching
6. OmniGen: Unified Image Generation
7. In-Context LoRA for Diffusion Transformers
8. TEXGen: a Generative Diffusion Model for Mesh Textures

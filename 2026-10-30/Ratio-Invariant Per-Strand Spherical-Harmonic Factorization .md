# Ratio-Invariant Per-Strand Spherical-Harmonic Factorization for Hair Reconstruction

## Motivation

Across both single-view braid fitting (T_0) and GaussianHair (T_1), image-space photometric and silhouette losses assume uniform or unknown lighting, so strand geometry and shading are conflated: a strand can absorb an illumination change by bending, and a change in shading can be explained by moving geometry. This meta-level entanglement recurs because neither lineage introduces a light-transport model per strand; T_1 explicitly names the uniform-lighting assumption, and T_0 inherits the same conflation through silhouette/appearance losses. Without a per-strand radiance model and a loss invariant to the unknown global light intensity, optimization cannot tell whether a pixel mismatch is due to geometry or illumination.

## Key Insight

The ratio between rendered and observed radiance cancels the unknown global light intensity, and when each strand carries its own low-order spherical-harmonic radiance profile optimized jointly with geometry, the remaining per-strand SH coefficients absorb only strand-local scattering while the ratio loss forces geometry to explain view-dependent radiance changes, breaking the geometry-shading entanglement without a separate lighting network.

## Method

**What it is** — SH-STRAND (Spherical-Harmonic Strand Radiance Factorization) takes multi-view images and camera poses plus an initial hair geometry (from GaussianHair or braid-parametric fitting) and jointly optimizes strand geometry and a per-strand low-order SH radiance profile, producing a strand-based hair model whose geometry is disentangled from illumination.

**How it works** — pseudocode:

```python
# Inputs: images {I_v}, cameras {C_v}, initial strands S = {s_i} (polylines)
# Hyperparameters: SH order L=2 (9 coefficients per strand), ratio-loss epsilon=1e-3, SH smoothness lambda_sh=0.01, geometry smoothness lambda_geo=0.1
# Gauge-fixing: after each SH update, subtract the mean SH coefficient across all strands from each c_i (so sum_i c_i = 0); this removes the per-strand scale/color degeneracy that would otherwise let SH absorb global intensity.
for iteration in range(N):
    for each view v:
        # 1. Differentiable rasterization of strand primitives (cylindrical Gaussians or line segments)
        #    Each strand s_i carries SH radiance coefficients c_i = {c_i^lm} for l<=L.
        #    Rendered radiance at pixel p: R_v(p) = sum_i alpha_i(p) * SH_eval(c_i, omega_v(p))
        #    where omega_v(p) is the view direction for strand i at pixel p.
        R_v = rasterize(S, {c_i}, C_v)
        # 2. Ratio loss: invariant to global light intensity scale
        #    L_ratio = sum_p [ (R_v(p) + eps) / (I_v(p) + eps) - 1 ]^2
        L_ratio += mean( ((R_v + eps) / (I_v + eps) - 1.0)**2 )
    # 3. Regularization
    L_sh = lambda_sh * sum_i ||c_i - mean_{j in N(i)} c_j||^2   # smooth SH across neighboring strands
    L_geo = lambda_geo * sum_i ||s_i''||^2                     # strand smoothness
    L = L_ratio + L_sh + L_geo
    # 4. Joint gradient update on strand control points and SH coefficients
    update(S, {c_i}, grad(L))
    # 5. Gauge-fixing projection: enforce zero-mean SH across strands
    c_i = c_i - mean_j(c_j) for all i
```

**Why this design** — We chose a per-strand low-order SH radiance profile over a global lighting network or a two-stage pipeline because a global network would reintroduce the same entanglement (the network can absorb geometry errors as illumination), accepting the cost that per-strand SH cannot model inter-strand shadowing and long-range light transport. We chose the ratio loss over a standard L2 photometric loss because the ratio cancels the unknown global light intensity scale, accepting the cost that the ratio loss is non-linear and can be unstable when observed radiance is near zero, mitigated by the epsilon floor. We chose joint optimization of geometry and SH coefficients over alternating optimization because joint gradients let the SH coefficients adapt to geometry changes within the same step, accepting the cost of a larger parameter space and the need for SH smoothness regularization across neighboring strands. We chose cylindrical Gaussian primitives (as in GaussianHair) over line segments because they provide a differentiable volumetric rendering model with alpha compositing, accepting the cost that the rasterizer must handle anisotropic primitives. A domain expert would not describe this as a variant of GaussianHair because GaussianHair uses a fixed uniform-lighting diffuse model and an L2 photometric loss, whereas SH-STRAND introduces per-strand SH radiance and a ratio loss that is invariant to global light intensity, a structural change to the image formation model and loss.

**Why it measures what we claim** — The ratio loss L_ratio measures geometry-shading disentanglement because the ratio (R+eps)/(I+eps) is invariant to any global scalar multiplication of both rendered and observed radiance, so the only way to reduce L_ratio is to match the spatial pattern of radiance, not its overall scale; this assumption fails when the observed image has spatially varying global illumination (e.g., a strong directional light gradient across the hair), in which case the ratio reflects local illumination mismatch rather than geometry error. The per-strand SH coefficients c_i measure strand-local scattering because SH order L=2 captures low-frequency angular variation of outgoing radiance for a slender cylinder, under the assumption that inter-strand shadowing is negligible; this assumption fails in dense hairstyles where shadowing creates high-frequency angular variation, in which case c_i absorbs shadowing as a per-strand color offset. The SH smoothness regularizer L_sh measures the physical prior that neighboring strands have similar scattering properties because hair fibers of the same type share material, under the assumption that strand neighborhoods are spatially coherent; this assumption fails at boundaries between hair and scalp or between different hair colors, in which case L_sh over-smooths and blurs color transitions. The gauge-fixing projection (zero-mean SH across strands) measures the removal of the per-strand scale/color degeneracy because it forces any global intensity change to be represented by the common mean that is subtracted, so the remaining per-strand SH can only encode relative angular variation; this assumption fails when the true per-strand albedo varies strongly across the hairstyle, in which case the projection removes legitimate color differences and the ratio loss must compensate via geometry.

## Contribution

['A per-strand spherical-harmonic radiance factorization that is jointly optimized with strand geometry, introducing a compact differentiable light-transport representation for hair that requires no separate lighting network or two-stage pipeline.', 'A ratio-based photometric loss that is invariant to the unknown global light intensity, which structurally prevents geometry from absorbing illumination changes and thus breaks the geometry-shading entanglement.', 'An end-to-end optimization framework that integrates the SH radiance model with differentiable rasterization of strand primitives (cylindrical Gaussians or line segments) and demonstrates that per-strand SH smoothness regularization is necessary to avoid color bleeding between neighboring strands.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Synthetic multi-view hair with known lighting | Ground-truth geometry isolates disentanglement |
| Primary metric | Strand geometry error under relit renders | Tests geometry-shading separation directly |
| Baseline | GaussianHair | Fixed diffuse model, L2 photometric loss |
| Baseline | Strand-aligned 3D Gaussians | Strong geometry but no SH radiance |
| Baseline | Braid-parametric fitting | Single-view, loose-hair assumptions |
| Ablation-of-ours | SH-STRAND with L2 loss (no ratio) | Removes intensity-invariance |
| Ablation-of-ours | SH-STRAND without gauge-fixing | Tests whether zero-mean SH constraint is necessary to prevent per-strand scale absorption |

### Why this setup validates the claim

The central claim is that per-strand SH radiance plus a ratio loss disentangles geometry from illumination, so geometry stays accurate when the observed light intensity or direction changes. A synthetic multi-view dataset with known ground-truth strands and controlled relighting is the right testbed because it lets us vary illumination while holding geometry fixed, isolating exactly the factor the claim is about. The primary metric — strand geometry error measured on renders under novel illumination — is the right detector because a method that merely fits pixels can look good under the training light yet drift under relighting; only true disentanglement keeps geometry error flat across lighting conditions. GaussianHair tests whether the fixed diffuse model is the bottleneck: if our gain over it appears only under relighting, the SH radiance is doing the work. Strand-aligned 3D Gaussians tests whether the ratio loss alone matters, since it has strong geometry but no intensity-invariant loss. The braid-parametric baseline tests the loose-hair assumption on structured hairstyles. The L2-loss ablation is the linchpin: it keeps per-strand SH but removes intensity invariance, so any gain attributable to the ratio loss must vanish there, pinning the effect to the claimed mechanism rather than to extra parameters. The no-gauge-fixing ablation tests whether the zero-mean SH projection is necessary to prevent per-strand SH from absorbing global intensity as a per-strand color offset; if the full method outperforms this ablation under relighting, the gauge constraint is doing the claimed work.

### Expected outcome and causal chain

**vs. GaussianHair** — On a case where the capture light is stronger than the relight light (or vice versa), GaussianHair's fixed diffuse model and L2 loss compensate for the intensity mismatch by nudging strand positions, so its geometry drifts and its relit renders show both shading and shape error. Our method instead lets per-strand SH absorb the angular radiance and the ratio loss cancel the global scale, so geometry stays anchored. We expect a clear geometry-error gap under relighting that shrinks to near-parity under the training light.

**vs. Strand-aligned 3D Gaussians** — On a case where a directional light grazes the hair, this baseline has accurate geometry but no per-strand angular radiance, so its L2 fit trades geometry against shading and produces strand drift concentrated on the lit side. Our method keeps the strong geometry prior but adds SH coefficients that explain the grazing highlight, and the ratio loss prevents scale absorption, so the drift disappears. We expect our advantage to be concentrated on the illuminated subset with parity on shadowed regions.

**vs. Braid-parametric fitting** — On a braided hairstyle with tight inter-strand structure, the braid-parametric baseline assumes loose-hair smoothness and cannot represent the braid's local geometry, producing blurred or merged strands. Our method's strand smoothness regularizer acts only along each strand, not across the braid pattern, so it preserves the braid topology while SH handles the complex shading. We expect a visible topology-accuracy gap on braided subsets but parity on loose hairstyles.

**vs. SH-STRAND without gauge-fixing** — On a case where the global light intensity changes between capture and relight, the no-gauge-fixing variant allows per-strand SH to absorb the intensity change as a per-strand color offset, so geometry drift persists and the relit renders show shape error. Our full method enforces zero-mean SH across strands, so the global intensity is represented only by the common mean that is subtracted, forcing geometry to explain the remaining spatial pattern. We expect the full method to have lower geometry error under relighting than the no-gauge-fixing variant, with parity under the training light.

### What would falsify this idea

If our geometry-error gain over GaussianHair is uniform across lighting conditions rather than concentrated under relighting, the disentanglement claim is wrong and the improvement is just extra capacity. Likewise, if the L2-loss ablation matches our full method under relighting, the ratio loss is not the causal mechanism. If the no-gauge-fixing ablation matches our full method under relighting, the gauge-fixing constraint is not necessary and the per-strand SH can absorb global intensity without a geometry penalty.

## References

1. HairGS: Hair Strand Reconstruction based on 3D Gaussian Splatting
2. Unsupervised 3D Braided Hair Reconstruction from a Single-View Image
3. Neural Haircut: Prior-Guided Strand-Based Hair Reconstruction
4. 3D Gaussian Splatting for Real-Time Radiance Field Rendering
5. Human Hair Reconstruction with Strand-Aligned 3D Gaussians
6. GaussianHair: Hair Modeling and Rendering with Light-aware Gaussians
7. Dr.Hair: Reconstructing Scalp-Connected Hair Strands without Pre-Training via Differentiable Rendering of Line Segments
8. Point-NeRF: Point-based Neural Radiance Fields

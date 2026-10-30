# Flux-Conserved Hybrid Strand Fields for Closed 3D Hair Reconstruction

## Motivation

Existing hair reconstruction representations are not closed over the hairstyle manifold: the braid-template of Unsupervised 3D Braided Hair Reconstruction cannot represent non-braid or partially braided regions, while photometric strand growth as used in HairGS and Dr.Hair cannot complete strands in occluded, frizzy, or weakly observed regions. Both failures share a structural root cause — structured and free-form strand generators are defined on disjoint parameter surfaces with no shared quantity that can move strand mass from one regime to the other, so any hybrid must be stitched by an external classifier and inherits the classifier's errors at every boundary. The field therefore needs a single conserved variable on which structured and free-form strands coexist and exchange mass, making the representation closed by construction.

## Key Insight

A single strand-density flux variable, conserved across the structured/free-form boundary and locally switched by a geometric compatibility condition (template tangent vs. photometric-growth tangent agreement), gives one continuous strand-mass surface whose two generative regimes are two limits of the same conservation law rather than two modules joined by a router.

## Method

```pseudocode
INPUT: single or few-view RGB images {I_v}, scalp mask S, camera poses {pi_v}
OUTPUT: strand set X = {x_i} with per-strand regime label r_i in {structured, free-form}

# Phase 1: seed flux field
Phi  <- initialize_strand_density(S, target_density=rho0)   # scalar per scalp texel, rho0=64 texels/cm^2
T    <- braid_template_library()                            # synthetic braid curves (from T_0)
G    <- photometric_growth_field(I_v, pi_v, S)              # HairGS-style differentiable Gaussian growth

# Phase 2: coupled optimization (differentiable Gaussian/strand rasterizer)
for iter in 1..N:   # N=3000 iterations, Adam lr=1e-3 for Phi, 1e-4 for T/G params
    # 2a. geometric compatibility switch (NOT learned) with curvature/scale agreement
    for each scalp texel u:
        # tangent agreement (original kappa)
        kappa_t(u) = <tangent(T, u), tangent(G, u)> / (|tangent(T,u)||tangent(G,u)| + eps)
        # curvature agreement: compare discrete curvature magnitudes
        kappa_c(u) = exp( - |curv(T,u) - curv(G,u)| / (|curv(T,u)| + |curv(G,u)| + eps) )
        # scale agreement: compare local strand thickness (Gaussian scale)
        kappa_s(u) = exp( - |scale(T,u) - scale(G,u)| / (scale(T,u) + scale(G,u) + eps) )
        # combined compatibility score in [0,1]
        kappa(u) = kappa_t(u) * kappa_c(u) * kappa_s(u)
        w_struct(u) = sigmoid((kappa(u) - kappa0) / tau)      # kappa0=0.7, tau=0.1
    # 2b. split conserved flux between regimes
    Phi_struct(u) = w_struct(u) * Phi(u)
    Phi_free(u)   = (1 - w_struct(u)) * Phi(u)                # conservation: Phi_struct+Phi_free=Phi
    # 2c. grow strands from flux (mass-conserving)
    X_struct <- sample_strands(T, density=Phi_struct)
    X_free   <- grow_strands(G,   density=Phi_free)
    # 2d. flux transport across boundary (exchange, not re-routing)
    # D(u) = local strand-tangent coherence = |<tangent(T,u), tangent(G,u)>|, clipped to [0,1]
    Phi <- Phi + div( D(u) * grad(Phi) ) * dt                 # dt=0.1, D from local strand-tangent coherence
    # 2e. differentiable render + photometric + silhouette loss
    L = sum_v || R(X_struct U X_free; pi_v) - I_v ||^2 + lambda_sil * silhouette + lambda_reg * strand_smooth
    # lambda_sil=0.1, lambda_reg=0.01
    backprop L through (2c)-(2e) to Phi, T-params, G-params
return X_struct U X_free
```

**Why this design.** We chose a single conserved flux variable `Phi` over a learned router or classifier (as would be the natural extension of the T_0 braid template plus HairGS growth) because a router introduces a discrete decision surface whose errors are unrecoverable at braid/free-form boundaries, whereas a conserved scalar admits a continuous exchange of strand mass and keeps the representation closed by construction — accepting the cost that `Phi` must be discretized on the scalp and its transport step adds a diffusion hyperparameter `D`. We chose the compatibility switch `kappa(u)` (tangent agreement between template and photometric growth) over a learned gating head because the switch must be determined by geometry that is already observable in the two generators, not by a signal that requires labels the single-view setting does not provide — accepting the cost that `kappa` is noisy where either generator is poorly constrained, which we absorb by the flux-transport step. We chose strand growth proportional to `Phi_struct`/`Phi_free` rather than independent strand counts because independent counts would let total strand mass drift and re-open the non-closure failure — accepting the cost that strand count is now globally coupled and cannot be tuned per-region. Unlike HairGS's post-hoc merging or Dr.Hair's pure line-segment growth, no component of our pipeline decides *which* regime a strand belongs to at inference time; the regime is read off from the flux split, so there is no router whose training is the hard problem.

**Why it measures what we claim.** The flux split `Phi_struct(u) = w_struct(u) Phi(u)` measures *closure over the hairstyle manifold* because the assumption is that local tangent agreement `kappa(u)` between the braid template and photometric growth is equivalent to local representability by the structured regime; this assumption fails when both generators are weakly constrained (severe occlusion or frizz), in which case `w_struct` reflects rasterizer noise rather than true regime membership, and the transport term redistributes that error spatially. The conservation residual `|Phi - Phi_struct - Phi_free|` measures *mass exchange integrity at the structured/free-form boundary* because the assumption is that strand mass is a locally conserved quantity on the scalp; this assumption fails at the scalp boundary where flux can leak, in which case the residual reflects boundary truncation rather than a genuine non-conservation. The transport diffusivity `D(u)`, derived from local strand-tangent coherence, measures *geometric compatibility of neighboring regimes* because the assumption is that coherent tangent fields imply compatible strand flow; this assumption fails at sharp braid-to-frizz transitions, in which case `D` reflects smoothing rather than compatibility. The photometric + silhouette loss measures *observational fidelity* because the assumption is that the differentiable rasterizer's gradient is informative for both regimes; this assumption fails for sub-pixel strands, in which case the loss reflects silhouette agreement only.

## Contribution

(1) A hybrid strand field representation in which a single conserved strand-density flux variable carries strand mass across a structured (braid-template) and free-form (photometric-growth) regime, with the regime split determined by a closed-form geometric compatibility condition rather than a learned router. (2) A differentiable, mass-conserving strand-growth and flux-transport optimization that lets structured and free-form strands exchange density within one end-to-end photometric objective, so partially braided and frizzy configurations are representable without a separate classifier. (3) A design principle — closure-by-conservation — showing that representation completeness over the hairstyle manifold can be obtained from a conserved shared variable instead of from a learned regime classifier.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Single-view braided hairstyles (synthetic + real) | Braid/free-form boundaries are the claim's stress point |
| Primary metric | Regime-boundary fidelity: strand-mass conservation residual (computed only on interior texels with a boundary mask of 5-texel width) + silhouette IoU | Directly tests closure and boundary exchange; boundary mask removes scalp-edge leakage |
| Secondary metric | Per-region strand-count drift (absolute difference in strand count per region between iterations 1000 and 3000) | Control metric not algebraically guaranteed to be zero |
| Baseline | Unsupervised braided hair reconstruction (single-view) | Tests structured-regime reconstruction sub-claim |
| Baseline | HairGS strand reconstruction | Tests photometric growth / free-form sub-claim |
| Baseline | Strand-aligned 3D Gaussians | Tests unstructured Gaussian avatar alternative |
| Ablation-of-ours | Fixed global regime (no flux split, no transport) | Isolates the conserved-flux mechanism |
| Ablation-of-ours | Transport sweep: dt in {0.01, 0.05, 0.1, 0.5} and D scaling {0.5x, 1x, 2x} | Tests stability of transport step and conservation-residual drift over iterations |
| Ablation-of-ours | kappa without curvature/scale terms (tangent-only) | Isolates the curvature/scale agreement contribution |

### Why this setup validates the claim

The central claim is that a single conserved flux split by a geometric compatibility switch yields closure over the hairstyle manifold without a learned router, and that this closure shows up specifically at braid-to-frizz boundaries. The single-view braided dataset is the right testbed because it concentrates instances where both generators (braid template and photometric growth) are simultaneously active and where a router would be forced to commit to a discrete decision. The braided-reconstruction baseline tests the structured sub-claim: it should do well on clean braids and degrade where free-form strands intrude, since it has no free-form channel. HairGS tests the free-form sub-claim and should degrade on ordered braid regions where its growth field lacks template structure. The strand-aligned Gaussian baseline tests whether the flux formulation is doing work beyond unstructured primitives. The ablation — freezing the regime split globally — is the falsification lever: if our gain is attributable to the conserved-flux exchange, removing it should reproduce the baseline failure pattern at boundaries while leaving interior regions intact. The primary metric is chosen because a mass-conservation residual (restricted to interior texels to avoid boundary leakage) plus silhouette IoU is exactly what would move if the closure argument holds, and would stay flat if the gain came from generic capacity. The secondary strand-count drift metric provides a non-tautological check that mass is not silently created or destroyed.

### Expected outcome and causal chain

**vs. Unsupervised braided hair reconstruction** — On a case where a braid frays into loose strands near the nape, the baseline produces a clean braid that terminates abruptly or an over-regularized strand bundle because its braid-template mechanism has no free-form channel to absorb the fray. Our method instead routes the frayed texels to `Phi_free` via low `kappa` (tangent, curvature, and scale disagreement) while keeping the braid core in `Phi_struct`, because the conserved flux split lets mass exchange across the boundary rather than forcing a discrete choice. We expect a noticeable silhouette-IoU gap concentrated on frayed-boundary crops but near parity on clean braid interiors.

**vs. HairGS** — On a case where tightly ordered braid ridges sit against diffuse frizz, the baseline produces blurry or merged strands at the ridges because its photometric growth field has no template prior to sharpen ordered flow. Our method instead assigns high `w_struct` along the ridges and grows template-aligned strands there, because `kappa` detects tangent, curvature, and scale agreement between template and growth. We expect a gap on ridge-definition metrics on braided regions but parity on fully free-form regions.

**vs. Strand-aligned 3D Gaussians** — On a case where a braid crosses a region of severe occlusion, the baseline produces strand mass that drifts or duplicates because unstructured Gaussians have no conserved quantity tying neighboring primitives. Our method instead keeps `|Phi - Phi_struct - Phi_free|` small (on interior texels) through the transport step, because flux is conserved by construction and diffusion redistributes rather than creates mass. We expect a lower conservation residual across occluded subsets, with the gap shrinking where occlusion is mild.

### What would falsify this idea

If the ablation with a frozen global regime matches our method's boundary-fidelity gain uniformly across all subsets — rather than our gain being concentrated on braid/free-form boundary crops and absent on uniform-regime interiors — then the conserved-flux exchange is not what produces the improvement and the central claim is wrong. Additionally, if the transport sweep shows that conservation-residual drift grows unboundedly for any dt in {0.01, 0.05, 0.1, 0.5} or D scaling, the transport step is not stabilizing the flux and the closure claim is unsupported.

## References

1. Unsupervised 3D Braided Hair Reconstruction from a Single-View Image
2. Neural Haircut: Prior-Guided Strand-Based Hair Reconstruction
3. HairGS: Hair Strand Reconstruction based on 3D Gaussian Splatting
4. 3D Gaussian Splatting for Real-Time Radiance Field Rendering
5. Human Hair Reconstruction with Strand-Aligned 3D Gaussians
6. GaussianHair: Hair Modeling and Rendering with Light-aware Gaussians
7. Dr.Hair: Reconstructing Scalp-Connected Hair Strands without Pre-Training via Differentiable Rendering of Line Segments
8. Point-NeRF: Point-based Neural Radiance Fields

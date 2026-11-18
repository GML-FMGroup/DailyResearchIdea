# Closure-Residual Coupling for 4D Scene Reconstruction

## Motivation

In the π^4 foundation model, the geometry branch (dense geometry, camera poses, metric scale) and the motion branch (image-space correspondences, 3D point tracks, moving-object masks) are coupled only by conditioning: the motion branch receives geometry as input, but nothing forces the lift operator from image correspondences to 3D tracks to be consistent with the geometry branch's own metric scale and poses. This unverified coupling is the structural root of temporal metric-scale drift: each branch can satisfy its own training objective while the composition of motion prediction and geometry lifting silently diverges, and no existing loss penalizes that divergence. The same gap recurs across the 4D reconstruction lineage—CAT4D optimizes a deformable Gaussian representation without a closed lift constraint, and ViDAR uses diffusion-based pseudo-supervision whose spatio-temporal inconsistency is corrected only by a heuristic loss—so the field lacks a testable composition invariant rather than merely a better backbone.

## Key Insight

The geometry–motion coupling becomes a closed, testable invariant when the lift operator is required to be a fixed point of the two branches' composition, so that any residual between lifted image correspondences and the motion branch's own 3D tracks is a directly measurable certificate of metric-scale inconsistency rather than an unobserved modeling gap.

## Method

**What it is.** We introduce CRC-4D (Closure-Residual Coupling for 4D), a training and evaluation procedure layered onto π^4 that adds a shared consistency loss over the lift operator. Inputs are uncalibrated RGB observations (single image, unordered collection, or video); outputs are the same π^4 predictions (dense geometry, camera poses, metric scale, 2D/3D point tracks, moving-object masks) plus a per-sequence closure residual that is minimized during training and reported at test time. The load-bearing assumption is stated explicitly: the closure residual r = ||T_lift − T_3d|| is treated as a certificate of geometry–motion branch inconsistency only under the assumption that the fixed lift operator built from the geometry branch's own depth, poses, and metric scale is a faithful, unbiased transport; this assumption fails when the geometry branch's depth is wrong (e.g., textureless or reflective regions), in which case r reflects depth error rather than branch inconsistency. To make this assumption testable rather than assumed, CRC-4D reports a per-region residual decomposition (static vs. dynamic, and depth-error vs. branch-inconsistency components) and includes a controlled synthetic experiment with ground-truth depth/poses injected to isolate the residual attributable to branch inconsistency.

**How it works.**

```pseudocode
# Given: geometry branch G, motion branch M (π^4 backbone, shared),
#        input views {I_v}, target views {I_t}
# Hyperparameters: lambda_crc = 1.0, lambda_scale = 0.1, tau_mask = 0.5

1. Geometry pass:
   (D, P, s) = G({I_v})            # dense depth, camera poses, metric scale

2. Motion pass (conditioned on geometry):
   (C_2d, T_3d, Mask) = M({I_v}, D, P, s)   # image-space correspondences,
                                            # 3D point tracks, moving-object masks

3. Lift operator L (fixed, differentiable):
   for each correspondence c in C_2d from view v to target view t:
       T_lift = s * P_t^{-1} * ( D_t[ pi(P_t * P_v^{-1} * K^{-1} * c) ] * K^{-1} * c )
   # i.e., lift c to 3D in view v using depth, then transport to target frame
   # using geometry-branch poses and metric scale s.

4. Closure residual:
   r = || T_lift - T_3d ||_2 * Mask      # masked by moving-object mask
   R_crc = mean(r)                        # per-sequence scalar

5. Scale-invariance probe (held-out sequences only):
   append K extra views to the sequence; recompute s'.
   R_scale = |log(s'/s)|                  # should vanish if closure holds

6. Joint training loss:
   L = L_geom + L_motion + lambda_crc * R_crc + lambda_scale * R_scale
   # L_geom, L_motion are π^4's original losses; lambda_crc, lambda_scale
   # are tuned on a held-out split (grid: {0.1, 0.5, 1.0, 2.0}).

7. Per-region residual decomposition (diagnostic, reported at test time):
   For each sequence, partition pixels into static (Mask=0) and dynamic (Mask=1).
   Report R_crc_static = mean(r | Mask=0) and R_crc_dynamic = mean(r | Mask=1).
   Additionally, compute a depth-error proxy: for each correspondence, compare D_t at the projected location to a held-out view's depth (when available) or to a multi-view photometric reprojection error; report the correlation between r and this proxy per region.

8. Controlled synthetic experiment (ground-truth injection):
   On a synthetic subset with known ground-truth depth D_gt and poses P_gt, compute T_lift_gt using D_gt, P_gt, and s_gt. Define r_gt = ||T_lift_gt - T_3d||_2 * Mask. The residual attributable to branch inconsistency is r_gt; the residual attributable to depth error is ||T_lift - T_lift_gt||_2 * Mask. Report both components and their sum.
```

**Why this design.** We chose a *fixed, differentiable lift operator* over a learned lift head because a learned head can absorb the residual and destroy the invariant (the learned head would simply learn to match T_3d regardless of geometric consistency), accepting the cost that the lift is only as accurate as the geometry branch's depth and poses. We chose to mask the residual by the motion branch's own moving-object mask rather than applying it uniformly because static regions are where metric-scale drift is most diagnostically visible and dynamic regions would otherwise dominate the residual, accepting the cost that mask errors leak into the loss. We chose to add the scale-invariance probe as a *training-time regularizer on held-out sequences only* rather than a post-hoc diagnostic because a post-hoc check cannot influence the solution, accepting the cost that the probe requires sequence-extension data during training. A domain expert would not describe CRC-4D as a variant of ViDAR's diffusion-aware loss: ViDAR corrects inconsistency between synthetic and real views via a heuristic loss on a Gaussian splatting representation, whereas CRC-4D imposes a fixed-point condition on the composition of two branches that already share a backbone, so the residual is a certificate about the branches' mutual consistency, not about synthetic-vs-real view agreement.

**Why it measures what we claim.** The closure residual R_crc measures *geometry–motion consistency* because the lift operator is constructed to be the unique composition of the geometry branch's metric scale, poses, and depth—the named assumption is that the geometry branch's depth and poses are locally accurate enough that the lift is a faithful transport; this assumption fails when depth is ambiguous in textureless or reflective regions, in which case R_crc reflects depth error rather than branch inconsistency. The scale-invariance residual R_scale measures *temporal metric-scale stability* because sequence extension should not change the metric scale if the closure holds—the named assumption is that the geometry branch's scale estimate is a function of the full observation set; this assumption fails when the added views are out-of-distribution, in which case R_scale reflects distribution shift rather than scale drift. The masked residual measures *consistency on static structure* because moving-object masks are assumed to correctly identify dynamic regions; this assumption fails when the mask misses a moving object, in which case the residual reflects motion-model error rather than scale inconsistency. The per-region decomposition (step 7) and the controlled synthetic experiment (step 8) operationalize the separation between depth-error and branch-inconsistency components, so that R_crc is reported as a consistency diagnostic whose confounds are explicitly modeled rather than as an unqualified certificate.

## Contribution

['A closure-residual coupling loss (CRC-4D) that enforces the lift operator from image-space correspondences to 3D tracks to be a fixed point under the geometry–motion composition, implemented as a shared consistency loss over the π^4 backbone.', 'A scale-invariance probe that uses sequence extension as a training-time regularizer and test-time diagnostic for temporal metric-scale drift, converting an unverified coupling into a measurable invariant.', 'An empirical demonstration that the closure residual vanishes on held-out sequences when the invariant is enforced and that metric scale remains invariant under sequence extension, providing a diagnostic for the 4D reconstruction lineage beyond π^4.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | π^4 mixed benchmarks: single image, unordered collection, video | Covers all three input regimes the method claims |
| Primary metric | Geometry–motion closure residual R_crc (lower better) | Directly measures the consistency certificate claimed |
| Secondary metric | R_crc_static, R_crc_dynamic, and depth-error proxy correlation | Tests whether residual concentrates on static structure and separates from depth error |
| Baseline | π^4 (unmodified backbone) | Tests whether CRC adds value over the source model |
| Baseline | ViDAR | Tests whether gains come from closure, not diffusion heuristics |
| Baseline | Flux4D | Tests whether gains survive an unsupervised 4D alternative |
| Ablation-of-ours | CRC-4D w/o masked residual (uniform residual) | Tests whether masking is what concentrates the gain |
| Ablation-of-ours | CRC-4D with learned lift head (instead of fixed lift) | Tests whether the fixed-point condition is non-absorbable |
| Ablation-of-ours | CRC-4D with cycle-consistency loss (instead of closure residual) | Tests whether closure is reducible to prior consistency losses |
| Controlled synthetic | Ground-truth depth/poses injected; report r_gt and depth-error component | Isolates branch inconsistency from depth error |

### Why this setup validates the claim

The central claim is that imposing a fixed-point closure condition across π^4's two branches yields a measurable, causally-attributable consistency improvement rather than a cosmetic loss term. Unmodified π^4 is the right baseline because it isolates exactly the delta CRC-4D contributes: same backbone, same outputs, same evaluation, so any R_crc gap is attributable to the closure loss and not to capacity. ViDAR is the right contrast because it is the closest published mechanism (diffusion-aware consistency on a splatting representation) and thus tests whether the fixed-lift formulation is doing work that a heuristic synthetic-vs-real loss cannot. Flux4D tests whether the gain survives against a fully unsupervised 4D reconstructor that never uses a shared backbone. The ablation — uniform instead of masked residual — is the load-bearing control: if masking is doing what we argue, the ablation should lose most of its benefit specifically on static-structure subsets, because that is where the mask is supposed to concentrate the diagnostic signal. The learned-lift-head ablation tests whether the fixed lift is necessary to prevent the residual from being absorbed, and the cycle-consistency ablation tests whether the closure condition is distinct from standard consistency losses. The controlled synthetic experiment with ground-truth depth/poses directly addresses the load-bearing assumption: by computing r_gt (branch inconsistency) and the depth-error component separately, we can verify that R_crc is not merely a proxy for geometry error. R_crc is the right primary metric because the claim is literally about the size of that residual; a downstream visual-quality metric would be confounded by rendering choices and would not distinguish "consistent branches" from "well-rendered but inconsistent branches." The scale probe R_scale is a secondary readout that should move with R_crc only if the closure story is correct.

### Expected outcome and causal chain

**vs. π^4** — On a sequence where the geometry branch's depth is locally accurate but the motion branch's 3D tracks drift slightly relative to the lifted correspondences, unmodified π^4 produces tracks that are individually plausible yet mutually inconsistent, because nothing in its training couples the two branches through a shared fixed-point condition. Our method instead penalizes exactly that composition mismatch via the fixed lift, forcing the motion branch to conform to the geometry branch's transport, so we expect a clear R_crc gap on sequences with mild branch disagreement and near-parity where the branches already agree. The per-region decomposition should show the gap concentrated on static structure (R_crc_static), with the depth-error proxy correlation remaining low, confirming that the improvement is not an artifact of depth error.

**vs. ViDAR** — On a monocular clip with a moving subject and sparse supervision, ViDAR's diffusion-aware loss corrects synthetic-vs-real view disagreement on its Gaussian splat, but it has no notion of whether its own geometry and motion predictions are mutually consistent, so on static background structure it can leave residual scale drift uncorrected. Our method instead measures the composition of the two shared-backbone branches directly, so we expect ViDAR to be competitive on dynamic-region appearance while our method shows a larger gap on static-structure closure and on the scale probe.

**vs. Flux4D** — On a large outdoor scene with rare or unknown moving objects, Flux4D's "as static as possible" regularization can absorb genuine motion into the static component when the photometric signal is ambiguous, producing tracks that are smooth but geometrically inconsistent with the reconstructed depth. Our method instead anchors motion to the geometry branch's lift, so we expect Flux4D to match us on scalability and generalization but to trail on R_crc in scenes with ambiguous dynamic objects.

**vs. CRC-4D w/o masked residual** — The uniform-residual variant applies the closure loss to all pixels equally, so dynamic regions with large motion dominate the gradient and the loss can be reduced by sacrificing static-structure consistency. We expect the masked version to show lower R_crc_static and a larger gap on static subsets, while the uniform variant may show similar overall R_crc but worse static performance.

**vs. CRC-4D with learned lift head** — A learned lift head can absorb the residual by learning to match T_3d regardless of geometric consistency, so we expect it to achieve low training R_crc but poor generalization to held-out sequences and a weaker correlation with the depth-error proxy, demonstrating that the fixed lift is necessary for the invariant to be non-absorbable.

**vs. CRC-4D with cycle-consistency loss** — A standard cycle-consistency loss enforces that lifting and projecting back recovers the original correspondence, but it does not enforce that the lifted 3D point matches the motion branch's own 3D track. We expect the cycle-consistency variant to show lower cycle error but higher R_crc than the closure residual, confirming that the fixed-point condition is a strictly different invariant.

**vs. controlled synthetic (ground-truth injection)** — On synthetic sequences with known depth and poses, we compute r_gt (branch inconsistency) and the depth-error component. We expect the depth-error component to be small in textureless regions when using ground-truth depth, and the branch-inconsistency component to dominate, validating that R_crc primarily measures branch inconsistency under the stated assumption.

### What would falsify this idea

If the R_crc improvement over unmodified π^4 is uniform across static and dynamic subsets rather than concentrated on static structure, or if the masked-residual ablation matches the full method, then the closure residual is not measuring branch consistency and the central claim is wrong. Likewise, if R_scale fails to track R_crc across the held-out extension probe, the scale-stability story does not hold. Additionally, if the controlled synthetic experiment shows that the depth-error component dominates r_gt even with ground-truth depth, then the residual is confounded by geometry error and the certificate claim is falsified.

## References

1. π^4: A Foundation Model for 4D Scene Geometry and Motion

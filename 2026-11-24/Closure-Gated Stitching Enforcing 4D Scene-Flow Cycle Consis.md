# Closure-Gated Stitching: Enforcing 4D Scene-Flow Cycle Consistency at the Reconstructor–Generator Interface

## Motivation

Model-stitched 4D world models such as Plan4D couple a frozen depth reconstructor (Depth Anything 3) with a frozen video generator (LongCat-Video) through an auxiliary DPT head, but the stitch has no formal rule requiring that depth, camera, and scene-flow predicted on the two sides of the interface describe the same 4D motion field; errors therefore accumulate silently across the boundary instead of being detected. This is a meta-level structural gap, not a local one: CAT4D, GenXD, and the fused view-time attention framework each enforce consistency only inside their own generative or reconstruction path, and none defines a closure invariant that a stitched interface must satisfy, so the field's trajectory toward composable frozen foundation modules inherits an unguarded seam.

## Key Insight

A differentiable scene-flow closure residual evaluated along closed spatio-temporal loops that cross the stitch boundary is identically zero for any physically realizable 4D scene and nonzero precisely when the two frozen modules disagree, so it converts an unobservable interface defect into a trainable, test-time-detectable signal without unfreezing either module.

## Method

**What it is.** We introduce CLOSURE-STITCH, a consistency head and training objective layered onto Plan4D's stitched architecture. Inputs are the observed context frames, the generator's predicted future RGB frames, and the DPT head's dense outputs (depth, camera intrinsics/extrinsics, scene flow); outputs are a scalar closure residual per spatio-temporal loop and a corrected scene-flow field obtained by distributing that residual back onto the DPT head's flow prediction.

**Load-bearing assumption (stated explicitly).** The closure residual r_L is intended to be nonzero only when the two frozen modules disagree at the stitch. This holds only if the true scene-flow field is integrable (a gradient of a per-timestep displacement potential) over the loop. In real dynamic scenes, occlusion/disocclusion, topology change, and non-rigid deformation make the true field non-integrable even for perfectly consistent geometry, so raw holonomy conflates seam defect with genuine scene non-integrability. To isolate the seam term, we (i) mask the residual to co-visible correspondences and (ii) subtract a null-model calibration computed from a single-module (non-stitched) baseline on the same clip, so the detector reports seam-specific excess holonomy rather than absolute non-integrability.

**How it works.**

```pseudocode
# Frozen: DepthAnything3 (DA3), LongCat-Video (LCV). Trainable: DPT head H, closure head C, occlusion head O.
Input: context frames I_ctx, text action a
1. F_hat, D_hat, K_hat, T_hat, S_hat = H(LCV(I_ctx, a))   # frames, depth, intrinsics, extrinsics, scene flow
2. # Build closed loops that cross the stitch boundary
   for each loop L = (t0 -> t1 -> t2 -> t0) sampled with stride dt in {1,2,4}:
       # forward path: generator-side flow, back-warped through DPT depth
       p_fwd = backwarp(S_hat[t0->t1], backwarp(S_hat[t1->t2], S_hat[t2->t0]))
       # return path: reconstructor-side flow from DA3 depth + camera pose
       S_da3[t_i->t_j] = unproject(D_hat[t_i], K_hat, T_hat) - unproject(D_hat[t_j], K_hat, T_hat)
       p_rev = compose(S_da3[t0->t2], S_da3[t2->t1], S_da3[t1->t0])
       # occlusion / co-visibility mask (3-layer MLP, hidden=128, GeLU, sigmoid output)
       m_L = O(D_hat[t0], D_hat[t1], D_hat[t2], T_hat[t0], T_hat[t1], T_hat[t2])  # per-pixel weight in [0,1]
       r_L_raw = C(p_fwd - p_rev, D_hat[t0], T_hat[t0])          # closure residual, C = 3-layer MLP w/ FiLM on depth
       r_L = m_L * r_L_raw
3. # Null-model calibration: run the same loop on a single-module baseline (DA3 depth + camera only, no generator)
   r_L_null = m_L * C(p_fwd_null - p_rev, D_hat[t0], T_hat[t0])  # p_fwd_null from DA3-only flow
   r_L_seam = r_L - r_L_null
4. L_closure = mean_L || r_L_seam ||^2 + lambda_cyc * ||warp(I[t0], p_fwd) - I[t0]||^2   (lambda_cyc = 0.5)
5. L_flow = ||S_hat - SG(S_da3)||^2  (stop-gradient on DA3 branch, weight 1.0)
6. L_total = L_flow + 1.0 * L_closure + 0.1 * L_rgb ; backprop only into H, C, O
7. At inference: r_L_seam > tau (tau = 0.02, set as 95th pct on held-out clean clips) flags a seam violation; S_hat is corrected by one Gauss-Newton step on r_L_seam before being emitted.
```

**Why this design.** We chose closed spatio-temporal loops over a single forward-backward pair because a two-frame check is satisfied by any invertible warp even when the underlying 4D field is non-integrable, whereas a loop over three timestamps has a nonzero holonomy exactly when the field is not a gradient of a displacement potential — accepting the cost of 3x more warp evaluations per step. We chose to keep Depth Anything 3 and LongCat-Video frozen and train only the DPT head, a small closure MLP, and a small occlusion MLP over end-to-end fine-tuning because unfreezing either foundation module would destroy the complementary priors Plan4D relies on and make the closure signal confounded with generator drift — accepting that the closure head can only correct, not re-synthesize, seam errors. We chose a stop-gradient on the DA3-derived flow in L_flow rather than a symmetric consistency loss because a symmetric loss lets the trainable head collapse both branches toward a trivially consistent constant flow, whereas the stop-gradient anchors the reconstructor side as a reference — accepting that DA3's own depth errors are treated as ground truth at the seam. We added the occlusion mask and null-model calibration because raw holonomy is nonzero for any non-integrable true field (occlusions, disocclusions), so without them r_L would conflate seam defect with genuine scene non-integrability; the mask restricts evaluation to co-visible correspondences, and the null model subtracts the holonomy expected from a single-module baseline on the same clip. A domain expert would not describe this as 'Plan4D plus a loss': Plan4D's DPT head is trained to match dense annotations, whereas CLOSURE-STITCH trains a residual that is zero by construction for any realizable scene and is used as a test-time detector and corrector, which is a different object than a supervised flow regressor.

**Why this measures what we claim.** The seam-specific closure residual `r_L_seam` measures seam inconsistency between the generator-side and reconstructor-side 4D fields because the named assumption is that any physically realizable dynamic scene induces a displacement field that is a gradient of a per-timestep potential, so its loop holonomy vanishes; this assumption fails when the scene contains occlusion boundaries or topology changes (e.g., an object entering/leaving the frame), in which case `r_L_seam` reflects genuine non-integrability of the true scene rather than a stitch defect — which is why we mask to co-visible correspondences and subtract the null-model holonomy. The cycle-photometric term `||warp(I[t0], p_fwd) - I[t0]||^2` measures whether the generator-side flow is grounded in observed pixels because the assumption is that Lambertian surfaces with sufficient texture make back-warping invertible; this fails on textureless or specular regions, where the term reflects photometric ambiguity rather than flow error. The stop-gradient flow loss `||S_hat - SG(S_da3)||^2` measures agreement between the trainable head and the frozen reconstructor because the assumption is that DA3's depth is locally accurate at the seam scale; this fails under large depth discontinuities, where the metric reflects DA3's own error rather than head error. The threshold `tau` measures whether a clip should be flagged because the assumption is that clean clips have closure residuals below the 95th percentile of the held-out distribution; this fails under distribution shift to novel scene types, where `tau` reflects calibration drift rather than true violation. The occlusion mask `m_L` measures co-visibility because the assumption is that the occlusion head can identify correspondences visible at all three timestamps; this fails under severe disocclusion where no co-visible correspondence exists, in which case the masked residual is undefined and the loop is discarded.

## Contribution

(1) A differentiable scene-flow closure residual and correction head (CLOSURE-STITCH) that operates at the frozen reconstructor–generator interface and yields a per-loop scalar plus a corrected flow field, trained jointly with Plan4D's DPT head under a stop-gradient anchor. (2) A design principle: loop holonomy over closed spatio-temporal paths is the minimal invariant that distinguishes true stitch inconsistency from ordinary per-frame prediction error, and it is measurable without unfreezing either foundation module. (3) A test-time seam-violation detector with a calibrated threshold and a one-step Gauss-Newton flow correction that can be applied to any model-stitched 4D pipeline.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Synthetic + real dynamic scenes with action-conditioned futures (Plan4D evaluation sets) | Matches Plan4D's own synthetic/real split and paired actions. |
| Primary metric | Seam closure-residual detection AUC vs. synthetic-injected seam violations (with human-annotated subset as sanity check) | Directly measures whether r_L_seam flags true stitch defects. |
| Baseline | Plan4D (frozen stitched generator + DPT head, no closure head) | Tests whether closure head adds detection over unmodified stitching. |
| Baseline | Forward-backward two-frame consistency check | Tests whether closed loops beat pairwise invertibility. |
| Baseline | Symmetric flow-consistency loss (no stop-gradient) | Tests whether stop-gradient avoids trivial-collapse. |
| Baseline | Standard cycle-consistency loss (Zhou et al. forward-backward) trained on the same DPT head | Tests whether loop-holonomy term is the novel ingredient. |
| Ablation-of-ours | CLOSURE-STITCH without Gauss-Newton correction (detector only) | Isolates correction contribution from detection. |
| Ablation-of-ours | CLOSURE-STITCH without occlusion mask and null-model calibration (raw holonomy) | Tests whether masking/calibration isolates seam term. |

### Why this setup validates the claim

The central claim is that a closed-loop holonomy residual, trained with a stop-gradient anchor, masked to co-visible correspondences, and calibrated against a single-module null model, isolates stitch-boundary inconsistency from genuine scene non-integrability. The Plan4D baseline tests whether the closure head adds anything beyond the unmodified stitched pipeline: if it does not, the residual carries no seam-specific signal. The two-frame forward-backward baseline tests the specific design choice of three-timestamp loops over pairwise invertibility, since a pairwise check is satisfied by any invertible warp even when the 4D field is non-integrable. The symmetric-loss baseline tests the stop-gradient rationale by exposing whether a trainable head collapses both branches toward trivially consistent constant flow. The standard cycle-consistency baseline tests whether the loop-holonomy term is the novel ingredient beyond known flow-consistency losses. The detector-only ablation isolates whether the Gauss-Newton correction actually improves emitted flow beyond flagging. The no-mask/no-calibration ablation tests whether the occlusion mask and null-model subtraction are necessary to isolate the seam term from genuine non-integrability. The primary metric — detection AUC against synthetic-injected seam violations — is the right one because the claim is specifically about the residual's diagnostic selectivity, not overall generation quality, which would conflate generator drift with seam defects. Synthetic injection (perturbing DA3 depth or camera pose at the boundary) provides labeled positives without costly human annotation; a small human-annotated subset (e.g., 100 clips) serves as a sanity check.

### Expected outcome and causal chain

**vs. Plan4D** — On a clip where a dynamic object crosses the stitch boundary between generated and reconstructed regions, Plan4D's unmodified DPT head emits a scene flow that is locally plausible but globally non-integrable across the seam, because nothing in its supervised dense-annotation objective penalizes loop holonomy. Our closure head instead computes r_L_seam over three-timestamp loops and back-propagates a residual that is zero by construction for realizable scenes, so we expect a clear AUC gap on seam-crossing clips but near-parity on clips whose motion stays within one region.

**vs. Forward-backward two-frame consistency** — On a clip with a rotating object whose displacement field is invertible frame-to-frame but accumulates non-zero holonomy over three timestamps, the pairwise check reports near-zero inconsistency because any invertible warp satisfies it. Our method instead composes three warps into a closed loop, so the accumulated holonomy surfaces as a nonzero r_L_seam, and we expect the two-frame baseline to miss a substantial fraction of flagged seams that our method catches, with the gap widening as stride dt increases.

**vs. Symmetric flow-consistency loss** — On a clip with sparse texture where the trainable head has slack, the symmetric loss lets both branches drift toward a constant flow that trivially minimizes the consistency term, so the head's flow degrades on textured regions. Our stop-gradient anchors the DA3 branch as a fixed reference, so the head cannot collapse, and we expect the symmetric variant to show lower flow error on textureless subsets but markedly higher error on textured subsets, while ours stays stable across both.

**vs. Standard cycle-consistency loss (Zhou et al.)** — On a clip with a seam-crossing object, the standard cycle-consistency loss (forward-backward flow agreement) is satisfied by any invertible warp and does not penalize loop holonomy, so it fails to detect seam violations that our holonomy residual catches. We expect our method to show higher detection AUC on seam-crossing clips, isolating the loop-holonomy term as the novel ingredient.

**vs. CLOSURE-STITCH without occlusion mask and null-model calibration** — On clips with heavy occlusion or disocclusion, raw holonomy is nonzero even for a perfectly consistent stitch, so the detector produces false positives. Our masked and calibrated residual subtracts the null-model holonomy and restricts to co-visible correspondences, so we expect higher AUC and lower false-positive rate on occlusion-heavy clips, with the gap widening as occlusion severity increases.

**vs. Detector-only ablation** — On seam-crossing clips, the detector-only variant flags violations but does not correct the emitted flow, so the flow error remains. Full CLOSURE-STITCH applies one Gauss-Newton step on r_L_seam, and we expect a measurable reduction in emitted-flow error (e.g., 10–20% lower EPE on seam-crossing clips) while matching the detector-only variant on detection AUC.

### What would falsify this idea

If the detection AUC gain over the two-frame baseline is uniform across seam-crossing and within-region clips rather than concentrated on the seam-crossing subset, then r_L_seam is not measuring stitch-boundary non-integrability but generic flow noise, and the central claim fails. Likewise, if the detector-only ablation matches full CLOSURE-STITCH on emitted-flow error, the correction step adds nothing and the 'detector and corrector' framing is wrong.

## References

1. Plan4D: Generative Plannable 4D Worlds

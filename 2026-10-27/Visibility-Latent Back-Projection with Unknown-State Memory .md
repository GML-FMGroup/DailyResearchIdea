# Visibility-Latent Back-Projection with Unknown-State Memory Attention for World-Consistent Video Generation

## Motivation

Existing large-angle camera-controlled video generators (WorldCast, Uni3C) back-project a single initial frame into a deterministic point cloud and treat it as a complete geometric guide, while their memory attention aggregates only observed tokens; neither represents that a 3D region was never seen. Structurally, this forces the generator to silently hallucinate occluded content with no consistency constraint and lets memory attention leak observed tokens into unobserved regions, because the pipeline has no state for 'unknown'. The field has the depth/pose/back-projection machinery and the memory-attention machinery, but lacks a structural coupling that makes visibility an explicit latent.

## Key Insight

Making per-point visibility an explicit latent whose variance gates both the geometric guidance and the memory read/write mask turns unobserved-region synthesis from an unconstrained hallucination into a constrained inference problem, because the generator can only place content in low-visibility regions through a learned prior over unobserved structure rather than through leakage of observed tokens.

## Method

**What it is.** VILA (Visibility-Latent Back-projection) replaces WorldCast's deterministic point cloud with a per-pixel distribution over 3D positions whose variance is driven by a visibility field, and couples that field to GAMA's memory-attention mask so unobserved tokens are aggregated under an explicit 'unknown' state. Inputs: initial frame, depth, relative camera poses, target trajectory. Outputs: geometric guidance with per-point visibility variance, an unknown-state memory bank, and generated video.

**How it works.**
```pseudocode
# Phase 1: Visibility-latent back-projection
D, sigma_D = depth_head(I0)                 # per-pixel depth + depth uncertainty
V = visibility_field(D, I0)                # per-pixel visibility logit in [0,1], 1=observed
mu_3d = backproject(I0, D, K, pose0)        # mean 3D position per pixel
sigma_3d = f(sigma_D, 1 - V)               # variance driven by depth unc. AND low visibility
P_dist = { (mu_3d[i], sigma_3d[i], V[i]) }  # per-point distribution, not a point cloud

# Phase 2: Render distribution along target trajectory
for each target pose pt:
    G_t, Cov_t, V_t = render_distribution(P_dist, pt)   # guidance, coverage, visibility map
    # V_t < tau_v (tau_v = 0.3) marks unknown regions

# Phase 3: Unknown-state memory attention (GAMA-U)
memory = []
for each generated frame t:
    tokens_t = patchify(frame_t)
    vis_t = V_t resampled to token grid
    for each memory frame m:
        rel_pose = pose_t^{-1} pose_m
        # read mask: observed tokens read normally; unknown tokens read an 'unknown' embedding
        read_mask = (vis_m > tau_v)
        agg = sum_m [ read_mask * attn(tokens_t, tokens_m)
                    + (1-read_mask) * attn(tokens_t, E_unknown) ]
    # write mask: only observed tokens written; unknown tokens write E_unknown
    write_mask = (vis_t > tau_v)
    memory.append( write_mask * tokens_t + (1-write_mask) * E_unknown )
    frame_t = denoise(tokens_t | G_t, Cov_t, agg)

# Phase 4: Prior-consistency loss on low-visibility regions
L_prior = E[ (1 - V_t) * || frame_t - Prior_unobs(G_t, Cov_t) ||^2 ]
L_total = L_diff + lambda_prior * L_prior   # lambda_prior = 0.5
```

**Why this design.** We chose a per-point distribution over a deterministic point cloud because a single depth estimate cannot express 'this region was never seen', accepting the cost of doubling the geometric guidance channels (mean + variance). We chose to drive variance by both depth uncertainty and visibility rather than depth uncertainty alone because depth noise and occlusion are structurally different failure modes, accepting that the visibility field must be supervised (we use the CLACT multi-view coverage masks as weak supervision). We chose an explicit 'unknown' embedding E_unknown in the memory read/write mask over simply zeroing unobserved tokens because zeroing is indistinguishable from a black observed token, accepting an extra embedding table and the risk that E_unknown collapses to a constant. We chose a prior-consistency loss on low-visibility regions over an adversarial hallucination loss because the prior must be a distribution over unobserved structure, not a discriminator, accepting that the prior is only as good as the training distribution of occluded content. A domain expert would not describe this as WorldCast + a variance head, because WorldCast's GAMA has no write mask and no unknown state — the coupling between visibility variance and the memory mask is the mechanism, not an add-on.

**Why it measures what we claim.** The visibility field V measures 'this 3D region was never seen' because we assume the coverage mask from rendering the back-projected distribution along the target trajectory is a sufficient statistic for observability; this assumption fails when the initial frame itself is blurry or the depth head is overconfident, in which case V reflects depth-head confidence rather than true occlusion. The per-point variance sigma_3d measures 'uncertainty over unobserved structure' because we assume depth uncertainty and visibility are independent contributors to back-projection error; this assumption fails when depth errors are correlated with occlusion boundaries, in which case sigma_3d overestimates uncertainty at object silhouettes. The read/write mask with E_unknown measures 'memory does not leak observed tokens into unobserved regions' because we assume the token grid resampling of V_t preserves the observed/unobserved boundary; this assumption fails when a token straddles the boundary, in which case the mask reflects a majority vote rather than a hard constraint. The prior-consistency loss measures 'synthesis in low-visibility regions is conditioned on a prior over unobserved structure' because we assume the prior network Prior_unobs trained on CLACT's multi-view observations generalizes to the target trajectory's occluded regions; this assumption fails when the target trajectory observes a region never seen in any CLACT training view, in which case the loss reflects the prior's training distribution rather than the true unobserved structure.

## Contribution

(1) A visibility-latent back-projection framework (VILA) that replaces the deterministic point cloud with a per-point 3D distribution whose variance is driven by a visibility field, and couples that field to a memory-attention read/write mask with an explicit 'unknown' state. (2) A design principle: unobserved-region synthesis in camera-controlled video generation is a constrained inference problem whose support is set by the visibility field, not an unconstrained hallucination problem. (3) A prior-consistency training objective that supervises low-visibility regions against a learned prior over unobserved structure rather than against the observed frame.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CLACT benchmark (large-angle trajectories) | Matches WorldCast; stresses occlusion from large camera motion |
| Dataset | Synthetic scenes with ground-truth visibility (e.g., Kubric or Habitat) | Provides pixel-accurate occlusion masks for calibration |
| Primary metric | Region-split FVD (observed vs unobserved) | Isolates gain where visibility variance is claimed to act |
| Secondary metric | AUROC of V vs. ground-truth visibility on synthetic | Tests sufficiency of V as observability statistic |
| Baseline | WorldCast | Tests deterministic point cloud without unknown state |
| Baseline | Uni3C | Tests 3D guidance without visibility-driven memory masking |
| Baseline | Wan (14B) | Tests strong generator lacking explicit occlusion handling |
| Ablation-of-ours | VILA w/o E_unknown (zeroed unknown tokens) | Tests whether explicit unknown embedding drives the gain |
| Ablation-of-ours | VILA w/o visibility-driven variance (sigma_3d from depth unc. only) | Isolates variance channel from coupling |
| Ablation-of-ours | VILA w/o write mask (memory writes all tokens) | Tests necessity of write mask for preventing leakage |

### Why this setup validates the claim

The central claim is that coupling a visibility-driven per-point variance to an explicit unknown-state memory mask improves synthesis specifically in regions the target trajectory never observed. CLACT is the right dataset because its large-angle trajectories guarantee substantial unobserved content, so the predicted effect has room to appear. The region-split FVD is the right metric because it separates observed from unobserved regions: a method that merely improves general video quality would lift both, whereas our mechanism predicts a concentrated lift on the unobserved subset. WorldCast tests the sub-claim that a deterministic point cloud cannot express "never seen"; Uni3C tests whether 3D guidance alone, without visibility-coupled masking, suffices; Wan tests whether raw generative capacity substitutes for explicit occlusion handling. The ablation isolates the unknown embedding, testing whether the benefit comes from the explicit unknown state rather than from variance alone. To ground the sufficiency assumption, we add a synthetic dataset with ground-truth visibility and measure AUROC of V against it; this directly tests whether V is a calibrated observability statistic rather than a coverage proxy. The additional ablations (w/o visibility-driven variance, w/o write mask) isolate the coupling mechanism from its components.

### Expected outcome and causal chain

**vs. WorldCast** — On a case where the target trajectory swings behind an object seen only from one side in the initial frame, WorldCast back-projects a single deterministic depth point cloud and GAMA reads/writes all tokens uniformly, so the unobserved side inherits observed texture and produces smeared, view-inconsistent content. Our method assigns high sigma_3d and low V to that region, marks it unknown, and reads E_unknown instead of leaking observed tokens, so we expect a clear FVD gap concentrated on the unobserved subset with near-parity on the observed subset.

**vs. Uni3C** — On a case where large camera motion reveals a region occluded at frame zero, Uni3C's 3D world guidance supplies geometry but its control signal has no notion of per-region observability, so it renders plausible-but-hallucinated structure that drifts across frames. Our method conditions denoising on G_t and Cov_t and gates memory through the visibility mask, so we expect our unobserved-region FVD to be lower while observed-region FVD stays comparable, with the gap widening as the trajectory's unobserved fraction grows.

**vs. Wan (14B)** — On a case where the trajectory passes through a region never seen in the initial frame, Wan's strong prior generates high-fidelity but temporally inconsistent content there, since nothing constrains it to a consistent unobserved state. Our method's prior-consistency loss and unknown-state memory anchor that region to a single prior distribution, so we expect Wan to match or beat us on observed-region FVD while we lead on unobserved-region FVD and on cross-frame consistency within unobserved regions.

**vs. VILA w/o E_unknown** — On the same unobserved-region cases, zeroing unknown tokens is indistinguishable from a black observed token, so the memory attention can still leak observed texture into unobserved regions through the zeroed tokens. We expect full VILA to outperform this ablation on unobserved-region FVD, confirming that the explicit unknown embedding is necessary for the coupling to work.

**vs. VILA w/o visibility-driven variance** — Using only depth uncertainty for sigma_3d (no V term) removes the coupling between visibility and geometric guidance, so the variance channel no longer signals 'never seen' but only 'depth noise'. We expect this ablation to perform similarly to full VILA on observed regions but worse on unobserved regions, isolating the contribution of the visibility-driven variance.

**vs. VILA w/o write mask** — Without the write mask, memory writes all tokens including unknown ones, so observed tokens can be written into unobserved regions and later read back, causing leakage. We expect this ablation to show increased FVD on unobserved regions and temporal inconsistency, confirming that the write mask is essential for preventing leakage.

### What would falsify this idea

If our improvement over WorldCast and Uni3C is uniform across observed and unobserved subsets rather than concentrated on the unobserved subset, or if the E_unknown ablation matches full VILA, then the visibility-variance-to-memory-mask coupling is not the mechanism and the central claim is wrong. Additionally, if V's AUROC against ground-truth visibility on synthetic scenes is near chance (≤0.6), then V is not a sufficient statistic for observability and the method's gating signal is invalid.

## References

1. WorldCast: World-Consistent Character Video Generation with Large-Angle Camera Control
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation

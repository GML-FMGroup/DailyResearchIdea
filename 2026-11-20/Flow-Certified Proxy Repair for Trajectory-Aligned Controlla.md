# Flow-Certified Proxy Repair for Trajectory-Aligned Controllable Video Generation

## Motivation

Across the controllable-video-generation tree, trajectory-to-output alignment is asserted at generation time but never re-verified after refinement or repair: FloVD conditions on optical flow yet offers no post-hoc check that generated background flow still matches the requested camera path, IM-Zero injects motion priors in a refinement stage without certifying that the injected priors preserved the user-specified trajectories, and Coda-DaVinci's agentic harness drives repairs purely through visual inspection and independent review, which cannot detect a proxy-granularity adaptation that silently drifts the rendered motion away from the requested path. The structural root cause is that every one of these systems treats the trajectory as an input prompt rather than as an invariant that must hold on the output, so the repair loop has no measurable closure condition and terminates on human judgment instead of on a verified alignment residual.

## Key Insight

Because optical flow estimated from a generated video is a directly comparable observable to the user-specified trajectory, the trajectory constraint can be made a checkable closure invariant on the output — so proxy-granularity adaptation and repair become certified decisions with a measurable residual rather than unverifiable visual judgments.

## Method

**What it is.** We introduce CERT-Repair (Certified Trajectory Repair), a closed-loop certifier that wraps Coda-DaVinci's agentic proxy-guided generation. Its inputs are the user-specified trajectory (object/instance point tracks plus identity references) and the video produced by any proxy-granularity choice or repair action; its output is a per-entity alignment residual vector that either certifies the video or routes a targeted re-repair request back to the agent harness.

**Load-bearing assumption (stated explicitly).** The certifier's closure rule assumes that optical flow estimated by RAFT on a generated video, restricted to an instance mask and integrated over time, yields a trajectory residual whose error is small relative to the certification threshold tau_traj, so that the residual is a calibrated, monotone measure of true trajectory-to-output alignment rather than of flow-estimation noise. This assumption fails when the instance deforms non-rigidly or is occluded, in which case r_i^traj reflects apparent rather than true motion and the certifier may falsely pass or fail. To make the closure condition a calibrated residual rather than an unvalidated constant, we replace the fixed absolute threshold with a per-video, per-entity calibration: we estimate the flow-residual floor by running the identical Certify pipeline on a known-aligned reference (a synthetic warp of the requested trajectory applied to the input keyframe sequence), and we certify only when r_i^traj exceeds that measured noise floor by a margin. We additionally gate certification on a flow-confidence/occlusion mask so occluded frames are excluded from the Chamfer integral rather than silently averaged in.

**How it works.**

```pseudocode
Input: user trajectory T = {(instance i, frame t, point p_it)}, identity refs R
Output: certified video V*, or per-entity repair directives

PHASE 0  Calibrate noise floor (once per video):
  V_ref <- synthetic_warp(input_keyframes, T)   # known-aligned reference
  for each instance i:
    F_i^ref <- RAFT_flow(V_ref, M_i)
    T_hat_i^ref <- integrate(F_i^ref)
    floor_i <- Chamfer(T_hat_i^ref, T_i)          # measured flow-residual floor
  tau_traj_i <- floor_i + margin (margin = 0.01 normalized)

PHASE 1  Generate:  V0 <- CodaDaVinci(task, proxy_granularity g_0)

PHASE 2  Certify(V):
  for each instance i with mask M_i(V):
    F_i <- RAFT_flow(V, M_i)                      # dense optical flow, mask-restricted
    C_i <- flow_confidence(F_i)                    # forward-backward consistency + boundary divergence
    M_i^valid <- M_i AND (C_i > c_min)             # c_min = 0.5; occluded/low-confidence frames excluded
    T_hat_i <- integrate(F_i, M_i^valid)           # re-estimated trajectory on valid frames only
    r_i^traj <- Chamfer(T_hat_i, T_i)              # trajectory residual
    r_i^id   <- 1 - cos(CLIP_embed(crop_i(V)), CLIP_embed(R_i))   # identity residual
  r <- [r_i^traj, r_i^id]_i
  return r

PHASE 3  Closed loop:
  r <- Certify(V0)
  while exists i with (r_i^traj > tau_traj_i or r_i^id > tau_id):
     i* <- argmax_i (r_i^traj / tau_traj_i + lambda * r_i^id / tau_id)   # lambda = 1.0
     g_{i*} <- GranularityPolicy(r_{i*})                   # refine granularity for worst entity only
     V <- CodaDaVinci.repair(V, entity=i*, directive=gradient_descent_step(r_{i*}))
     r <- Certify(V)
  return V
```

**Why this design.** We chose a mask-restricted optical-flow re-estimation over a learned alignment classifier because flow is a physically interpretable observable that is directly comparable to the input trajectory, accepting the cost that flow estimation fails under severe occlusion and must then be flagged rather than scored. We chose per-entity residual attribution over a single global residual because Coda-DaVinci's granularity adaptation is already per-entity, accepting the cost of extra flow passes per instance. We chose a threshold-triggered targeted re-repair over unconditional re-generation because re-generating the whole video destroys already-certified entities, accepting the cost that the loop may stall on a hard entity and needs a max-iteration cap (we set 6). We chose to keep Coda-DaVinci's agent harness as the repair executor rather than replace it, because the harness already preserves scene assets and the certifier only needs to supply a measurable closure condition — a domain expert would not describe this as a variant of Coda-DaVinci because the certifier is an external invariant checker whose decision rule is fixed by the calibrated residual thresholds, not an agent policy, and would not describe it as retrieval or as a router since no library lookup or confidence gating is involved.

**Why it measures what we claim.** The trajectory residual r_i^traj measures trajectory-to-output alignment because optical flow integrated over a mask is, under the assumption that the mask tracks a rigid or near-rigid instance and that flow error is bounded well below the calibrated threshold, an unbiased estimate of that instance's true image-space motion; this assumption fails when the instance deforms non-rigidly or is occluded, in which case r_i^traj reflects apparent rather than true motion and the certifier may falsely pass or fail. The identity residual r_i^id measures identity preservation because CLIP embedding cosine distance is monotone in perceptual identity similarity under the assumption that CLIP's embedding space is locally metric for the object category; this assumption fails for fine-grained or rare categories, in which case r_i^id reflects generic semantic similarity rather than instance identity. The calibrated threshold tau_traj_i measures the claim that alignment is verifiable because a per-video residual bound derived from a known-aligned reference is equivalent to a bounded trajectory error only if flow estimation error is small relative to the margin; when it is not, the threshold certifies noise rather than alignment.

## Contribution

(1) CERT-Repair, a closed-loop certification mechanism that converts trajectory-to-output alignment from an input prompt into a checkable output invariant, with per-entity optical-flow and identity residuals supplying a measurable closure rule for Coda-DaVinci's repair loop. (2) A design principle showing that proxy-granularity adaptation in agentic video generation can be driven by a fixed residual-gradient directive rather than visual inspection, with thresholds that make repair termination auditable. (3) An evaluation protocol that reports per-entity alignment residuals alongside generation quality, exposing which entities the proxy policy systematically fails to align.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Frame In-N-Out trajectory benchmark | Provides user-specified motion trajectories with identity refs |
| Primary metric | Per-entity trajectory residual (Chamfer, normalized) | Directly measures trajectory-to-output alignment claim |
| Secondary metric | Fraction of instances certified under occlusion/deformation gate | Quantifies how often the rigid-motion assumption holds |
| Baseline | Coda-DaVinci (direct generation) | Tests whether certifier loop adds alignment gain |
| Baseline | Coda-DaVinci (proxy-guided, fixed coarse granularity) | Tests whether targeted re-repair beats static granularity |
| Baseline | IM-Zero (zero-shot motion control) | Tests alignment against an independent motion-control method |
| Ablation-of-ours | CERT-Repair with global residual (no per-entity attribution) | Isolates the value of per-entity residual attribution |
| Ablation-of-ours | CERT-Repair with fixed absolute threshold (no per-video calibration) | Tests whether calibration is necessary for valid closure |

### Why this setup validates the claim

The central claim is that a calibrated, per-entity residual certifier wrapped around Coda-DaVinci's harness produces measurably better trajectory alignment without destroying already-good entities. The Frame In-N-Out benchmark is the right dataset because it supplies exactly the inputs the certifier consumes — user-specified motion trajectories plus identity references — so residuals are computed against ground-truth intent rather than a proxy. The primary metric, per-entity normalized Chamfer residual, is the right detector because it is the same quantity the certifier thresholds; if the loop works, this metric must drop on entities the loop targeted. The secondary metric, the fraction of instances certified under the occlusion/deformation gate, directly quantifies the load-bearing assumption: if the gate excludes most instances, the certifier is operating outside its validity regime and the alignment gains are suspect. The direct-generation baseline tests whether the certifier loop adds anything at all; the fixed-coarse-granularity baseline tests whether targeted re-repair beats Coda-DaVinci's own granularity adaptation when it is frozen; IM-Zero tests whether alignment gains are an artifact of Coda-DaVinci's harness by comparing to an independent motion-control method. The global-residual ablation is the decisive control: if per-entity attribution is the causal ingredient, removing it should degrade the worst-entity residual specifically, not all entities uniformly. The fixed-threshold ablation tests whether the calibration step is necessary: if the calibrated threshold does not outperform a fixed absolute threshold, then the load-bearing assumption is not being addressed and the closure rule is unvalidated. Together these form a falsifiable test because each baseline isolates one sub-claim of the mechanism.

### Expected outcome and causal chain

**vs. Coda-DaVinci (direct generation)** — On a case where an entity's trajectory requires fine-grained proxy structure but the agent's default granularity is coarse, direct generation produces a video whose entity drifts off the specified track because nothing measures the residual against the user trajectory. Our method instead computes a per-entity Chamfer residual, detects the drift, and routes a targeted re-repair for that entity only. We expect a noticeable residual gap concentrated on hard-trajectory entities, with near-parity on easy entities that already pass certification on the first pass.

**vs. Coda-DaVinci (proxy-guided, fixed coarse granularity)** — On a case where coarse proxies favor natural motion but a single entity needs medium-grained structure for consistency, the fixed-granularity baseline produces a video that is globally natural yet locally inconsistent for that entity because its granularity policy never adapts to a measured residual. Our method instead refines granularity only for the worst entity, driven by the residual gradient. We expect the gap to appear as a lower worst-entity residual for our method while the mean residual across all entities stays close, showing the gain is targeted rather than global.

**vs. IM-Zero (zero-shot motion control)** — On a case with a long trajectory and a breaking-new-identity reference, IM-Zero produces plausible coarse motion but drifts in identity and fine trajectory because its zero-shot refinement injects priors without a closed-loop alignment check against the specified trajectory. Our method instead certifies identity and trajectory residuals per entity and re-repairs until both fall below threshold. We expect our method to show a clearer separation on the identity residual for newly entering entities, with a smaller but consistent trajectory-residual advantage on long tracks.

**vs. CERT-Repair with fixed absolute threshold (no per-video calibration)** — On videos where flow estimation noise is large relative to the fixed threshold, the fixed-threshold variant will either certify noise (false pass) or stall on an unachievable bound, because its closure rule is not calibrated to the actual flow-residual floor. Our method instead measures the floor per video and certifies only when the residual exceeds that floor by a margin. We expect the calibrated variant to show a lower false-certification rate (measured as the fraction of certified videos whose true trajectory error, computed from ground-truth keypoints, exceeds the fixed threshold) and a higher completion rate within the max-iteration cap.

### What would falsify this idea

If the global-residual ablation matches the per-entity certifier's worst-entity residual gains, then per-entity attribution is not the causal ingredient and the central claim collapses. Likewise, if gains are uniform across easy and hard trajectory subsets rather than concentrated where the predicted drift failure occurs, the certifier is not detecting the mechanism we claim. Additionally, if the calibrated-threshold variant does not reduce false certifications relative to the fixed-threshold variant, then the load-bearing assumption is not being addressed and the closure rule remains unvalidated.

## References

1. Coda-DaVinci: Reimagining Video Generation with Code2Video

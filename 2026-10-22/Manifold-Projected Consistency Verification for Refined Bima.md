# Manifold-Projected Consistency Verification for Refined Bimanual Dexterous Tracking Policies

## Motivation

Iterative refinement pipelines for dexterous bimanual tracking — DONIC's behavioral-cloning/PPO feedback loop and ManipTrans's residual fine-tuning — both optimize action likelihood or constraint satisfaction without any explicit check that the refined policy still lies on the reference trajectory's structural manifold. Because behavioral cloning maximizes per-step action likelihood independently of trajectory-level structure, a refined policy can drift off the reference's monotone-progress and contact-consistent waypoint structure while still scoring high on likelihood, so success recycling and residual correction silently reinforce off-manifold executions rather than the human reference they were meant to track.

## Key Insight

The reference trajectory's structural manifold — monotone task progress plus contact-consistent waypoint transitions — is a differentiable constraint that behavioral cloning leaves unenforced, so projecting refined rollouts back onto it via a consistency loss makes refinement drift detectable and correctable at the trajectory level rather than the action level.

## Method

**What it is.** Manifold-Projected Consistency Verification (MPCV) is an auxiliary training objective added to a dexterous bimanual tracker (DONIC-style BC+PPO loop). It takes the reference trajectory and the refined policy's rollouts, computes a differentiable projection of each rollout onto the reference's structural manifold, and back-propagates a consistency loss that penalizes off-manifold deviation alongside the behavioral-cloning and PPO losses. The structural manifold is defined per-trajectory from the reference demonstration itself: a progress coordinate s_t^ref = phi_ref(o_t^ref) obtained by fitting a monotone-or-piecewise-linear model to the reference's normalized geodesic distance to goal (allowing backtracking segments), and a contact-role segmentation c_t^ref obtained by fitting a per-task contact-role classifier on the reference's (q_t^ref, o_t^ref) pairs. The load-bearing assumption, stated explicitly, is: the reference trajectory's structural manifold (its own progress profile and contact-role segmentation) is recoverable from object pose and hand joints by a per-trajectory model fitted to the reference, so that penalizing deviation from it corrects drift rather than imposing an arbitrary inductive bias. This assumption is verified per-task by held-out calibration of phi_ref and f_contact on the reference demonstration before refinement begins.

**How it works.**

```pseudocode
Input: reference trajectory R = {(o_t, q_t^ref, c_t^ref)}_{t=1..T}
       (object pose, hand joint targets, contact-role labels per hand)
       refined policy pi_theta, rollout buffer D
Hyperparams: lambda_prog=0.5, lambda_contact=0.3, lambda_way=0.2, tau_prog=0.1

# 0. Per-trajectory manifold calibration (frozen before refinement)
#    (a) Fit progress model phi_ref: normalized geodesic distance to goal,
#        piecewise-linear with K=5 knots, monotone segments allowed to reset
#        at detected backtracking points (velocity sign changes in o_t^ref).
#        Calibration: fit on first 80% of R, validate on last 20%; require
#        R^2 > 0.9 on held-out progress, else fall back to global monotone phi.
#    (b) Fit contact-role classifier f_contact: 2-layer MLP, hidden=256, GeLU,
#        input [q_t^ref, o_t^ref], output per-hand role logits (3 roles:
#        support, transport, free). Pre-train on all BiHOI reference contacts
#        (N=512 labeled frames per task), freeze before refinement.
#        Calibration: report ECE and per-role accuracy on held-out 20% of R;
#        require ECE < 0.1 and per-role accuracy > 0.85, else flag task.

for each refinement iteration:
    # 1. Roll out refined policy
    for each demonstration in D:
        tau = rollout(pi_theta, demo)          # (o_t, q_t, c_t)_{t=1..T}

    # 2. Structural manifold projection (differentiable)
    #    (a) progress coordinate s_t in [0,1] from object pose via
    #        the per-trajectory phi_ref (not a global monotone map)
    s_t = phi_ref(o_t)
    #    (b) enforce consistency with reference progress profile:
    #        projected progress s_t_proj = projection onto phi_ref's
    #        piecewise-linear profile (isotonic within monotone segments,
    #        reset at reference backtracking points), differentiable via
    #        softmax surrogate: s_t_proj = sum_k softmax(-|s_t - s_k^ref|/tau_s) * s_k^ref
    #        with tau_s=0.1.
    L_prog = mean_t ||s_t - s_t_proj||^2

    #    (c) contact-consistent waypoints: for each hand, compare the
    #        reference contact role c_t^ref to the rollout's contact role c_t
    #        via the frozen per-task classifier f_contact(q_t, o_t)
    L_contact = mean_t BCE(f_contact(q_t, o_t), c_t^ref)

    #    (d) waypoint anchoring: nearest reference waypoint in progress-time
    #        w(t) = argmin_k |s_t_proj - s_k^ref|, anchor joint targets
    L_way = mean_t ||q_t - q_{w(t)}^ref||^2

    # 3. Total loss
    L_BC = behavioral cloning loss on D
    L_PPO = PPO clipped objective
    L_MPCV = lambda_prog*L_prog + lambda_contact*L_contact + lambda_way*L_way
    L_total = L_BC + L_PPO + L_MPCV

    # 4. Update + recycle verified successes into D
    theta <- theta - lr * grad(L_total)
    D <- D + {tau : L_MPCV(tau) < tau_prog}
```

**Why this design.** We chose a differentiable isotonic projection (cumulative-max with a softmax surrogate) over a hard monotonicity filter because a hard filter yields zero gradient on already-monotone rollouts and cannot correct small drifts before they compound, accepting the cost that the surrogate introduces a small bias in the progress gradient. We chose contact-role consistency via a fixed differentiable classifier over a learned discriminator because a learned discriminator would itself drift with the policy and cannot serve as a stable reference, accepting the cost that the fixed classifier may be imperfect on out-of-distribution contacts. We chose progress-time nearest-waypoint anchoring over global trajectory matching because global matching is invariant to temporal misalignment and would not penalize a rollout that visits the right waypoints in the wrong order, accepting the cost that nearest-waypoint assignment can be ambiguous when waypoints are dense. Unlike ManipTrans's residual fine-tuning, which corrects actions under interaction constraints but never checks trajectory structure, and unlike DONIC's success-recycling loop, which recycles executions based on task success alone, MPCV makes the structural manifold itself a training signal; a domain expert would not describe this as a variant of residual learning or success recycling because the loss operates on trajectory-level structural quantities (progress monotonicity, contact-role consistency, progress-time waypoint anchoring) rather than on action residuals or binary success labels.

**Why it measures what we claim.** The progress-monotonicity loss L_prog measures *trajectory-consistency preservation under refinement* because the projection s_t_proj is the closest point on the reference's own progress profile to the rollout's progress coordinate, so ||s_t - s_t_proj||^2 is zero exactly when the rollout follows the reference's progress profile (including its backtracking segments); this assumption fails when the per-trajectory phi_ref is mis-fit (held-out R^2 < 0.9), in which case L_prog penalizes legitimate reference structure rather than drift. The contact-role loss L_contact measures *contact-consistent waypoint structure* because the frozen per-task classifier f_contact maps (q_t, o_t) to the same contact-role label space as the reference, so BCE is zero exactly when the rollout's contact role matches the reference's; this assumption fails when the classifier is miscalibrated on novel object geometries (held-out ECE > 0.1 or per-role accuracy < 0.85), in which case L_contact reflects classifier error rather than contact inconsistency. The waypoint-anchoring loss L_way measures *proximity to the reference manifold at the correct progress-time* because nearest-waypoint assignment in progress-time aligns rollout states to reference states with the same task progress; this assumption fails when progress-time assignment is ambiguous (dense waypoints, i.e., >10 waypoints per unit progress), in which case L_way reflects assignment noise rather than manifold deviation.

## Contribution

(1) A trajectory-consistency verification module (MPCV) that projects refined policy rollouts onto the reference trajectory's structural manifold via a differentiable isotonic-progress projection, a fixed contact-role classifier, and progress-time waypoint anchoring, added as an auxiliary training objective to a BC+PPO dexterous bimanual tracker. (2) A design principle: refinement drift in dexterous bimanual tracking is a trajectory-level structural phenomenon that action-level likelihood and residual correction cannot detect, and enforcing manifold consistency as a training signal reduces off-manifold executions that success-recycling would otherwise reinforce. (3) A verified-success recycling criterion based on manifold-consistency loss rather than task success alone, which changes which executions enter the demonstration buffer.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | BiHOI bimanual sequences (229 objects) | Same distribution as DONIC tracker baseline |
| Primary metric | Reference-tracking success rate | Directly captures manifold-consistent execution |
| Baseline | DONIC tracker | Tests structural signal beyond success recycling |
| Baseline | ManipTrans residual fine-tuning | Tests trajectory structure vs action residuals |
| Baseline | Object-centric dexterous policy | Tests bimanual contact-role structure necessity |
| Ablation-of-ours | MPCV minus L_contact term | Isolates contact-role consistency contribution |
| Ablation-of-ours | MPCV with global monotone phi (no per-trajectory calibration) | Tests necessity of per-trajectory manifold calibration |
| Ablation-of-ours | MPCV with f_contact frozen at random initialization | Tests necessity of calibrated contact classifier |

### Why this setup validates the claim

The central claim is that making the trajectory's structural manifold an explicit training signal improves refinement beyond what action-level correction (ManipTrans) or success-only recycling (DONIC) can achieve. DONIC tests whether structural consistency adds anything over its success-recycling loop: if our gain over DONIC is real, it must concentrate on sequences where success alone does not enforce ordering or contact structure. ManipTrans tests whether trajectory-level structure beats residual action correction: if our gain over ManipTrans concentrates on temporally-structured tasks (backtracking-free but contact-transition-heavy), the mechanism is validated. The object-centric baseline tests whether bimanual contact-role structure matters at all versus single-object-centric tracking. The ablation (dropping L_contact) isolates whether the contact-role term, not just progress monotonicity, drives any gain. The additional ablations test the load-bearing assumption: replacing the per-trajectory phi_ref with a global monotone map tests whether calibration to the reference's own progress profile (including backtracking) is necessary; replacing the calibrated f_contact with a random-initialized frozen classifier tests whether contact-role calibration is necessary. Success rate is the right primary metric because it is the outcome the structural manifold is supposed to protect: a rollout that drifts off-manifold should fail the task, so a metric that directly counts task completion is the most falsifiable test of the causal chain from structural consistency to task success.

### Expected outcome and causal chain

**vs. DONIC tracker** — On a case where the refined policy drifts slightly off the reference manifold mid-trajectory (e.g., a bimanual re-grasp where one hand's contact role should switch from support to transport), the DONIC baseline produces a rollout that still reaches the goal and is recycled as a success, because its recycling criterion is binary task success and never inspects trajectory structure, so the drift compounds across iterations. Our method instead penalizes the off-manifold deviation via L_prog and L_contact before it compounds, because the per-trajectory projection and calibrated contact-role classifier give gradient exactly where the drift begins. We expect a noticeable success-rate gap concentrated on contact-transition-heavy sequences, with near-parity on simple single-contact sequences.

**vs. ManipTrans residual fine-tuning** — On a case where the rollout visits the correct waypoints but in a slightly wrong temporal order (e.g., beginning to close the second hand before the first hand's contact is established), ManipTrans produces a rollout that corrects the action residuals locally but preserves the misordered structure, because its residual objective is action-level and never checks trajectory-level ordering. Our method instead penalizes the misordering through progress-time nearest-waypoint anchoring and progress consistency, because L_way and L_prog are sensitive to temporal structure that action residuals cannot see. We expect a gap on temporally-structured bimanual tasks and parity on tasks with a single dominant waypoint.

**vs. Object-centric dexterous policy** — On a case where both hands must maintain complementary contact roles on the same object (e.g., one stabilizing while the other manipulates), the object-centric baseline produces a rollout where both hands attempt the manipulating role, because its objective is object-centric and does not encode per-hand contact-role structure. Our method instead enforces per-hand contact-role consistency against the reference via L_contact, because the calibrated per-task classifier maps each hand's (q_t, o_t) to the reference's role label space. We expect a gap concentrated on bimanual complementary-contact sequences, with parity on single-hand-dominant sequences.

**vs. MPCV with global monotone phi** — On a case where the reference itself contains a backtracking segment (e.g., hand-over-hand regrasping), the global monotone phi penalizes the legitimate backtrack, because it imposes a monotone constraint that the reference violates. Our method with per-trajectory phi_ref instead projects onto the reference's own progress profile, so L_prog is zero on the backtrack. We expect a gap concentrated on backtracking sequences, with parity on monotone sequences.

**vs. MPCV with random-initialized f_contact** — On a case where contact roles switch mid-trajectory, the random-initialized classifier produces uninformative role logits, so L_contact is noise. Our method with calibrated f_contact (held-out ECE < 0.1) gives meaningful contact-role gradients. We expect a gap concentrated on contact-transition-heavy sequences, with parity on static-contact sequences.

### What would falsify this idea

If our gain over DONIC and ManipTrans is uniform across all sequence subsets rather than concentrated on contact-transition-heavy and temporally-structured subsets, the structural-manifold mechanism is not the cause of any improvement. Likewise, if the L_contact ablation shows no degradation on complementary-contact sequences, or if the per-trajectory calibration ablation shows no degradation on backtracking sequences, the contact-role and progress-profile terms are not measuring what we claim.

## References

1. DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations
2. ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning
3. Object-Centric Dexterous Manipulation from Human Motion Data
4. DexH2R: Task-Oriented Dexterous Manipulation From Human to Robots

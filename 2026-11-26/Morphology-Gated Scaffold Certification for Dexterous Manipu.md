# Morphology-Gated Scaffold Certification for Dexterous Manipulation

## Motivation

Existing dexterous manipulation pipelines that bootstrap from coarse references — vision-language-model scaffolds (Scaffolding Dexterous Manipulation with Vision-Language Models) and distilled programmatic policies (PyRUA-Dex) — consume references without ever checking whether the target hand's kinematics, joint limits, and contact geometry can realize them. This is a structural blind spot, not a tuning issue: the reference generator (a VLM or a coding agent) operates in a space that is agnostic to hand morphology, so infeasible segments are only discovered downstream, where the residual RL tracker or program interpreter either silently distorts the motion or fails, wasting training signal and masking the true cause of failure.

## Key Insight

A reference segment is executable by a specific hand morphology iff its keypoint trajectory lies in the intersection of per-keypoint reachability envelopes and its implied contact set is realizable by the hand's link geometry — a condition that can be checked by differentiable forward kinematics and contact-consistency residuals before any low-level tracker or interpreter is invoked, turning an assumed-executable reference into a certified one.

## Method

**What it is.** We introduce MORPHOGATE, a morphology-aware feasibility gate that sits between any coarse reference source (VLM scaffold or distilled program segment) and its downstream consumer (residual RL tracker or program interpreter). Its inputs are a reference segment (a sequence of keypoint positions, object poses, and implied contact events) and a target hand model (link lengths, joint limits, palm geometry); its output is a per-segment feasibility certificate plus, when infeasible, a repaired segment that is certified.

**Load-bearing assumption (stated explicitly).** MORPHOGATE assumes that a reference segment's keypoint trajectory and implied contact flags are a faithful, noise-free specification of the intended task, so that a nonzero differentiable-FK reachability residual or contact-consistency residual can be attributed to hand-morphology infeasibility rather than to error in the reference itself. This assumption is load-bearing: the reference sources are exactly the noisy ones (a VLM that identifies keypoints and synthesizes coarse 3D trajectories, a coding agent that distills programmatic segments whose keypoints are not metrically calibrated to any hand, and diffusion planners that can produce ghost states where the object moves without hand contact). To make the gate's claim conditional on attribution rather than on raw residual magnitude, we add a two-hypothesis attribution test (Phase 3 below) that compares the residual under the target hand model against the residual under a high-DoF 'oracle' hand and under reference-perturbation resampling; only morphology-attributable segments are certified FEASIBLE/REPAIRED/REJECTED, while non-attributable segments receive an UNCERTAIN/REFERENCE-NOISE label and are passed through ungated. This requires no change to the downstream tracker or interpreter.

**How it works.**

```pseudocode
Input: reference segment R = {(k_t, o_t, c_t)}_{t=1..T}  # keypoints, object poses, contact flags
Input: hand model H = (FK, joint_limits, link_geometry)
Input: oracle hand model H_oracle = (FK_oracle, joint_limits_oracle, link_geometry_oracle)  # high-DoF, e.g., 30-DoF anthropomorphic hand
Output: certificate cert in {FEASIBLE, REPAIRED, REJECTED, UNCERTAIN}, repaired R'

# Phase 1: Reachability projection (target hand)
for t in 1..T:
    q_t = IK_solve(FK(H), k_t, warm_start=q_{t-1}, max_iter=50)   # joint config
    r_t = ||FK(H, q_t) - k_t||                                     # reachability residual
    viol_t = max(0, |q_t| - joint_limits)                          # joint-limit violation

# Phase 2: Contact consistency (target hand)
for each contact event (t, link_l, object_o) in c:
    d_t = signed_distance(FK_link(H, q_t, l), o_t)                # penetration if < 0
    contact_res_t = |d_t| if d_t > 0 else |d_t| + lambda_pen * max(0, -d_t)  # lambda_pen=10

# Phase 3: Two-hypothesis attribution test
# 3a: residual under oracle hand (same procedure as Phases 1-2 but with H_oracle)
for t in 1..T:
    q_t_oracle = IK_solve(FK_oracle(H_oracle), k_t, warm_start=q_{t-1}_oracle, max_iter=50)
    r_t_oracle = ||FK_oracle(H_oracle, q_t_oracle) - k_t||
    viol_t_oracle = max(0, |q_t_oracle| - joint_limits_oracle)
for each contact event (t, link_l, object_o) in c:
    d_t_oracle = signed_distance(FK_link(H_oracle, q_t_oracle, l), o_t)
    contact_res_t_oracle = |d_t_oracle| if d_t_oracle > 0 else |d_t_oracle| + lambda_pen * max(0, -d_t_oracle)

# 3b: reference-perturbation resampling (N=10 jitter samples per keypoint)
for n in 1..N:
    k_t^(n) = k_t + epsilon_t^(n),  epsilon_t^(n) ~ N(0, sigma_jitter^2 I),  sigma_jitter = 0.005 m
    # recompute r_t^(n), viol_t^(n), contact_res_t^(n) under target hand H
    # (same IK and contact procedures as Phases 1-2)

# 3c: attribution decision (per segment)
residual_target = max_t (r_t + viol_t + contact_res_t)
residual_oracle = max_t (r_t_oracle + viol_t_oracle + contact_res_t_oracle)
residual_jitter_std = std_n( max_t (r_t^(n) + viol_t^(n) + contact_res_t^(n)) )

if residual_target <= tau_feas:                              # tau_feas = 0.01 m
    cert = FEASIBLE; R' = R
elif residual_target <= tau_repair:                          # tau_repair = 0.05 m
    if (residual_target - residual_oracle) > tau_attr and residual_jitter_std < tau_jitter:  # tau_attr = 0.02 m, tau_jitter = 0.005 m
        R' = project_to_feasible(R, H)   # gradient descent on residuals, 20 steps, lr=0.01
        cert = REPAIRED
    else:
        cert = UNCERTAIN; R' = R   # pass through ungated
else:
    if (residual_target - residual_oracle) > tau_attr and residual_jitter_std < tau_jitter:
        cert = REJECTED; R' = None
    else:
        cert = UNCERTAIN; R' = R   # pass through ungated
return cert, R'
```

**Why this design.** We chose a differentiable forward-kinematics residual over a learned feasibility classifier because the hand model is known exactly and the residual is a direct geometric quantity, accepting the cost that the gate cannot generalize to unmodeled contact phenomena (e.g., soft-finger deformation). We chose a closed-form thresholded decision rule over a learned gating head because thresholds are interpretable and calibratable against the hand model, accepting the cost that thresholds must be set per hand morphology. We chose a repair-by-projection step over outright rejection because many VLM scaffolds are only slightly infeasible and a small projection recovers them, accepting the cost that repaired segments may drift semantically from the original task description. We chose to interface at the segment level (not per-timestep) so that the gate composes with both the residual RL tracker of Scaffolding Dexterous Manipulation with Vision-Language Models and the program interpreter of PyRUA-Dex without modifying either, accepting the cost that segment-level rejection is coarser than per-step rejection. We added the two-hypothesis attribution test (Phase 3) because the load-bearing assumption that reference keypoints and contact flags are faithful is not guaranteed for the noisy reference sources we target; the test makes the gate's claim conditional on attribution rather than on raw residual magnitude, accepting the cost of an additional oracle-hand IK solve and N=10 jitter resamples per segment.

**Why this measures what we claim.** The reachability residual r_t measures morphological realizability because the assumption that the hand's reachable set is exactly the image of its forward kinematics under joint limits makes r_t = 0 equivalent to the keypoint being reachable; this assumption fails when the hand has compliant or underactuated joints whose effective workspace differs from the kinematic model, in which case r_t reflects kinematic-model error rather than true infeasibility. The joint-limit violation viol_t measures kinematic admissibility because the assumption that joint limits are hard constraints makes viol_t > 0 equivalent to an inadmissible configuration; this assumption fails when the controller uses soft limits or torque-limited joints, in which case viol_t reflects a soft-boundary excursion rather than a hard infeasibility. The contact residual contact_res_t measures contact-set consistency because the assumption that the reference's contact flags are correct makes a nonzero signed distance equivalent to a contact that the hand cannot realize; this assumption fails when the reference's contact flags are themselves noisy, in which case contact_res_t reflects reference noise rather than morphological infeasibility. The attribution test (Phase 3) measures whether a residual is morphology-attributable because the assumption that a large target-hand residual paired with a small oracle-hand residual and low jitter sensitivity indicates morphological infeasibility makes the difference (residual_target - residual_oracle) > tau_attr and residual_jitter_std < tau_jitter equivalent to the segment being infeasible for the target hand but not for a higher-DoF hand and stable under keypoint noise; this assumption fails when the oracle hand is also infeasible (e.g., the reference is semantically impossible) or when the reference is so noisy that jitter sensitivity is high, in which case the test emits UNCERTAIN and the segment is passed through ungated rather than mis-certified.

## Contribution

["A morphology-aware feasibility gate (MORPHOGATE) that certifies or repairs coarse reference segments against a target hand's kinematics, joint limits, and contact geometry before downstream execution, with a closed-form thresholded decision rule and a gradient-based repair projection.", 'A design principle: reference generators and reference consumers should be decoupled by an explicit feasibility certificate, so that infeasibility is diagnosed at the reference boundary rather than masked by downstream tracker or interpreter failure.', 'An interface specification that makes the gate composable with both VLM-scaffold residual-RL pipelines and distilled-program interpreters without modifying either.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DexManipNet + HumanoidGen-generated bimanual episodes | Contact-rich, morphology-diverse, includes infeasible-by-construction segments |
| Primary metric | Segment-level feasibility-certificate precision/recall against ground-truth IK-solvability oracle | Directly measures gate's core claim |
| Baseline | Ungated reference (raw VLM scaffold / program segment) | Tests whether gating matters at all |
| Baseline | Learned feasibility classifier (MLP on keypoint poses) | Tests geometric residual vs. learned head |
| Baseline | Reject-only gate (no repair projection) | Tests whether repair recovers marginal segments |
| Ablation-of-ours | MORPHOGATE without contact-consistency phase | Isolates reachability vs. contact contribution |
| Ablation-of-ours | MORPHOGATE without attribution test (raw residual thresholding) | Isolates the contribution of the two-hypothesis attribution test |
| Ablation-of-ours | MORPHOGATE with controlled contact-flag noise injection (5%, 10%, 20% flipped flags) | Quantifies how precision degrades under reference noise |

### Why this setup validates the claim

The central claim is that a morphology-aware, differentiable-FK feasibility gate with closed-form thresholds and repair-by-projection improves downstream dexterous manipulation outcomes by filtering or repairing segments that a specific hand cannot realize. The dataset choice matters because DexManipNet and HumanoidGen episodes span multiple hand morphologies and contain contact-rich segments (pen capping, bottle unscrewing, door opening) where kinematic infeasibility is genuinely present rather than hypothetical; a dataset of only feasible segments would make the gate's contribution invisible. The primary metric is certificate precision/recall against an IK-solvability oracle because that oracle operationalizes the exact geometric quantity the method claims to measure: a segment is feasible iff its keypoints lie in the FK image under joint limits. If the gate's certificates disagree with the oracle, the residual does not measure what we claim. The ungated baseline tests the sub-claim that gating matters at all; the learned-classifier baseline tests the sub-claim that an explicit geometric residual beats a data-driven head on a known-morphology problem; the reject-only baseline tests the sub-claim that repair recovers marginally infeasible segments rather than merely discarding them; the no-contact ablation tests whether the contact-consistency phase contributes beyond pure reachability; the no-attribution ablation tests whether the two-hypothesis attribution test is necessary to avoid mis-certifying noisy references; and the contact-flag noise injection tests the load-bearing assumption that reference contact flags are correct by quantifying precision degradation under controlled noise. Together these form a falsifiable design: if the gate's advantage is uniform across baselines and ablations rather than concentrated where the predicted failure modes live, the mechanism story is wrong.

### Expected outcome and causal chain

**vs. Ungated reference** — On a case where a VLM scaffold specifies keypoints just outside the target hand's reachable set (e.g., a cabinet-handle keypoint requiring a palm orientation the 22-DoF hand cannot achieve), the ungated baseline passes the segment to the residual RL tracker, which then either fails to track or converges to a distorted pose that loses object contact because the tracker has no signal that the target is unrealizable. Our method instead flags the segment as REPAIRED and projects it to the nearest reachable keypoint configuration because the FK residual is exactly the quantity the tracker cannot see, so we expect a clear success-rate gap concentrated on the infeasible-segment subset and parity on already-feasible segments.

**vs. Learned feasibility classifier** — On a case where the hand morphology is swapped at test time (e.g., a 22-DoF hand replaced by a lower-DoF gripper-like hand), the learned classifier produces confidently wrong certificates because its decision boundary was fit to the training morphology's implicit workspace and does not encode joint limits or link geometry. Our method instead recomputes residuals from the new hand model's FK and joint limits because the gate is parameterized by the model rather than learned, so we expect the learned baseline to degrade sharply under morphology shift while our gate's precision stays roughly flat, with the gap largest on the shifted-morphology subset.

**vs. Reject-only gate** — On a case where a segment is only slightly infeasible (e.g., a keypoint 2–3 cm outside reach, within tau_repair), the reject-only baseline discards the entire segment because it has no repair mechanism, forcing the downstream consumer to fall back to exploration or a default policy and losing the semantic guidance the segment provided. Our method instead runs a 20-step gradient projection that recovers the segment and certifies it because the residual is differentiable with respect to keypoints, so we expect a higher fraction of marginally infeasible segments retained and a downstream success gain concentrated on the near-threshold band (residuals between tau_feas and tau_repair), with parity on clearly feasible and clearly infeasible segments.

**vs. MORPHOGATE without attribution test** — On a case where the reference keypoints are noisy (e.g., a VLM places a keypoint 2 cm off due to semantic ambiguity rather than morphological infeasibility), the no-attribution baseline uses raw residual thresholding and mis-certifies the segment as REPAIRED or REJECTED because it cannot distinguish reference noise from morphological infeasibility. Our method instead compares the residual under the target hand against the residual under a high-DoF oracle hand and under jitter resampling, emitting UNCERTAIN and passing the segment through ungated when the residual is not morphology-attributable, so we expect higher certificate precision on the noisy-reference subset and parity on clean-reference segments.

**vs. MORPHOGATE with controlled contact-flag noise** — On a case where contact flags are flipped at rates of 5%, 10%, and 20%, the no-attribution baseline's contact_res_t reflects reference noise rather than morphological infeasibility, causing precision to degrade monotonically with noise rate. Our method's attribution test flags high-jitter-sensitivity segments as UNCERTAIN, so we expect precision to degrade more slowly and to remain above the no-attribution baseline at all noise rates, with the gap largest at 20% noise.

### What would falsify this idea

If MORPHOGATE's gain over the ungated baseline is uniform across the feasible, marginally infeasible, and clearly infeasible subsets rather than concentrated on the marginal band where the repair projection is predicted to act, or if the learned classifier matches our gate under morphology shift, or if the attribution test does not improve precision on noisy-reference segments relative to raw residual thresholding, then the geometric-residual-plus-repair-plus-attribution mechanism is not what produces the benefit and the central claim is wrong.

## References

1. PyRUA-Dex: Learning Programmatic Policies for Dexterous Bimanual Manipulation with Coding Agents

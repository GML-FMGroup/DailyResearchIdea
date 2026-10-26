# Precondition-Audited Perception-Action Interface with Scale-Invariant Action Commitment

## Motivation

Across the VLM-agent and navigation frontiers, the perception-action interface is fixed before deployment and never audits its own preconditions: HexaShell hard-wires calibrated multi-view geometry with known planes for ray-plane intersection, and LightZeroNav assumes frontal RGB sufficiency, yet neither detects when calibration drift, plane misidentification, or monocular scale ambiguity invalidates that model at test time. The structural failure is not a missing perception module but an interface that hallucinates metric quantities (distances, plane intersections) precisely when the observation model's identifiability preconditions no longer hold, so downstream actions inherit errors the agent cannot see. This recurs because both trees treat the observation model as an axiom rather than a testable hypothesis.

## Key Insight

An observation model's calibration preconditions are themselves falsifiable from the agent's own action-conditioned prediction residuals, and once falsified, action correctness can be preserved exactly by restricting commitment to the scale-invariant quotient of the action space (relative bearing and topological progress) whose validity does not depend on the unidentifiable metric scale.

## Method

**What it is.** FARA (Falsification-Audited Residual Action) is a thin wrapper around any perception-action policy that, before each action, computes an action-conditioned prediction residual to test the current observation model's calibration/sufficiency preconditions, and when the test fires, projects the proposed action onto the scale-invariant quotient of the action space so that only components invariant to unidentifiable metric scale are executed. The load-bearing assumption is stated explicitly: the residual R is a valid, low-false-positive detector of the specific precondition violations (calibration drift, plane misidentification, monocular scale ambiguity) that FARA then neutralizes via quotient projection — i.e., R is both sensitive to those violations and quiet when they are absent. To make this assumption testable, FARA decomposes R into a scale-invariant shape term and a scale term, and audits the scale term against an external metric reference (known camera height, stereo/IMU baseline, or a metric anchor) so that the audit can detect the global affine scale ambiguity that is in the null space of a pure render-and-compare residual.

**How it works.**

```pseudocode
Input: observation model M (point-map + intrinsics), policy pi, history buffer H, threshold tau=0.15, window k=5,
       metric reference ref (known camera height h_cam, or stereo/IMU baseline b, or metric anchor set A)
Output: executed action a_exec

1. PROPOSE: a_metric = pi(obs)                      # policy's metric action (e.g., 3D waypoint)
2. PREDICT: for each of the last k actions in H, compute
       y_hat_t = M.render(H.obs_t, H.a_t)            # action-conditioned predicted observation
       r_t     = || y_hat_t - H.obs_{t+1} ||_geo     # geometric residual (reprojection + depth L1)
3. DECOMPOSE: for each r_t, split into
       r_t_shape = || y_hat_t - H.obs_{t+1} ||_geo after normalizing both by their median depth
       r_t_scale = | log( median_depth(y_hat_t) / median_depth(H.obs_{t+1}) ) |
4. AUDIT:   R_shape = median(r_t_shape over window k)
            R_scale = median(r_t_scale over window k)
            S = scale_identifiability_test(M, ref)   # returns 1 if scale is pinned by ref, else 0
            if R_shape <= tau and (S == 1 or R_scale <= tau_scale=0.05):
                M is calibrated -> a_exec = a_metric          (metric branch)
            else:
                M is falsified -> go to 5                     (scale-invariant branch)
5. QUOTIENT PROJECTION:
       decompose a_metric = (a_bearing, a_range) in the egocentric frame
       a_quotient = a_bearing                          # bearing + topological progress only
       a_exec = pi_quotient(a_quotient)                # low-level controller consumes bearing
6. LOG: append (obs, a_exec, next_obs, R_shape, R_scale, S) to H; return a_exec
```

**Why this design.** We chose a *residual-based falsification test over a learned calibration head* because the residual is a direct consequence of the observation model's own predictive claim, so it cannot be gamed by a network trained to look calibrated, accepting the cost that the test is only as sensitive as the geometric residual metric. We chose *median-over-window aggregation over a single-step residual* because single-step residuals are dominated by motion blur and occlusion noise, accepting the cost of a k=5-frame detection latency. We chose *projection onto the scale-invariant quotient over a full fallback policy* because bearing and topological progress are provably invariant to the unidentifiable monocular scale (MoGe/MoGe-2's affine-invariant representation makes this exact), accepting the cost that the quotient branch cannot exploit metric shortcuts even when they would be correct. We chose to *decompose the residual into shape and scale terms and audit the scale term against an external metric reference* because a global affine scale factor is in the null space of a pure render-and-compare residual, so a single scalar R cannot fire on the scale ambiguity it is designed to catch; the external reference (known camera height, stereo/IMU baseline, or metric anchor) pins the scale and makes the ambiguity detectable, accepting the cost that the audit requires one such reference to be available at test time. A domain expert would not describe FARA as a variant of HexaShell's tool-repair loop, because HexaShell repairs tools *after* failure using logs while FARA falsifies the observation model *before* acting and restricts the action space rather than repairing perception; nor is it a variant of AutoEval-style success detection, which judges outcomes, not preconditions.

**Why it measures what we claim.** The action-conditioned geometric residual `R_shape` measures *observation-model precondition violation* because the assumption that M's predicted observation equals the realized observation under the executed action is exactly the calibration/sufficiency precondition; this assumption fails when the scene violates M's structural assumptions (e.g., a plane is not actually planar, or monocular scale is unidentifiable), in which case R_shape reflects scene-model mismatch rather than calibration drift and FARA conservatively enters the quotient branch. The scale term `R_scale` together with the identifiability test `S` measures *scale identifiability* because the assumption that the point map's scale is pinned by an independent metric cue (known camera height, stereo/IMU baseline, or metric anchor) makes the global affine scale factor observable; this assumption fails when no such cue is available, in which case S=0 and the audit cannot certify the metric branch, so FARA enters the quotient branch. The projection onto `a_bearing` measures *scale-invariant action correctness* because the assumption that bearing and topological progress are invariant under the affine scale ambiguity of the point map makes the quotient action identical for all metric scales consistent with the observation; this assumption fails when the ambiguity is not a pure scale (e.g., a reflection or non-uniform affine distortion), in which case the quotient action is no longer guaranteed correct and the agent must fall back to a re-observation step. The median-over-window statistic measures *persistent* rather than transient precondition violation because the assumption that noise is zero-median over k frames makes the median converge to the systematic residual; this assumption fails under correlated motion blur, in which case the test may fire spuriously and the agent over-restricts to bearing-only actions.

## Contribution

(1) FARA, a self-auditing perception-action interface that falsifies its own observation model's calibration/sufficiency preconditions from action-conditioned prediction residuals before every action. (2) A scale-invariant quotient action policy that provably preserves action correctness under unidentifiable metric scale by committing only to bearing and topological-progress components. (3) A design principle showing that precondition auditing plus quotient projection dominates both fixed-observation-model agents and post-hoc tool-repair loops on layouts where calibration or scale assumptions are violated.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MoGe-2 mixed geometry benchmark, monocular-scale subset | Provides affine-invariant point maps where scale is genuinely unidentifiable |
| Dataset (second violation type) | HexaShell-style ray-plane scenes with injected intrinsics calibration drift (focal length perturbed by ±5%, ±10%) and non-planar surfaces (planes replaced by curved meshes) | Tests whether the audit generalizes beyond monocular scale to calibration drift and plane misidentification |
| Primary metric | Task success rate on scale-ambiguous vs. scale-identifiable splits | Directly isolates where quotient projection should help |
| Secondary metric | Audit detection ROC and detection latency (frames) on injected-violation episodes | Grounds the audit as a standalone capability and reports false-positive/false-negative rates as a function of tau |
| Baseline | Unwrapped policy (raw metric action) | Tests whether falsification+projection beats no wrapper at all |
| Baseline | Learned calibration head | Tests residual test against a gameable learned calibration signal |
| Baseline | Single-step residual gate | Tests whether median-over-window aggregation matters |
| Ablation-of-ours | FARA with quotient branch always-on | Tests whether the audit actually earns its keep |
| Ablation-of-ours | FARA with shape-only residual (no scale term, no identifiability test) | Tests whether the scale decomposition and external metric reference are necessary to detect scale ambiguity |

### Why this setup validates the claim

The central claim is that a residual-based falsification test on the observation model, followed by projection onto the scale-invariant quotient, recovers performance specifically where monocular metric scale is unidentifiable. The dataset choice makes this falsifiable: the monocular-scale subset of MoGe-2's benchmark contains scenes where the affine ambiguity is real, while the scale-identifiable subset provides a control where the metric branch should fire and no restriction should occur. The second dataset (HexaShell-style scenes with injected calibration drift and non-planar surfaces) tests whether the audit generalizes beyond monocular scale to other precondition violations, as the reviewer requested. The unwrapped-policy baseline tests the sub-claim that restricting the action space beats trusting the policy's metric action; the learned-calibration-head baseline tests the sub-claim that a residual derived from the model's own predictive claim is harder to game than a network trained to look calibrated; the single-step-residual baseline tests the sub-claim that median-over-window aggregation is what makes the test fire on persistent rather than transient violation. The always-on-quotient ablation is the key control: if FARA's audit contributes nothing, always-on quotient should match FARA on the ambiguous subset and only lose on the identifiable subset, which is a distinguishable pattern from the predicted one. The shape-only ablation tests the load-bearing assumption directly: if a pure render-and-compare residual is blind to global scale ambiguity, shape-only FARA should fail to fire on the ambiguous subset and underperform full FARA there. Success rate split by ambiguity is the right primary metric because it is the only quantity that separates the predicted causal story (gain concentrated where scale is unidentifiable) from generic improvement (uniform gain). The audit ROC and detection latency ground the threshold choice: we set tau=0.15 and tau_scale=0.05 from the null distribution of R_shape and R_scale on held-out clean sequences to target a 5% false-positive rate, and report empirical false-positive/false-negative rates as a function of tau on both splits.

### Expected outcome and causal chain

**vs. Unwrapped policy (raw metric action)** — On a scene where the point map's scale is off by an unknown affine factor, the unwrapped policy commits to a metric waypoint whose range component is wrong, so the end-effector overshoots or undershoots and the grasp or placement fails; its mechanism gap is that it has no way to detect that the metric component is unidentifiable. FARA's scale term R_scale and identifiability test S fire because the external metric reference (known camera height or stereo/IMU baseline) is inconsistent with the point map's scale, and it projects to bearing-only, which is invariant to the affine factor, so the low-level controller still makes topological progress. We expect a large success-rate gap on the scale-ambiguous subset and near-parity on the scale-identifiable subset, where the metric branch fires and FARA reduces to the unwrapped policy.

**vs. Learned calibration head** — On a scene where the point map is locally plausible but structurally wrong (a surface not actually planar), a calibration head trained on labeled calibration data sees in-distribution features and reports high confidence, so the baseline executes the full metric action and fails; its mechanism gap is that the head was trained to predict calibration, not to test the model's own predictive claim, so it can be confidently wrong exactly where structure violates the model's assumptions. FARA's shape residual R_shape is computed from the model's own render-versus-realized comparison, so the structural violation shows up as a persistent residual regardless of what any head believes, and the quotient branch engages. We expect FARA to show a clear advantage concentrated on the structurally-violating subset, with the calibration-head baseline trailing by a visible margin there while matching on clean scenes.

**vs. Single-step residual gate** — On a scene with motion blur or transient occlusion, a single-step residual spikes even though the observation model is fine, so the baseline spuriously enters the restricted branch, loses metric precision, and underperforms on tasks that needed range; its mechanism gap is that one frame of noise dominates its statistic. FARA's median over a five-frame window converges to the systematic component, so transient spikes are suppressed and the metric branch stays active. We expect the single-step baseline to lose mainly on the scale-identifiable subset due to spurious restriction, while FARA keeps parity there and retains its gain on the ambiguous subset, producing a crossover pattern rather than a uniform gap.

**vs. FARA with shape-only residual (no scale term, no identifiability test)** — On a scene where the point map is globally rescaled by an unknown affine factor, the shape-only residual re-renders consistently with itself and does not fire, so the baseline executes the full metric action and fails; its mechanism gap is that a global scale factor is in the null space of a pure render-and-compare residual. FARA's scale term R_scale and identifiability test S detect the inconsistency with the external metric reference and enter the quotient branch. We expect shape-only FARA to underperform full FARA on the scale-ambiguous subset while matching on the scale-identifiable subset, confirming that the scale decomposition and external reference are necessary to detect the ambiguity the paper is built around.

### What would falsify this idea

If FARA's gain over the unwrapped policy were uniform across the scale-ambiguous and scale-identifiable subsets rather than concentrated on the ambiguous subset, the claim that the residual test is detecting scale unidentifiability specifically would be wrong. Likewise, if the always-on-quotient ablation matched FARA on the ambiguous subset and lost only on the identifiable subset, the audit would be contributing nothing beyond the projection itself. If shape-only FARA matched full FARA on the scale-ambiguous subset, the scale decomposition and external metric reference would be unnecessary and the load-bearing assumption would be false.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. AlphaEvolve: A coding agent for scientific and algorithmic discovery
3. MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details
4. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
5. MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision
6. Multi-objective Evolution of Heuristic Using Large Language Model
7. Amplifying human performance in combinatorial competitive programming
8. Autonomous Improvement of Instruction Following Skills via Foundation Models

# Failure-Substitution Distillation: Recovery-Aware Cross-Morphology Policy Transfer for Quadrupedal Loco-Manipulation

## Motivation

On-policy distillation is support-preserving by construction: the student's KL bound confines it to the teacher's action manifold, so a student distilled from a demonstration-imitating teacher (WAM-Policy's OPSD stage, reaching only 76.7% versus 77.0% transferred) inherits the teacher's inability to represent recovery actions that never appear in successful demonstrations. WAM-Policy patches this with a post-hoc RL teacher-tuning stage decoupled from the distilled student, meaning the recovery competence is acquired by a slow specialist and only merged afterward via M-OPD rather than being present in the support the student is bounded to. The structural problem is that every distillation-based loco-manipulation pipeline treats the teacher's success manifold as the ceiling of what can be distilled, so failure recovery must be bolted on rather than made distillable.

## Key Insight

If the teacher's training distribution is augmented with synthesized failure-recovery trajectories before distillation, then the KL-bounded student's support provably contains recovery actions, and a failure-predicate-triggered distillation objective can make the student's recovery behavior a first-class distillation target rather than a post-hoc RL artifact.

## Method

**What it is.** FSD (Failure-Substitution Distillation) is a two-phase pipeline that modifies WAM-Policy's cross-morphology adaptation stage so the arm-centric WAM teacher is trained on a mixture of successful demonstrations and synthesized failure-recovery trajectories, then distills into a few-step student with a failure-substitution objective that fires when a learned failure predicate detects the student leaving the success manifold. Inputs: QLM-Bench demonstrations (1,000 per task, no teleoperation), a failure predicate, and a recovery-trajectory synthesizer. Outputs: a language-conditioned quadrupedal loco-manipulation policy with one video step and two action steps.

**Load-bearing assumption (stated explicitly).** The pipeline assumes that synthesized failure-recovery trajectories, after being filtered for physical feasibility and verified by the teacher, produce a teacher whose support contains physically valid, locally recovery-optimal actions such that the KL-bounded few-step student can inherit genuine recovery competence rather than the synthesizer's artifacts. This assumption is load-bearing: if the teacher's recovery branch is not a valid H=5 action-value oracle, the student is pulled toward infeasible actions precisely at failure states. We therefore make the assumption testable by (i) filtering synthesized recovery trajectories through a simulator constraint check and a learned feasibility critic, admitting only trajectories the teacher can execute without contact/balance violations, and (ii) keeping a frozen copy of the original success-only teacher and applying the KL term against it on the success manifold so the failure mixture cannot shift the support-preservation target.

**How it works.**

```pseudocode
PHASE 1 — Failure-conditioned teacher adaptation
  Input: arm-centric WAM W_arm, demos D_succ per task, failure predicate f_phi,
         simulator constraint checker C_sim, learned feasibility critic C_learn
  W_succ <- frozen copy of W_arm adapted on D_succ only   # support-preservation anchor
  for each task t:
    D_fail_raw <- SynthesizeRecovery(t, D_succ, f_phi)   # trip/fall recovery, object-drop re-grasp
    D_fail <- { tau in D_fail_raw : C_sim(tau)=0 and C_learn(tau) >= 0.5 }  # feasibility filter
    D_mix <- D_succ ∪ D_fail                         # mixture ratio 0.7 succ / 0.3 fail
    W_t <- AdaptMorphology(W_arm, D_mix)             # cross-morphology adaptation on mixture

PHASE 2 — Failure-substitution OPSD
  Input: teacher W_t, frozen success-only teacher W_succ, student S_theta (1 video step, 2 action steps)
  for each rollout step k of student on task t:
    a_S <- S_theta(o_k, l_t)
    if f_phi(o_k) = FAIL:                            # failure predicate fires
      target <- argmax_a Q_recovery(a | o_k, t)     # recovery-substitution target over D_fail
      L <- KL( S_theta(·|o_k) || W_t(·|o_k) ) + lambda * ||a_S - target||^2
    else:                                            # success manifold
      L <- KL( S_theta(·|o_k) || W_succ(·|o_k) )    # standard OPSD against frozen success-only teacher
    theta <- theta - eta * grad L                    # eta = 3e-4, lambda = 0.5
  return S_theta
```

The failure predicate `f_phi` is a binary classifier trained on the union of successful and failure-recovery rollouts from Phase 1, thresholded at 0.5; it is not a log-prob gate but a supervised failure detector whose training labels are the synthesized failure states. Concretely, `f_phi` is a 2-layer MLP (hidden=256, GeLU) over the concatenated observation `o_k` (proprioception + object pose + terrain heightmap patch), trained with binary cross-entropy for 10k steps, batch size 256, Adam lr=1e-3, class-balanced by oversampling failure states to 50%. The recovery target `Q_recovery` is a short-horizon (H=5) action-value estimate computed by rolling out the teacher's recovery branch from the failure state over the verified set `D_fail`, not a learned value head. The feasibility critic `C_learn` is a 2-layer MLP (hidden=256, ReLU) trained to predict whether a trajectory segment satisfies simulator contact/balance constraints, using the same architecture and training budget as `f_phi`; `C_sim` is the MuJoCo constraint checker reporting non-zero for joint-limit, self-collision, or foot-slip violations.

**Why this design.**

We chose to synthesize recovery trajectories and mix them into the teacher's training data (Phase 1) rather than run RL on the teacher afterward (WAM-Policy's second stage) because the KL-bounded student can only inherit what the teacher's support contains; mixing failure data into the teacher's distribution makes recovery actions distillable by construction, accepting the cost that synthesized recovery trajectories may be physically implausible and inject noise into the teacher. We mitigate this cost by filtering synthesized trajectories through `C_sim` and `C_learn` before mixing, and by anchoring the success-manifold KL to a frozen success-only teacher `W_succ` so the failure mixture cannot shift the support-preservation target. We chose a supervised failure predicate `f_phi` over a log-prob-based gate because log-probability is not calibrated for task correctness (a known open problem), accepting the cost that the predicate requires labeled failure states from Phase 1 and will miss novel failure modes outside its training distribution. We chose a short-horizon action-value recovery target `Q_recovery` over a full RL specialist because it keeps the distillation objective closed-loop and student-coupled, accepting the cost that the H=5 horizon may be too short for multi-step recovery (e.g., re-grasp after a drop requires longer coordination). A domain expert would not describe this as "WAM-Policy + failure data": WAM-Policy's second stage is a separate RL teacher-tuning loop whose output is merged via M-OPD, whereas FSD's recovery signal is injected into the distillation loss itself and is conditioned on a student-state failure predicate, so the student's recovery behavior is trained on-policy rather than merged post-hoc.

**Why it measures what we claim.**

The failure predicate `f_phi(o_k)` measures "the student has left the demonstrated success manifold" because the assumption that supervised binary classification on synthesized failure states is a sufficient statistic for manifold membership holds when the failure states span the relevant failure modes; this assumption fails when the student encounters a failure mode not represented in `D_fail` (e.g., a novel terrain perturbation), in which case `f_phi` reflects distributional novelty rather than true failure. The recovery target `Q_recovery` measures "the action that substitutes a recovery for the demonstrated success action" because the assumption that the teacher's recovery branch under the failure-conditioned mixture provides a valid H=5 action-value estimate holds when the teacher's recovery trajectories are locally optimal; this assumption fails when synthesized recovery trajectories are physically infeasible, in which case `Q_recovery` reflects the synthesizer's bias rather than a recovery-optimal action. The KL term on the success manifold measures "support preservation" because the assumption that the frozen success-only teacher `W_succ` is the correct target holds when `W_succ` was trained on `D_succ` only; this assumption fails when `W_succ` is itself biased by the failure mixture, in which case the KL reflects a shifted target rather than the original success policy.

## Contribution

(1) A failure-conditioned teacher adaptation procedure that mixes synthesized failure-recovery trajectories into the cross-morphology adaptation stage, making recovery actions part of the teacher's support before distillation. (2) A failure-substitution on-policy distillation objective that replaces the separate RL teacher-tuning stage with a closed-loop, predicate-triggered recovery target computed from the student's own rollout states. (3) A supervised failure predicate trained on synthesized failure states that provides a calibrated failure signal without relying on log-probability gating.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | QLM-Bench (10 tasks, 1,000 demos each) | Standard quadrupedal loco-manipulation benchmark, no teleoperation needed |
| Primary metric | Task success rate on failure-triggering episodes | Directly measures recovery substitution, not just nominal success |
| Baseline | Transferred arm-centric WAM policy | Tests whether failure-conditioned teacher adaptation helps at all |
| Baseline | WAM-Policy OPSD student | Tests whether on-policy substitution beats plain distillation |
| Baseline | WAM-Policy full (RL + M-OPD) | Tests whether in-loss recovery injection beats post-hoc merging |
| Ablation-of-ours | FSD without failure predicate (always standard OPSD) | Isolates the contribution of the substitution objective |
| Ablation-of-ours | FSD without feasibility filter (raw synthesized recovery) | Tests whether the load-bearing assumption holds without filtering |
| Ablation-of-ours | FSD with KL against mixed teacher (not frozen W_succ) | Tests whether support-preservation anchor matters |

### Why this setup validates the claim

The central claim is that recovery behavior must live inside the teacher's support and be distilled on-policy through a student-state failure predicate, rather than merged post-hoc as WAM-Policy does. QLM-Bench is the right dataset because it already contains the ten tasks and demonstration budget WAM-Policy reports on, so any gain is attributable to our pipeline change rather than data. The primary metric is success on failure-triggering episodes specifically, because nominal-task success would be insensitive to the substitution mechanism: a policy that never recovers but also rarely fails looks identical to one that recovers well. Each baseline tests a distinct sub-claim. The transferred arm-centric WAM tests whether failure-conditioned teacher adaptation grants the teacher recovery support at all. The plain OPSD student tests whether on-policy substitution beats standard distillation from that same teacher. The full WAM-Policy tests whether injecting recovery into the distillation loss beats WAM-Policy's separate RL plus M-OPD merge. The ablation removes the failure predicate and always applies standard OPSD, isolating whether the substitution objective itself, not just the extra mixed data, produces the gain. The additional ablations test the load-bearing assumption: without the feasibility filter, the teacher's recovery branch may be synthesizer-biased; without the frozen success-only teacher, the KL target is shifted by the failure mixture. If FSD beats all three baselines and the no-predicate ablation collapses to plain-OPSD performance, the causal story is supported; if the ablation matches FSD, the predicate and substitution are doing nothing and the claim is false. We also report the feasibility-filter admission rate (fraction of synthesized trajectories passing `C_sim` and `C_learn`) and the teacher's recovery-branch success rate on the admitted set, as a quantitative validation that the load-bearing assumption holds.

### Expected outcome and causal chain

**vs. Transferred arm-centric WAM policy** — On a case where the three-legged quadruped trips mid-stride during a front-leg manipulation task, the transferred WAM produces the demonstrated success action because its training support contains only successful demonstrations, so it has no recovery mode to sample. Our method instead routes the failure state through the failure-substitution objective, whose teacher was adapted on a 0.7/0.3 success-recovery mixture after feasibility filtering, so a recovery action is available in the teacher's support. We expect a clear success-rate gap concentrated on trip-and-fall episodes, with near-parity on episodes that never leave the success manifold.

**vs. WAM-Policy OPSD student** — On a case where the student drops the object during a re-grasp and the state drifts off the demonstrated manifold, the OPSD student imitates the teacher's success-manifold distribution because its KL target is the unmodified teacher, so it keeps emitting the pre-drop action and the object is lost. Our method instead detects the failure with the supervised predicate and substitutes an H=5 action-value recovery target computed over the verified recovery set, pulling the student back toward the re-grasp branch on-policy. We expect the gain to appear as fewer lost-object terminations and a moderate overall success lift, not as faster nominal sampling.

**vs. WAM-Policy full (RL + M-OPD)** — On a case where a novel-but-in-distribution failure mode requires a short coordinated recovery, the full WAM-Policy's student inherits a merged specialist that was tuned and merged post-hoc, so its recovery behavior is not coupled to the student's own failure states and can be mismatched at the moment of failure. Our method instead conditions the recovery signal on the student's own state through the predicate, so the substitution fires exactly when the student is off-manifold. We expect FSD to match or slightly exceed the full pipeline on recovery episodes while remaining clearly cheaper to train, with the gap largest on tasks whose failures are short-horizon.

**vs. FSD without feasibility filter** — On a case where synthesized recovery trajectories violate contact/balance constraints, the unfiltered teacher's recovery branch is synthesizer-biased, so `Q_recovery` reflects infeasible actions and the student is pulled toward them at failure states. Our method filters these trajectories through `C_sim` and `C_learn` before mixing, so the teacher's recovery support contains only physically valid actions. We expect the unfiltered variant to show lower success on failure-triggering episodes and higher joint-torque/contact-force violation rates during recovery, with the gap largest on tasks where raw synthesis is most unconstrained.

**vs. FSD with KL against mixed teacher** — On a case where the failure mixture shifts the teacher's success distribution, the mixed-teacher KL target is corrupted by the 0.3 failure mixture, so the student's success-manifold behavior drifts. Our method anchors the success-manifold KL to a frozen success-only teacher `W_succ`, so the support-preservation target is not shifted. We expect the mixed-teacher variant to show degraded nominal-task success and a larger divergence from the original success policy, with the gap largest on tasks where the failure mixture is most dissimilar to the success demonstrations.

### What would falsify this idea

If FSD's gain over the plain OPSD student is uniform across episodes rather than concentrated on failure-triggering episodes, the substitution mechanism is not what is producing the improvement, and the central claim is wrong. Equally, if the no-predicate ablation matches FSD, the failure predicate and recovery target are inert and the design reduces to WAM-Policy plus extra mixed data; and if the feasibility-filter admission rate is near zero or the teacher's recovery-branch success on admitted trajectories is low, the load-bearing assumption fails and the recovery signal is synthesizer bias rather than genuine recovery competence.

## References

1. WAM-Policy: World Action Models with On-Policy Distillation for Quadrupedal Loco-Manipulation

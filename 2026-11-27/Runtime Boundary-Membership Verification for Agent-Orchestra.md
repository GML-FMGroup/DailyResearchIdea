# Runtime Boundary-Membership Verification for Agent-Orchestrated Visuomotor Policies

## Motivation

Agent-orchestrated manipulation systems such as Boundary-Aware Policy Learning and MAESTRO both rely on a handoff protocol that is fixed at training time: Boundary-Aware Policy Learning shapes and expands a handoff boundary and communicates it to the agent once, while MAESTRO's plan-act-replan loop invokes tools without any execution-time check that the current state lies inside the region where the invoked policy is reliable. The structural root cause is that the boundary exists only as a training-time artifact — the agent's action set is not closed under out-of-boundary invocation, so a policy can be called from states it never saw, and neither branch supplies a runtime predicate that can veto the call. This is a meta-level closure gap: any system that composes a learned policy with a symbolic planner inherits it whenever the policy's competence region is declared statically rather than checked against the live state.

## Key Insight

The boundary classifier and the visuomotor policy are trained on the same contact-segment distribution, so the classifier's decision surface is calibrated by construction to the policy's actual competence region — a property no post-hoc confidence score or retrieval descriptor has, because the classifier is trained on the policy's own success/failure labels at the exact handoff states the policy will encounter.

## Method

**What it is.** We introduce **RBV** (Runtime Boundary Verifier), a learned binary classifier `V_φ(s)` that is trained jointly with a visuomotor policy `π_θ` and queried at every handoff instant. Inputs: the current observation `s` (proprioception + wrist camera features) and the policy's proposed action chunk. Output: a scalar `m ∈ [0,1]` (membership score) and a hard decision `in/out`, which either permits `π_θ` execution or returns control to the agent's plan-act-replan loop.

**How it works.**

```pseudocode
# Phase 1: boundary-labeled data collection
for each task T, each contact segment:
    sample start poses p ~ P_diverse(T)          # same distribution as Boundary-Explicit Training
    rollout π_θ from p, record (s_t, success_t)  # success = task segment completed
    label s_t with y=1 if success_t else y=0
    # boundary points: states where success flips within a small neighborhood
    # concrete estimator: for each state s_t, find its k=10 nearest neighbors in the
    # shared feature space (proprio + wrist-camera features); if the success labels
    # of these neighbors are not unanimous, mark s_t as a boundary point.
    keep (s_t, y) pairs where the k-NN success labels are mixed (i.e., not all 0 or all 1)

# Phase 2: joint training
train π_θ on imitation loss L_imit (as in Boundary-Explicit Training)
train V_φ on boundary-labeled pairs with weighted BCE:
    L_V = -[ w_pos * y * log V_φ(s) + w_neg * (1-y) * log(1 - V_φ(s)) ]
    w_pos = 2.0, w_neg = 1.0   # upweight positives: false negatives (missed out-of-boundary) cost more
    V_φ shares the visual encoder with π_θ; a 2-layer MLP head (hidden 256, GeLU activation) outputs m

# Phase 3: runtime handoff (execution-time predicate)
at each handoff instant t_handoff:
    m = V_φ(s_t)                       # membership score
    if m >= τ (τ = 0.5, calibrated on held-out boundary set):
        execute π_θ for one action chunk
    else:
        return control to agent's plan-act-replan loop   # veto the call
    # re-verify at every subsequent chunk boundary, not just initial handoff

# Phase 4: boundary expansion with verifier in the loop
retrain π_θ on curated successful system rollouts (Boundary-Expanding Self-Improvement)
re-fit V_φ on the new boundary points after each expansion round
# the verifier's decision surface tracks the expanding competence region
```

**Why this design.** We chose a *jointly trained* boundary classifier over a post-hoc threshold on the policy's action-chunk variance or an ensemble disagreement score because the classifier sees the same contact-segment distribution that defines the policy's competence region, accepting the cost that it requires labeled success/failure rollouts (which Boundary-Explicit Training already produces). We chose to share the visual encoder between `V_φ` and `π_θ` rather than train a separate perception stack because the boundary is defined in the same feature space the policy acts in, accepting the cost that verifier errors and policy errors become correlated. We chose a hard threshold `τ = 0.5` calibrated on a held-out boundary set over a soft gating weight because the handoff protocol needs a binary decision the agent can act on, accepting the cost that near-boundary states are decided by a single scalar rather than a graded fallback. We chose to re-verify at every chunk boundary rather than only at the initial handoff because the policy can drift out of its competence region mid-execution, accepting the cost of one classifier forward pass per chunk. A domain expert would not describe RBV as a variant of DPPO or of Boundary-Aware Policy Learning: DPPO fine-tunes the policy with policy gradients but never introduces a runtime membership predicate, and Boundary-Aware Policy Learning shapes and expands the boundary but communicates it statically — RBV adds the execution-time closure neither provides.

**Why it measures what we claim.** `V_φ(s)` measures boundary membership because the classifier is trained on success/failure labels drawn from the same contact-segment distribution that defines the policy's competence region; this assumption fails when the deployment state distribution shifts (e.g., novel object poses outside `P_diverse`), in which case `V_φ` reflects similarity to training boundary states rather than true competence. The threshold `τ` measures the agent's safe handoff condition because it is calibrated on a held-out boundary set so that the false-negative rate (out-of-boundary states admitted) is below a target `ε = 0.05`; this assumption fails when the held-out set under-represents a failure mode, in which case `τ` reflects the calibration set's composition rather than deployment risk. The re-verification at chunk boundaries measures mid-execution drift because the policy's state distribution is Markovian given the observation, so each chunk boundary is an independent draw from the same competence region; this assumption fails under partial observability (as in the AAWR setting), in which case the verifier sees an observation that under-determines the true state and membership is estimated from an incomplete sufficient statistic.

## Contribution

(1) RBV, a runtime boundary-membership verifier trained jointly with a visuomotor policy that converts a static handoff declaration into an execution-time predicate, closing the agent's action set under out-of-boundary invocation. (2) A design principle: sharing the visual encoder between the boundary classifier and the policy calibrates the verifier's decision surface to the policy's actual competence region, and re-verifying at chunk boundaries catches mid-execution drift that initial-handoff checks miss. (3) A boundary-labeled data protocol that reuses the contact-segment rollouts already collected by Boundary-Explicit Training, requiring no additional human labeling.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | RoboTwin 2.0 contact segments | Matches boundary-shaping distribution of prior work |
| Primary metric | Task success rate under agent handoff | Directly measures safe handoff, not raw policy skill |
| Baseline | Boundary-Aware Policy Learning (static boundary) | Tests whether runtime closure beats static boundary communication |
| Baseline | DPPO fine-tuned diffusion policy | Tests whether RL fine-tuning alone removes need for verifier |
| Baseline | Ensemble-disagreement gating | Tests whether joint boundary supervision beats post-hoc uncertainty |
| Baseline | Post-hoc success classifier with separate encoder | Tests whether joint training (shared encoder) is necessary vs. a predicate trained on the same labels but with a separate encoder |
| Ablation-of-ours | RBV without re-verification at chunk boundaries | Isolates mid-execution drift detection contribution |
| Ablation-of-ours | RBV with separate encoder (no sharing) | Isolates the effect of sharing the visual encoder between V_φ and π_θ |

### Why this setup validates the claim

The central claim is that a jointly trained runtime membership predicate, queried at every chunk boundary, prevents the agent from invoking the policy outside its competence region and thereby raises end-to-end task success. The metric must therefore be *system-level* success under the agent's plan-act-replan loop, not isolated policy success: if we measured only policy rollouts we would never observe the handoff failures RBV is designed to catch. Boundary-Aware Policy Learning tests the sub-claim that a *static* boundary description is insufficient, because it shapes and expands the boundary but lets the agent call the policy without an execution-time check; any gain over it must come from the runtime predicate. DPPO tests the sub-claim that better policy weights alone do not solve handoff failures, since a stronger policy with a wide-but-unverified competence region can still be invoked out of distribution. Ensemble-disagreement gating tests whether the *supervision signal* matters — disagreement is post-hoc and unsupervised on success, so if RBV's boundary-labeled training is doing real work, RBV should beat it where competence boundaries are sharp but action variance is not. The post-hoc success classifier with a separate encoder tests whether the *joint training* (shared encoder) is the source of any gain: if RBV matches this baseline, the benefit comes from having a predicate at all, not from joint training; if RBV outperforms, the shared encoder is doing real work. The no-re-verification ablation isolates the mid-execution-drift sub-claim: it should match RBV on episodes that fail at the initial handoff and lose ground only on episodes where the policy drifts after a valid start. The separate-encoder ablation isolates the effect of sharing the visual encoder: if RBV with a separate encoder performs similarly, the shared encoder is not necessary; if it degrades, the shared features are beneficial. Together these six comparisons decompose the claim into separately falsifiable pieces.

### Expected outcome and causal chain

**vs. Boundary-Aware Policy Learning (static boundary)** — On a case where the agent invokes the policy from a start pose just outside the shaped handoff boundary (a pose the boundary-shaping distribution under-covered), the baseline executes the policy and produces a failed or degenerate contact trajectory because its boundary is communicated once and never re-checked against the actual state. Our method instead vetoes the call at the handoff instant because `V_φ` scores that state below `τ`, returning control to the replan loop, so we expect a visible success-rate gap concentrated on out-of-boundary start poses with near-parity on in-boundary poses.

**vs. DPPO fine-tuned diffusion policy** — On a case where the policy has been fine-tuned to high average reward but its competence region remains narrow around the fine-tuning distribution, the baseline is invoked by the agent from an unfamiliar pose and produces a confident but wrong action chunk because policy-gradient improvement does not create a runtime signal about *where* the policy is reliable. Our method admits the same policy only where `V_φ` certifies membership, so we expect DPPO to show higher isolated-rollout reward yet lower system-level success, with the gap widening as the agent's invocation distribution drifts further from `P_diverse`.

**vs. Ensemble-disagreement gating** — On a case where the policy is near a competence boundary but its action-chunk variance happens to be low (a confidently wrong region), the baseline admits the call because ensemble disagreement is a proxy for uncertainty rather than for success, and it produces a failed segment. Our method instead rejects the call because `V_φ` was trained directly on success/failure labels at boundary states, so we expect RBV to outperform disagreement gating specifically on the low-variance-but-out-of-boundary subset while the two are comparable on high-variance states.

**vs. Post-hoc success classifier with separate encoder** — On a case where the policy's visual features are optimized for action prediction, a separate encoder may capture boundary-relevant features that the shared encoder misses, or vice versa. If RBV's gain comes from joint training (shared encoder), we expect RBV to outperform this baseline on out-of-boundary start poses; if the gain is merely from having a predicate, the two should be comparable. This baseline isolates the contribution of joint training from the mere existence of a runtime predicate.

**vs. RBV without chunk-boundary re-verification (ablation)** — On a case where the policy starts inside the boundary but drifts out mid-execution after several chunks, the ablation checks membership only once and continues executing into a failure because it has no mechanism to detect the drift. Our method re-queries `V_φ` at each chunk boundary and vetoes once membership drops, so we expect the ablation to match RBV on episodes that fail at the initial handoff and to lose specifically on longer episodes where drift occurs, producing a gap that grows with episode horizon.

**vs. RBV with separate encoder (ablation)** — On a case where the shared encoder's features are co-optimized for action prediction, the verifier may inherit the policy's blind spots, leading to confidently wrong membership scores. If the shared encoder is beneficial, RBV should outperform this ablation; if it is harmful, the ablation should win. This ablation isolates the effect of encoder sharing on verifier accuracy.

### What would falsify this idea

If RBV's gain over static Boundary-Aware Policy Learning is uniform across in-boundary and out-of-boundary start poses rather than concentrated on out-of-boundary poses, the runtime predicate is not doing the work we claim and the improvement comes from something else (e.g., extra shared-encoder training). Likewise, if the no-re-verification ablation matches RBV even on long-horizon episodes, the mid-execution-drift mechanism is not real and the chunk-boundary re-verification is unnecessary. If RBV matches the post-hoc success classifier with a separate encoder, the joint training (shared encoder) is not the source of the gain, and the novelty claim is weakened.

## References

1. Boundary-Aware Policy Learning for Agent-Orchestrated Robot Manipulation
2. MAESTRO: Orchestrating Robotics Modules with Coding Agents for Zero-Shot Generalist Robots

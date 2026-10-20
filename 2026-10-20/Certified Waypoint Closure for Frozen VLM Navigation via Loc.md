# Certified Waypoint Closure for Frozen VLM Navigation via Local Occupancy and Learned Executability

## Motivation

Zero-shot continuous VLN systems such as LightZeroNav, Fast-SmartWay, and Open-Nav delegate action emission entirely to a frozen VLM that outputs continuous waypoints, while the components that historically guaranteed spatial validity (trained waypoint predictors in MossVLN, occupancy-aware losses) and executability (RL locomotion policies) have been deliberately removed to preserve the zero-shot regime. This leaves a structural hole: a continuous waypoint sampled from a language model's action-token distribution carries no formal invariant that it lies in free space or that the low-level controller can actually track it, so a single geometrically-infeasible emission silently corrupts the trajectory and the frozen model has no mechanism to recover except an external planner or policy — exactly the components the field removed. The meta-level problem is that action-set closure under spatial validity and executability is asserted nowhere in the frozen-emitter paradigm, and neither LightZeroNav nor Fast-SmartWay provides a checkable substitute.

## Key Insight

A frozen VLM's continuous waypoint emission can be made a closed action set without retraining if and only if every candidate waypoint is certified against two independently-checkable invariants — membership in the locally reconstructed free-space occupancy field and membership in the forward-executability image of the low-level controller — and infeasibility is resolved by backtracking within the emitter's own discrete action-token prefix, so the certificate is a property of the token sequence rather than of an external planner.

## Method

**What it is.** We introduce CWC (Certified Waypoint Closure), a wrapper around a frozen VLM waypoint emitter. Inputs: the frozen model's per-step action-token logits over its waypoint vocabulary, the current RGB-D frame, and the recent executed trajectory. Outputs: either a certified continuous waypoint or a re-decoded waypoint obtained by truncating the action-token prefix to a previously certified prefix and re-sampling. CWC never updates VLM weights and never invokes a separate planner or RL locomotion policy.

**Load-bearing assumption (stated explicitly).** CWC assumes that the frozen emitter's action-token prefix is a sufficient statistic for the instruction-conditioned sub-goal, so that truncating to a previously certified prefix and re-decoding under hard masking yields a waypoint that still serves the instruction rather than merely a locally feasible one. This assumption is checkable per re-decode via the intent-preservation test in step 18 below; when it fails, CWC falls back to stop-and-reobserve rather than emitting a feasible-but-off-instruction waypoint.

**How it works.**

```pseudocode
# Per step t. Frozen emitter M, depth frame D_t, executed poses P_{t-1..t-k}
# Occupancy field O_t: voxel grid at 5cm resolution from D_t via TSDF fusion
# Executability operator E_phi: 3-layer MLP (hidden 128, ReLU) trained offline on (waypoint, achieved displacement) pairs
# Intent-preservation score I(w, instruction): frozen emitter's own log-likelihood of the original instruction given the re-decoded waypoint tokens, normalized by the argmax waypoint's log-likelihood

1. logits = M(observation, instruction, history)          # frozen, no gradients
2. prefix_stack = []                                       # certified token prefixes
3. for attempt in range(MAX_BACKTRACK=4):
4.     tokens = sample_or_argmax(logits)                  # waypoint token sequence
5.     w = decode(tokens)                                 # metric waypoint (x,y,theta)
6.     # Invariant 1: spatial validity
7.     S_valid = raycast_free(O_t, w, clearance=0.25m)    # all cells along segment free
8.     # Invariant 2: forward executability
9.     e_hat = E_phi(w, local_height_map(O_t), P_{t-1..t-k})
10.    E_valid = (e_hat.success_prob >= tau_exec=0.7)
11.    if S_valid and E_valid:
12.        prefix_stack.append(tokens); emit(w); break
13.    else:
14.        # backtrack in action-token space, not to a planner
15.        if prefix_stack: tokens = truncate_to(prefix_stack[-1], keep=len-1)
16.        logits = M.redecode(observation, tokens)       # constrained re-decode
17.        mask_infeasible_tokens(logits, O_t, E_phi)     # zero logits of failing continuations
18.        # Intent-preservation test (operationalizes the load-bearing assumption)
19.        w_new = decode(argmax(logits))
20.        I_score = I(w_new, instruction)                # frozen emitter log-likelihood ratio
21.        if I_score < tau_intent=0.8:
22.            emit(stop_and_reobserve); break            # assumption fails, safe fallback
23. if not certified: emit(stop_and_reobserve)          # safe fallback, no planner
```

The occupancy field `O_t` is a truncated-signed-distance voxel grid fused from the current depth frame and the last `k=3` poses; `raycast_free` checks the swept segment from the agent to `w` at 0.25 m clearance. The executability operator `E_phi` is a 3-layer MLP (hidden 128, ReLU) trained offline on 40k (waypoint, achieved displacement) pairs from a simulator, predicting the probability that the low-level differential-drive controller tracks `w` within 0.2 m and 15 degrees. `mask_infeasible_tokens` evaluates each candidate token continuation's decoded waypoint against both invariants and sets its logit to `-inf` if either fails, so the frozen decoder is constrained without weight updates. The intent-preservation score `I(w, instruction)` is the frozen emitter's log-likelihood of the original instruction given the re-decoded waypoint tokens, divided by the log-likelihood of the original argmax waypoint tokens; it is computed with a single forward pass of `M` and requires no additional training. The threshold `tau_intent=0.8` was set on a held-out set of 512 val-seen episodes to maximize the fraction of re-decodes that preserve the instruction's sub-goal while keeping the stop-and-reobserve rate below 10%.

**Why this design.** We chose a token-space backtracking mechanism over re-invoking a separate waypoint predictor (as in MossVLN) because the frozen emitter's own prefix distribution already encodes instruction-conditioned intent, and truncating it preserves that intent while a separate predictor would reintroduce the removed component and break zero-shot status; the cost is that backtracking can only recover waypoints reachable within the emitter's own token lattice, so a genuinely unreachable instruction yields a stop-and-reobserve rather than a creative detour. We chose an offline-trained executability MLP over an RL locomotion policy because a forward predictor can be certified per-waypoint without closing a control loop, whereas an RL policy would need online rollouts that a frozen zero-shot system cannot afford; the cost is that `E_phi` is only as accurate as its offline training distribution. We chose hard masking of infeasible token continuations over soft logit penalties because hard masking yields a closed action set (every emitted token sequence is certified by construction), while soft penalties leave a nonzero probability of infeasible emission; the cost is that aggressive masking can prune the argmax and force lower-probability but feasible waypoints, slightly reducing instruction fidelity. The intent-preservation test (step 18) makes the load-bearing assumption checkable per re-decode: if the re-decoded waypoint's instruction log-likelihood ratio falls below `tau_intent`, CWC does not emit it and instead falls back to stop-and-reobserve, so the certificate is a property of (token sequence AND instruction alignment) rather than of the token prefix alone.

**Why this measures what we claim.** The quantity `raycast_free(O_t, w, 0.25m)` measures spatial validity because the TSDF-fused voxel grid is assumed to be a faithful local metric reconstruction of the scene within the depth sensor's range; this assumption fails when depth is specular or beyond-range, in which case the raycast reflects sensor dropout rather than true free space. The quantity `e_hat.success_prob` from `E_phi` measures forward executability because the offline training pairs are assumed to be drawn from the same controller and dynamics distribution as deployment; this assumption fails under sim-to-real dynamics shift, in which case `E_phi` reflects in-distribution tracking success rather than the deployed controller's actual capability. The quantity `truncate_to(prefix_stack[-1], keep=len-1)` measures intent-preserving recovery because the frozen emitter's prefix is assumed to be a sufficient statistic for the instruction-conditioned sub-goal; this assumption fails when the instruction requires a disjunctive plan whose first branch is infeasible, in which case backtracking reflects local repair rather than global re-planning. The quantity `I(w_new, instruction)` measures instruction-subgoal alignment because the frozen emitter's log-likelihood of the instruction given the waypoint tokens is assumed to reflect how well the waypoint serves the instruction; this assumption fails when the emitter's likelihood is dominated by surface-form fluency rather than semantic alignment, in which case `I` reflects token-level plausibility rather than sub-goal fidelity.

## Contribution

(1) A certified waypoint closure mechanism that turns a frozen VLM's continuous waypoint emission into a closed action set under two independently-checkable invariants — local occupancy-field membership and learned forward-executability — without retraining the VLM or invoking a separate planner or RL policy. (2) A token-space backtracking procedure that resolves infeasibility by truncating and constrained-re-decoding the emitter's own action-token prefix, preserving instruction-conditioned intent while guaranteeing that every emitted waypoint is certified. (3) An offline-trained forward-executability operator that supplies a per-waypoint executability certificate, replacing the removed RL locomotion policy with a checkable predictor rather than a control loop.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VLN-CE (R2R val-unseen) | Continuous, depth-available, standard zero-shot testbed |
| Primary metric | Success Rate (SR) | Captures end-to-end goal completion, not just waypoint error |
| Baseline | LightZeroNav | Lightweight VLM zero-shot; tests our gain over frozen emitters |
| Baseline | Fast-SmartWay | Direct action emission; tests value of per-waypoint certification |
| Baseline | MossVLN | Waypoint-predictor pipeline; tests token-space backtracking vs. replanning |
| Baseline | Constrained decoding (spatial-only) | Masks only on `raycast_free`; isolates spatial invariant |
| Baseline | Constrained decoding (executability-only) | Masks only on `E_phi`; isolates executability invariant |
| Baseline | Soft-penalty variant | Adds logit penalty instead of hard mask; tests closed action set |
| Ablation-of-ours | CWC w/o hard masking | Isolates contribution of closed-action-set certification |
| Ablation-of-ours | CWC w/o intent-preservation test | Isolates contribution of the load-bearing-assumption check |

### Why this setup validates the claim

The central claim is that wrapping a frozen VLM waypoint emitter with a per-waypoint certificate (spatial validity + forward executability) plus token-space backtracking reduces infeasible emissions and thus raises end-to-end success, without any weight updates or external planner. VLN-CE val-unseen is the right testbed because it forces generalization to unseen scenes and supplies depth for the TSDF certificate, so any gain cannot be attributed to scene memorization. Success Rate is the correct primary metric because the mechanism's predicted effect is on completed episodes: a certificate that prunes infeasible waypoints should convert would-be collisions and stalls into feasible detours, which shows up as SR, whereas waypoint L2 error could improve without changing success. LightZeroNav tests the sub-claim that certification helps a lightweight frozen emitter specifically; Fast-SmartWay tests the sub-claim that per-waypoint certification beats direct uncertainty-aware emission; MossVLN tests the sub-claim that token-space backtracking matches a separate predictor without breaking zero-shot status. The spatial-only and executability-only constrained-decoding baselines test whether the conjunction of the two invariants is necessary: if either alone recovers most of the gain, the certificate is not doing the claimed joint work. The soft-penalty variant tests whether hard masking (closed action set) is required over soft penalties. The ablation removing hard masking tests whether the closed action set, rather than backtracking alone, drives the effect. The ablation removing the intent-preservation test tests whether the load-bearing assumption (prefix-as-sufficient-statistic) holds in practice: if removing the test does not change SR, the assumption is empirically supported on this benchmark; if it drops SR, the test is doing causal work.

### Expected outcome and causal chain

**vs. LightZeroNav** — On a case where the instruction says "turn left past the doorway" but the lightweight emitter's argmax waypoint clips the doorframe, LightZeroNav produces a collision-and-stall trajectory because it emits the raw decoded waypoint with no occupancy check. Our method instead raycasts the swept segment at 0.25 m clearance, rejects the waypoint, and re-decodes from a certified prefix, so we expect a noticeable SR gap concentrated on collisions-adjacent-to-openings cases but near-parity on open-corridor instructions where the raw emitter is already feasible.

**vs. Fast-SmartWay** — On a case where the panorama-free observation yields a waypoint that is spatially free but geometrically untrackable (sharp heading change the differential-drive controller cannot follow), Fast-SmartWay emits it directly because its uncertainty reasoning scores instruction confidence, not forward executability. Our method instead queries E_phi, sees success_prob below 0.7, masks that token continuation, and emits a lower-probability but trackable waypoint, so we expect a gap on tight-turn and near-obstacle subsets with parity on straight-line segments.

**vs. MossVLN** — On a case where the first proposed sub-goal is infeasible and the instruction has a viable alternate branch, MossVLN's separate waypoint predictor re-invokes a learned replanner whose output distribution no longer matches the frozen emitter's intent, sometimes drifting from the instruction. Our method truncates the frozen emitter's own prefix and re-decodes under masking, preserving instruction-conditioned intent, so we expect comparable SR on single-branch instructions but a measurable advantage on disjunctive or recovery-heavy instructions where replanning loses instruction fidelity.

**vs. Constrained decoding (spatial-only)** — On a case where the waypoint is spatially free but untrackable, the spatial-only baseline emits it because it has no executability check. Our method rejects it via E_phi, so we expect a gap on tight-turn subsets and near-parity on open-space subsets, confirming the executability invariant is necessary.

**vs. Constrained decoding (executability-only)** — On a case where the waypoint is trackable but clips an obstacle, the executability-only baseline emits it because it has no occupancy check. Our method rejects it via raycast_free, so we expect a gap on collision-adjacent subsets and near-parity on open-space subsets, confirming the spatial invariant is necessary.

**vs. Soft-penalty variant** — On a case where the argmax waypoint is infeasible but has high probability, the soft-penalty variant may still sample it because the penalty does not zero its probability. Our method hard-masks it, so we expect a gap on high-confidence-infeasible cases, confirming the closed action set is necessary.

**vs. CWC w/o intent-preservation test** — On a case where the re-decoded waypoint is feasible but off-instruction, the no-test variant emits it because it only checks spatial and executability invariants. Our method rejects it via the intent-preservation score, so we expect a gap on disjunctive or recovery-heavy instructions, confirming the load-bearing assumption check is doing causal work.

### What would falsify this idea

If our SR gain is uniform across subsets rather than concentrated on collision-adjacent and tight-turn instructions where the predicted failure modes occur, the certificate is not doing the causal work claimed. If the no-hard-masking ablation matches full CWC, the closed-action-set mechanism is not the driver and the central claim is wrong. If the no-intent-preservation ablation matches full CWC, the load-bearing assumption is empirically supported on this benchmark and the intent-preservation test is unnecessary.

## References

1. LightZeroNav: Zero-Shot Vision Language Navigation in Continuous Environments Based on Lightweight VLMs
2. MossVLN: Memory-Observation Synergistic System for Continuous Vision-Language Navigation
3. Fast-SmartWay: Panoramic-Free End-to-End Zero-Shot Vision-and-Language Navigation
4. MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation
5. Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs
6. NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation
7. LangNav: Language as a Perceptual Representation for Navigation
8. NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models

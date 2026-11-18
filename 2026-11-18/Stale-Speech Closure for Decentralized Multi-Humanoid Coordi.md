# Stale-Speech Closure for Decentralized Multi-Humanoid Coordination

## Motivation

Heave-Ho's speech-based leader-follower protocol is presented as sufficient for decentralized synchronization, but the abstract reports no analysis of message loss, latency, or adversarial speech, and training is performed with two identical humanoids under assumptions not detailed in the abstract. Structurally, a decentralized execution protocol that treats a lossy, delayed speech channel as reliable has no closure rule: once a follower misses or misorders a leader directive, the physically coupled tracking policy has no invariant guaranteeing that local observations alone can reconcile the team back into a feasible coupling configuration. This is a meta-level gap for decentralized multi-robot coordination, not only a Heave-Ho implementation detail: any coordination layer whose synchronization channel is not modeled as lossy lacks a stated condition under which local-observation execution remains physically coupled.

## Key Insight

Physical coupling between humanoids supplies a task-intrinsic, observation-observable coupling residual (relative pose and contact-force agreement) that is invariant to speech loss, so a bounded-staleness barrier driven by that residual rather than by message acknowledgment yields a closure property the leader-follower speech protocol alone cannot provide.

## Method

**What it is.** We introduce SSC (Stale-Speech Closure), a communication-failure-aware coordination layer that sits between Heave-Ho's vision-language planner and its multi-agent post-trained tracking policy. Inputs: the planner's issued leader directives with timestamps, each robot's local observation stream (proprioception, local relative-pose estimate to coupled partners, contact/force sensing), and per-link speech delivery metadata (sequence number, send time, receive time, drop flag). Outputs: a per-robot admissible directive set and a closure flag that gates whether the low-level tracking policy may execute a new directive or must hold the last consensus-feasible coupling configuration.

**Load-bearing assumption (stated explicitly).** The closure rule rests on the assumption that the coupling residual r_i(t), computed purely from each robot's local relative-pose and contact-force estimates against the last accepted directive, is a faithful and locally observable feasibility certificate for physical coupling, such that r_i(t) <= eps implies the post-trained tracking policy's coupling preconditions actually hold. This assumption is load-bearing: if r_i(t) can be small while both robots are consistently wrong (shared drift, or both tracking the same stale directive), or if r_i(t) is noise-dominated during intermittent contact (hand-offs, regrasping), then the CLOSED flag does not certify coupling feasibility. We therefore (i) define r_i(t) against a physically grounded invariant rather than only against the last accepted directive's prediction, (ii) add an explicit contact-state gate that suspends CLOSED during intermittent-contact phases, and (iii) report a calibration curve of r_i(t) versus true coupling violation on HH-Bench (see experiment) to verify the assumption empirically rather than assert it.

**How it works.**

```pseudocode
# Per robot i, per control tick t. All quantities local; no inter-robot network.
# Hyperparameters inline.

STATE: last_accepted_directive d_i, last_accept_time tau_i,
       coupling_residual_history H_i (window W=10 ticks),
       staleness_budget B=3 ticks, residual_tolerance eps=0.02 m / 0.5 N,
       contact_state c_i in {STABLE, INTERMITTENT}

on receive speech packet p from leader j at time t:
    age = t - p.send_time
    if p.seq <= last_seen_seq[j]: drop as duplicate
    if age > B * dt: mark p as STALE; do not execute; log drop
    else: candidate_directive = p.directive

# Externally anchored coupling-residual computation (local, no communication)
# r_i(t) = closed-chain kinematic residual + force-balance residual on the grasped object
#   closed-chain term: || pose_estimate_i(t) - pose_estimate_i_from_last_coupling(d_i) ||
#   force-balance term: || contact_force_i(t) - expected_force(d_i) ||
# where pose_estimate_i_from_last_coupling(d_i) and expected_force(d_i) are
# obtained by rolling the post-trained tracking policy forward one tick from
# the last accepted directive d_i and reading its predicted relative pose and
# commanded contact force (i.e., the policy's own dynamics model, not a
# free-floating prediction).
r_i(t) = || pose_estimate_i(t) - pose_estimate_i_from_last_coupling(d_i) ||
       + lambda_f * || contact_force_i(t) - expected_force(d_i) ||   # lambda_f=1.0

# Contact-state gate: suspend CLOSED where the certificate is undefined
c_i = INTERMITTENT if contact_force_i(t) < F_contact_min (F_contact_min=2.0 N)
      or |d contact_force_i / dt| > Fdot_max (Fdot_max=50 N/s)
      else STABLE

# Bounded-staleness synchronization barrier
if candidate_directive exists and age <= B*dt:
    if c_i == STABLE and r_i(t) <= eps and max(H_i) <= eps:
        accept candidate_directive; tau_i = t; reset H_i
    else:
        # residual says physical coupling is already violated, or contact is
        # intermittent so the certificate is undefined: do NOT blame speech;
        # enter reconciliation
        enter RECONCILE
elif no fresh directive for > B*dt:
    enter RECONCILE

RECONCILE:
    # local-observation state reconciliation: search over the finite set of
    # last K=5 accepted directives for the one whose predicted coupling
    # residual best matches current local observation
    d* = argmin_{d in last_K_accepted} || r_i(t; d) ||
    if c_i == STABLE and || r_i(t; d*) || <= eps:
        set d_i = d*; closure_flag_i = CLOSED
    else:
        closure_flag_i = HOLD   # freeze tracking policy at last feasible config
    broadcast (via speech, best-effort) closure_flag_i and d*

# Low-level tracking policy gating
if closure_flag_i == CLOSED: execute d_i through post-trained tracking policy
if closure_flag_i == HOLD:   hold last feasible coupling configuration
```

**Why this design.** We chose a coupling-residual-driven barrier over an acknowledgment/timeout barrier because acknowledgment only certifies delivery, not physical feasibility, accepting the cost that residual estimation must be locally observable (true here because humanoids are physically coupled and can sense relative pose and contact). We chose a finite last-K directive reconciliation search over a learned recovery policy because the admissible directive set is small and enumerable at execution time, accepting the cost that recovery cannot synthesize directives never previously accepted; this keeps the closure rule auditable and avoids a second learned controller. We chose a hard staleness budget B over an adaptive timeout because B is a stated protocol hyperparameter whose violation is exactly the failure mode we claim to close, accepting the cost that a legitimately slow but valid directive is dropped and must be re-issued. We chose to gate the tracking policy with a HOLD flag rather than let it track stale directives because the post-trained policy assumes physically coupled interaction, accepting the cost of temporary task-progress stalls during reconciliation. We anchor r_i(t) to the tracking policy's own one-step predicted pose and commanded force (its dynamics model) rather than to a free-floating prediction, and we add a contact-state gate that suspends CLOSED during intermittent contact, because the residual is otherwise a self-referential consistency check that can be small when both robots are consistently wrong and is undefined during hand-offs; this keeps the certificate externally grounded where it is defined and explicitly silent where it is not. A domain expert would not describe SSC as a variant of Heave-Ho's speech leader-follower protocol or of standard reliable-messaging/consensus layers, because neither computes a task-intrinsic coupling residual from local physical observations as the closure certificate; Heave-Ho assumes the speech channel is sufficient and standard consensus layers certify message agreement, not physical coupling feasibility.

**Why it measures what we claim.** The coupling residual r_i(t) measures physical-coupling feasibility because the assumption that relative pose and contact-force agreement against the tracking policy's one-step predicted pose and commanded force are necessary and locally observable conditions for a physically coupled multi-humanoid configuration makes r_i(t) equivalent to a feasibility certificate; this assumption fails when contact is intermittent (e.g., during a hand-off), in which case r_i(t) reflects transient force noise rather than coupling violation, which is why the contact-state gate suspends CLOSED there. The staleness budget B measures speech-channel loss tolerance because the assumption that directives older than B ticks are no longer valid for the current physical configuration makes age a proxy for directive relevance; this assumption fails when the environment is quasi-static and an old directive remains valid, in which case the budget rejects a still-correct directive. The reconciliation argmin over last-K accepted directives measures local-observation recoverability because the assumption that the feasible directive set at any tick is contained in the recently accepted set makes the finite search exhaustive; this assumption fails after a novel planner directive that was never accepted, in which case reconciliation returns the closest prior directive rather than the correct one. The closure_flag measures protocol closure because the assumption that CLOSED implies the tracking policy's coupling preconditions hold makes the flag a sufficient execution gate; this assumption fails when residual estimation is biased by sensor drift, in which case the flag reflects estimated rather than true coupling, and we report a calibration curve of r_i(t) versus true coupling violation on HH-Bench to bound this gap empirically.

## Contribution

['SSC, a communication-failure-aware coordination layer that defines a closure property for decentralized multi-humanoid execution via a bounded-staleness synchronization barrier driven by a locally observable physical-coupling residual rather than by message acknowledgment.', 'A local-observation state-reconciliation procedure over the finite set of recently accepted directives that restores a feasible coupling configuration without inter-robot network communication, together with a HOLD gate that prevents the post-trained tracking policy from executing infeasible stale directives.', 'A design principle: closure under lossy speech is achievable when the closure certificate is task-intrinsic and physically observable, not when it is derived from the communication channel itself.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HH-Bench (2–10 humanoids, incl. heterogeneous) | Matches claimed coupled, long-horizon, decentralized regime |
| Primary metric | Task success rate under injected speech-packet loss | Directly probes the failure mode SSC claims to close |
| Secondary metric | Calibration curve: r_i(t) vs. true coupling violation (AUC, false-CLOSED rate at eps) | Verifies load-bearing residual assumption empirically |
| Baseline | Heave-Ho (speech leader-follower, no closure layer) | Tests sub-claim: naive speech protocol breaks under loss |
| Baseline | Ack/timeout reliable-messaging barrier over Heave-Ho | Tests sub-claim: delivery certification ≠ physical feasibility |
| Baseline | Adaptive-timeout staleness variant | Tests sub-claim: fixed budget beats adaptive timeout |
| Baseline | Impedance/force-consensus barrier over Heave-Ho | Tests sub-claim: SSC is not a relabeling of existing residual gating |
| Ablation-of-ours | SSC without RECONCILE (HOLD-only, no last-K search) | Isolates contribution of local-observation reconciliation |
| Ablation-of-ours | SSC without contact-state gate (CLOSED asserted during intermittent contact) | Isolates contribution of the gate to the residual certificate |

### Why this setup validates the claim

The central claim is that gating execution on a locally-computed coupling residual, rather than on message acknowledgment or timeout, is what preserves coordinated execution when speech packets are lost or stale. HH-Bench is the right substrate because it is the only cited benchmark whose tasks require physically coupled whole-body interaction among 2–10 humanoids, so coupling residual is actually observable and task-intrinsic; on looser multi-agent benchmarks the residual would be undefined. Injecting packet loss and staleness is the correct perturbation because it is exactly the condition under which the baseline's assumption (speech channel sufficient) fails; without that perturbation every method would look identical. The primary metric — task success rate under loss — is the right detector because the predicted effect is a coordination breakdown, not a tracking-quality difference, so success/failure is where the causal signature should appear. The calibration curve is the right check on the load-bearing assumption because it directly measures whether r_i(t) <= eps implies true coupling feasibility (false-CLOSED rate) rather than merely asserting it. The ack/timeout baseline isolates the sub-claim that delivery agreement is insufficient, the adaptive-timeout baseline isolates the sub-claim that a hard budget is the correct closure rule, the impedance/force-consensus baseline isolates the sub-claim that SSC's residual certificate is not a relabeling of existing physical-feasibility gating, and the no-RECONCILE and no-contact-gate ablations isolate the sub-claims that finite last-K reconciliation recovers feasible directives and that the contact-state gate is necessary where the certificate is undefined. Together these form a falsifiable test: if SSC's gain were driven by something other than residual-gated closure, the ablation or the ack baseline would match it.

### Expected outcome and causal chain

**vs. Heave-Ho** — On a case where the leader issues a coupling directive during a two-humanoid lift and the speech packet is dropped, the baseline produces uncoordinated force application because it assumes the speech channel delivered the directive and keeps tracking the prior plan. Our method instead detects that no fresh directive arrived within B ticks and enters RECONCILE, selecting the last accepted directive whose predicted coupling residual matches local pose/force sensing, so we expect a clear success-rate gap concentrated on loss-injected tasks and near-parity on loss-free tasks.

**vs. Ack/timeout reliable-messaging barrier** — On a case where the acknowledgment arrives but the physical configuration has already drifted (one robot slipped), the baseline produces execution of a directive that is delivered-but-infeasible because it certifies message agreement, not coupling feasibility. Our method instead computes r_i(t) locally and enters RECONCILE when residual exceeds tolerance regardless of ack status, so we expect the ack baseline to fail specifically on drift-with-delivery instances while SSC succeeds, showing a gap on the drift subset but parity on clean-delivery subset.

**vs. Adaptive-timeout staleness variant** — On a case where a directive is legitimately slow but still valid in a quasi-static scene, the baseline produces premature rejection and re-issue churn because its timeout adapts to channel statistics rather than to the stated failure mode. Our method instead uses a fixed auditable budget B and reconciles via residual rather than re-issuing, so we expect fewer redundant directive re-issues and comparable or better success on slow-but-valid instances, with the adaptive variant paying a measurable overhead in reconciliation churn.

**vs. Impedance/force-consensus barrier** — On a case where a directive is delivered and the physical configuration has drifted, the baseline produces execution of a directive that is delivered-but-infeasible because it certifies force consensus among robots, not agreement against the tracking policy's one-step predicted pose and commanded force. Our method instead computes r_i(t) against the policy's own dynamics model and enters RECONCILE when residual exceeds tolerance, so we expect the impedance/force-consensus baseline to fail specifically on the drift-with-delivery subset where SSC succeeds, establishing that SSC's residual certificate is not a relabeling of existing physical-feasibility gating.

**vs. SSC without RECONCILE (HOLD-only)** — On a case where a stale directive is dropped but a previously accepted directive remains feasible, the baseline ablation produces a task-progress stall because HOLD freezes at the last feasible configuration with no search for a recoverable directive. Our method instead searches the last K accepted directives and re-enters CLOSED when a feasible one is found, so we expect the ablation to show longer stall durations and lower success on long-horizon tasks where recovery opportunities exist, while matching SSC on short tasks with no recovery window.

**vs. SSC without contact-state gate** — On a case where contact is intermittent (hand-off or regrasp), the baseline ablation produces a false CLOSED because r_i(t) is noise-dominated and can fall below eps while true coupling is violated. Our method instead suspends CLOSED during INTERMITTENT contact and enters RECONCILE, so we expect the ablation to show a higher false-CLOSED rate on the calibration curve and lower success on hand-off-heavy tasks, while matching SSC on tasks with stable contact throughout.

### What would falsify this idea

If SSC's success-rate gain over Heave-Ho is uniform across loss-injected and loss-free subsets rather than concentrated where packet loss and stale directives occur, the closure mechanism is not doing the claimed causal work. Likewise, if the calibration curve shows a high false-CLOSED rate at eps (r_i(t) <= eps while true coupling is violated) or if the no-RECONCILE ablation matches SSC on long-horizon tasks with recovery opportunities, the load-bearing residual assumption or the reconciliation search is not contributing and the central claim collapses.

## References

1. Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration

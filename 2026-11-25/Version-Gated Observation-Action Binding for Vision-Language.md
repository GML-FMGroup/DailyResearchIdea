# Version-Gated Observation-Action Binding for Vision-Language Computer-Use Agents

## Motivation

Across vision-language agent trees, decision policies are still permitted to act on inputs whose correspondence to the execution-time environment is assumed rather than enforced: NightWatch (arxiv:2611.28947) adds a streaming System-1 monitor that reduces empirical staleness, but leaves the binding between the policy's grounded snapshot and the monitor's current state formally unverified, so a stale action remains dispatchable; the same structural gap appears in embodied benchmarks such as EmbodiedBench (venue:icml:2025:f9c18fc261f5), where vision-driven agents are scored on discrete snapshot inputs with no correctness condition tying perception to action. The root cause is that discrete visual samples are treated as adequate proxies for continuous state, so no invariant exists that a policy can check before committing an action.

## Key Insight

A monotonically versioned interface-state token emitted by the streaming monitor turns observation-action binding into a checkable equality invariant on a shared version counter, so staleness becomes a detectable contract violation rather than an unobservable statistical tendency.

## Method

**A. What it is.** VG-Bind (Version-Gated Binding) is a wrapper protocol around any VLM computer-use policy and a streaming perception monitor. Inputs: the policy's proposed action, the version token the policy grounded on, and the monitor's current version token. Outputs: one of {DISPATCH(action), REGROUND(new snapshot, new version), ABORT(reason)}. It does not modify the policy's model, prompt, or action space; it adds a version oracle and a commit rule.

**B. How it works.**

```pseudocode
# Monitor side (streaming, per interface change event)
state v = 0                      # monotone version counter
state h = H(initial_screenshot)  # content hash of current interface state
on interface_change_event(e):
    h_new = H(patch(screenshot, e))
    if perceptual_distance(h, h_new) > tau_pix:   # tau_pix = 0.02 normalized pixel L2
        v = v + 1
        h = h_new
        broadcast Token(version=v, hash=h, t=now())

# Policy side (per decision step)
def decide(instruction, token_at_grounding):
    snap, v_g = capture_with_version()          # v_g = monitor version at capture
    action = VLM_policy(instruction, snap)       # unchanged policy
    return action, v_g

# Gate (commit rule)
def commit(action, v_g, token_now, max_reground=2):
    if token_now.version == v_g:
        return DISPATCH(action)
    elif reground_count < max_reground:
        return REGROUND(request_fresh_snapshot(), token_now)
    else:
        return ABORT(reason="state_currency_violation")

# Version oracle consistency check (monitor self-audit)
on Token(v, h):
    if h != H(current_screenshot):
        emit ALARM(monitor_desync)   # monitor is not a trustworthy oracle
```

Hyperparameters: perceptual change threshold tau_pix = 0.02 (tuned on a held-out desktop trace); maximum re-grounding attempts max_reground = 2; monitor update rate is event-driven (no fixed polling interval).

**Load-bearing assumption (stated explicitly).** The version counter increments on every perceptually significant interface change and only then, so that version equality between the policy's grounding token and the execution-time token certifies that the policy's grounded content matches the current interface state. This assumption fails when a change falls below tau_pix (e.g., a subtle text-field update, caret/selection change, incremental DOM mutation, canvas/WebGL repaint without a discrete event), in which case the equality reflects monitor insensitivity rather than true state currency. The hash self-audit cannot catch this failure because the monitor's hash and the policy's screenshot are both computed from the same stale monitor state; the audit only detects monitor-vs-screenshot desync, not monitor-vs-environment desync. We therefore treat the equality as a conditional guarantee, not an unconditional correctness invariant, and we measure the monitor's sensitivity explicitly (see Section D).

**C. Why this design.** We chose an event-driven monotone version counter over a wall-clock timestamp because timestamps order events but do not certify that the policy's grounded content matches the current content, accepting the cost that a missed change event produces a false DISPATCH rather than a detectable ordering error; we mitigate this with the hash self-audit. We chose a content-hash equality check over a learned staleness classifier because the equality is a deterministic predicate on the monitor's own output, accepting the cost that any monitor desynchronization becomes a hard ABORT rather than a soft warning; this is deliberate because the contract's value is exactly that it fails loudly. We chose bounded re-grounding (max_reground = 2) over unbounded retry because unbounded retry converts a perception failure into an infinite loop, accepting the cost that some recoverable states are aborted. A domain expert would not describe this as NightWatch-plus-a-guard: NightWatch is a passive refresher whose benefit is measured statistically, whereas VG-Bind makes the monitor a version oracle whose output is a precondition on the action, and the central object is the invariant, not the monitor. Nor is it a controller/router over two subsystems (anti-pattern 4): the gate has a closed-form decision rule (version equality) with no learned arbitration.

**D. Why it measures what we claim.** The quantity `token_now.version == v_g` measures observation-action binding because the assumption is that the monitor's version counter increments on every perceptually significant interface change and only then; this assumption fails when a change falls below tau_pix (e.g., a subtle text-field update), in which case the equality reflects monitor insensitivity rather than true state currency. The quantity `h != H(current_screenshot)` measures monitor trustworthiness because the assumption is that the content hash is computed over the same rendering pipeline the policy sees; this assumption fails under rendering nondeterminism (anti-aliasing, cursor blink), in which case the alarm reflects rendering noise rather than desynchronization. The quantity `state_currency_violation` rate measures the staleness that NightWatch only estimates empirically because the assumption is that every dispatched action was grounded on a version equal to the execution-time version; this assumption fails when the environment changes between the gate check and the actual OS-level action execution, in which case the metric reflects residual execution latency rather than perception staleness. To validate this metric against true staleness, we label a held-out set of dispatched actions with an oracle ground-truth staleness label (human-verified or replay-verified) and report the agreement between the gate's violation flag and the oracle label; we also report the monitor's true-positive and false-negative change-detection rates on a labeled change corpus, so that the certified-binding rate is a bounded guarantee conditional on measured monitor sensitivity rather than an asserted invariant.

**E. Scope.** Designed for desktop and dynamic web environments where interface changes are observable as events or screenshot diffs; assumes the monitor runs on the same machine as the policy and that the policy can be wrapped without modifying its internals.

## Contribution

['VG-Bind: a version-gated observation-action binding protocol that converts a streaming perception monitor into a version oracle and enforces a checkable state-currency invariant on every dispatched action, with a closed-form commit rule (DISPATCH / REGROUND / ABORT).', 'A monitor self-audit mechanism using content-hash equality that distinguishes monitor desynchronization from genuine state change, yielding an explicit trust signal for the oracle rather than an implicit assumption.', 'An operationalization of staleness as a measurable contract-violation rate (state_currency_violation) that is checkable per action, in contrast to aggregate stale-action counts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OSWorld + dynamic web traces | Matches live interfaces where staleness arises |
| Primary metric | Dispatched-action staleness rate | Directly measures observation-action binding failure |
| Secondary metric | Downstream task success | Tests whether gating helps end-to-end |
| Secondary metric | Catastrophic-action rate | Counts clicks into wrong surfaces or destructive confirmations |
| Baseline | Vanilla snapshot CUA policy | Tests whether gating beats no monitoring |
| Baseline | NightWatch-attached CUA | Tests whether version oracle adds over passive refresh |
| Baseline | Wall-clock timestamp gate | Tests whether content versioning beats ordering |
| Ablation-of-ours | VG-Bind without hash self-audit | Tests whether desync detection drives the gain |
| Ablation-of-ours | VG-Bind with tau_pix swept {0.005, 0.01, 0.02, 0.05, 0.1} | Tests whether result is an artifact of one threshold |

### Why this setup validates the claim

The central claim is that making the monitor a version oracle whose equality is a precondition on dispatch reduces actions grounded on stale interface state, and that this reduction is attributable to version-gated binding rather than to monitoring per se. The dataset choice matters because staleness only manifests where screens change on their own during inference latency, which is exactly the desktop and dynamic web regime; a static benchmark would make every gate trivially pass. The baseline set is a ladder testing distinct sub-claims: vanilla snapshot CUA tests whether any currency mechanism helps; NightWatch-attached CUA tests whether a passive refresher, whose benefit is statistical, is sufficient without a commit rule; wall-clock timestamp gating tests whether ordering alone certifies content match. The ablation isolates the hash self-audit, which is the component that turns silent monitor desynchronization into a loud abort, so if the gain survives without it, our stated causal mechanism is wrong. The primary metric is dispatched-action staleness rate because it is the only quantity that directly operationalizes the invariant; task success alone would confound perception staleness with policy competence, and monitor alarm counts would measure the monitor rather than the binding. To validate the primary metric against true staleness, we label a held-out set of dispatched actions with an oracle ground-truth staleness label (human-verified or replay-verified) and report agreement between the gate's violation flag and the oracle label; we also report the monitor's true-positive and false-negative change-detection rates on a labeled change corpus, so that the certified-binding rate is a bounded guarantee conditional on measured monitor sensitivity rather than an asserted invariant. We sweep tau_pix over {0.005, 0.01, 0.02, 0.05, 0.1} and report staleness rate as a function of tau_pix to show the result is not an artifact of one threshold. We also sweep VLM inference latency to establish when the contract matters.

### Expected outcome and causal chain

**vs. Vanilla snapshot CUA policy** — On a case where a modal dialog appears after the policy captures its screenshot but before its click executes, the baseline dispatches the click into the wrong surface because it has no notion that its grounded content expired. Our method instead compares the monitor version at capture against the execution-time version, sees inequality, and re-grounds on a fresh snapshot, so we expect a concentrated drop in staleness rate on the dynamic-interface subset with near-parity on static pages. We also expect a drop in catastrophic-action rate and a gain in downstream task success on the dynamic subset.

**vs. NightWatch-attached CUA** — On the same dialog case, NightWatch refreshes the agent's perception and reduces idle waiting, but because its benefit is statistical and it imposes no precondition, some fraction of actions still dispatch against the pre-dialog snapshot when the refresh lands after the decision. Our method converts the monitor output into a closed-form commit rule, so stale dispatches become regroundings or aborts, and we expect a gap over NightWatch specifically on the highest-change-rate traces rather than uniformly. We also expect the gap to widen as VLM inference latency increases, because the perception-action gap grows with latency.

**vs. Wall-clock timestamp gate** — On a case where the interface changes and then reverts to a visually identical state, a timestamp gate sees a newer time and either blocks a valid action or, if the clock granularity is coarse, dispatches a stale one, because timestamps order events without certifying content. Our method compares content-hash-derived versions, so identity of content yields equality and dispatch, and we expect fewer false aborts on revert-heavy traces plus lower staleness on coarse-clock traces.

### What would falsify this idea

If the staleness reduction is uniform across static and dynamic subsets rather than concentrated where interface change events occur, the gate is not binding on version currency and the causal story is wrong. If removing the hash self-audit leaves the gain unchanged, then the claimed desynchronization-detection mechanism is not what produces the effect. If the gate's violation flag disagrees with the oracle ground-truth staleness label on the held-out set, the primary metric does not measure true staleness.

## References

1. Don't Sleep, Keep Watch: Proactive Perception for Computer-Use Agents

# Monotone Precondition Refinement for Causal World-State Planning

## Motivation

CausGround and the broader family of structured planners (scene-graph, long-term reasoning-state, and object-centric latent abstractions) treat the factorized causal world-state as a static snapshot: preconditions distilled from interaction experience are grounded once against the current observation and then used for backward/forward reasoning, with no mechanism that keeps them sound as new observations arrive. Structurally, this is because the factorization has no belief-transition semantics — there is no operator that constrains how a precondition's truth status may change under refinement, so a later observation that contradicts a precondition can either be silently ignored (unsound execution) or arbitrarily flip it (invalidating already-committed plan prefixes). The result is that CausGround's causal knowledge, which is already incomplete for novel objects, degrades without any detectable signal of when a plan's preconditions have become untrustworthy.

## Key Insight

If every observation-triggered update is restricted to a conservative operator that can only strengthen a precondition from UNKNOWN to TRUE or retract it from TRUE to FALSE while forbidding FALSE→TRUE resurrection without an explicit re-derivation witness, then the set of TRUE preconditions is monotone non-decreasing along any refinement chain, so plan prefixes committed under earlier beliefs remain sound and every retraction is attributable to a specific observation rather than to silent drift.

## Method

**What it is.** We introduce **MonoRef** (Monotone Precondition Refinement), an online belief-refinement kernel layered on CausGround's factorized causal world-state. Its inputs are (i) CausGround's factorized state $S_t$ over object attributes and inter-object relations, (ii) its distilled precondition set $\Pi = \{\pi_a\}$ for each action $a$, and (iii) a stream of observations $o_1, o_2, \dots$; its output is a belief state $B_t = \langle \text{status}: \Pi \to \{\text{UNKNOWN}, \text{TRUE}, \text{FALSE}\}, W_t, \Phi_t \rangle$ where $W_t$ is a per-precondition witness log, and $\Phi_t$ is a per-committed-action effect witness log (see below), together with a plan-validity flag for any committed prefix.

**Load-bearing assumption (stated explicitly).** MonoRef assumes that a precondition's truth status, once committed to TRUE under a plan prefix, can be treated as sound-by-construction under the monotone lattice because the planner only ever needs the current truth value — i.e., that retracting a TRUE precondition to FALSE and then replanning is sufficient to preserve committed-prefix soundness. This assumption holds only if the already-executed actions of the prefix did not occur under a now-invalidated belief. To make the assumption checkable rather than implicit, we extend the witness log to record, per committed action, the post-condition/effect it assumed, and we gate prefix extension on effect-consistency of the executed prefix (see $\Phi_t$ and `EFFECT_CONSISTENT` below). This does not add a new planning mechanism; it makes the existing monotone restriction apply to the executed state trajectory, not only to the pending precondition set.

**How it works.**

```pseudocode
# State: status[p] in {UNKNOWN, TRUE, FALSE}; witness[p] = list of observation ids
#        effect_witness[a] = list of (observation id, assumed post-condition) for executed action a
# tau_hi = 0.85, tau_lo = 0.15  (evidence log-odds thresholds)
# kappa = 3  (min independent observations to permit a retraction)
# independence: two observations are independent iff they have distinct timestamps
#   AND distinct sensor modalities (RGB, depth, proprioception, force)

function REFINE(B, o_t):
    E = GROUND_PRECONDITIONS(o_t, Pi)      # per-precondition evidence e_t[p] in [-1,1]
    for p in Pi:
        l = LOGODDS(B.status[p]) + e_t[p]  # accumulate evidence
        if B.status[p] == UNKNOWN and l >= tau_hi:
            B.status[p] = TRUE;  B.witness[p].append(o_t)
        elif B.status[p] == TRUE and l <= tau_lo and COUNT_INDEP(B.witness[p]) >= kappa:
            B.status[p] = FALSE; B.witness[p].append(o_t)   # retraction, logged
        elif B.status[p] == FALSE and l >= tau_hi:
            # resurrection is FORBIDDEN silently: require re-derivation witness
            if HAS_REDERIVATION_WITNESS(o_t, p):
                B.status[p] = TRUE; B.witness[p].append(o_t)
            else:
                B.status[p] = FALSE   # stays retracted; flag CONFLICT
    return B

function RECORD_EFFECT(B, a, o_t):
    # after executing action a, log the post-condition it assumed and the observation
    B.effect_witness[a].append((o_t, POSTCONDITION(a)))
    return B

function EFFECT_CONSISTENT(B, prefix):
    # executed prefix is effect-consistent iff no executed action's assumed
    # post-condition is contradicted by a later observation that retracted a
    # precondition it relied on
    for a in EXECUTED(prefix):
        for (o_id, post) in B.effect_witness[a]:
            if any(B.status[p] == FALSE for p in post):
                return False
    return True

function PLAN_VALID(B, prefix):
    # a committed prefix is sound iff no pending precondition it relied on is FALSE
    # AND the executed portion is effect-consistent
    return all(B.status[p] != FALSE for p in PRECONDITIONS(prefix)) \
           and EFFECT_CONSISTENT(B, prefix)

# Planning loop (wraps CausGround backward/forward reasoning)
while not done:
    o_t = OBSERVE(); B = REFINE(B, o_t)
    if not PLAN_VALID(B, committed_prefix): REPLAN(B)   # retraction or effect inconsistency triggers replan
    else: EXTEND_PREFIX(B)                              # monotone: safe to continue
    a = NEXT_ACTION(committed_prefix); EXECUTE(a); B = RECORD_EFFECT(B, a, o_t)
```

**Why this design.** We chose a *monotone status lattice* over a full probabilistic belief update because CausGround's preconditions are symbolic and the planner consumes only truth values, so a distributional belief would add cost without changing any planner decision, accepting the consequence that we cannot express graded uncertainty — instead we expose it as an explicit UNKNOWN state and a conflict flag. We chose *evidence log-odds accumulation with hysteresis* ($\tau_{hi}=0.85$, $\tau_{lo}=0.15$) over a single-observation flip because a lone noisy grounding should not retract a precondition the planner has already committed to, accepting the cost that genuine fast-changing scenes are retracted with a delay of a few observations. We chose a *witness log with a minimum independent-observation count* $\kappa=3$ over an unconstrained retraction because it makes every FALSE status attributable to concrete observations, accepting the cost of extra bookkeeping per precondition. We chose to *forbid silent FALSE→TRUE resurrection* and require a re-derivation witness because that is the one transition that would break monotonicity of the TRUE set and silently invalidate committed prefixes, accepting the cost that a genuinely reversible precondition (e.g., a door re-closed then re-opened) must be re-grounded from scratch rather than cheaply flipped back. We extended the witness log to record per-action assumed post-conditions ($\Phi_t$) and gate prefix extension on effect-consistency because the monotone lattice alone protects the set of TRUE preconditions, not the execution of a committed prefix; this makes the load-bearing assumption checkable without adding a new planner. A domain expert would not describe this as a variant of a Kalman-filter or particle-filter belief tracker (the closest named prior family, e.g., object-centric latent filters in embodied planning): those maintain a distribution over continuous latents and have no notion of a symbolic precondition lattice or of a committed plan prefix whose soundness is preserved by construction, whereas MonoRef's contribution is precisely the *transition restriction* on the symbolic status lattice, not the estimator.

**Evidence model (specification of $e_t[p]$ and independence).** The grounding module returns, for each precondition $p$, a calibrated log-likelihood ratio $e_t[p] = \log \frac{P(o_t \mid p=\text{TRUE})}{P(o_t \mid p=\text{FALSE})}$, clipped to $[-1,1]$; calibration is performed on a held-out set of 512 grounding examples per precondition type by isotonic regression. Two observations $o_i, o_j$ are counted as independent witnesses for $p$ iff they have distinct timestamps and distinct sensor modalities (RGB, depth, proprioception, force). Under these definitions, the TRUE-set is monotone non-decreasing along any refinement chain because the only transition into TRUE from FALSE requires an explicit re-derivation witness, and the only transition out of TRUE is a retraction that is logged and attributable to $\kappa$ independent observations.

**Block D — Why it measures what we claim.** Plan-validity rate measures committed-prefix soundness because it counts the fraction of episodes in which no relied-upon precondition is FALSE and the executed prefix is effect-consistent; this assumes grounding errors are temporally independent so hysteresis filters them without masking genuine changes, and it fails when correlated grounding errors (e.g., occlusion) defeat independence and cause stale TRUE statuses, in which case plan-validity rate reflects the persistence of stale beliefs rather than true soundness. The witness-count $\kappa$ measures retraction reliability because it assumes independent observations are available; this fails when all observations come from a single modality, in which case $\kappa$ reflects redundant rather than independent evidence. Effect-consistency measures executed-prefix soundness because it assumes post-conditions are correctly logged; this fails when post-conditions are mis-specified, in which case it reflects logging errors rather than execution validity.

## Contribution

(1) MonoRef, an online belief-refinement kernel with a monotone precondition-invariant operator layered on CausGround's factorized causal world-state, comprising a status lattice, an evidence-accumulation update with hysteresis, and a witness-log retraction rule that forbids silent FALSE→TRUE resurrection. (2) A design principle — conservative refinement with explicit UNKNOWN and conflict states — that converts a static scene-graph factorization into a belief-transition system in which committed plan prefixes remain sound under observation accumulation, with retractions attributable to specific observations. (3) A reproducible planning loop (REFINE / PLAN_VALID / REPLAN) that integrates the kernel with CausGround's backward-forward causal reasoning and exposes precondition conflicts as a first-class signal for replanning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VirtualHome + RoboCas adapted splits | Matches CausGround; tests symbolic preconditions and real manipulation |
| Primary metric | Plan-validity rate under noisy grounding | Directly measures committed-prefix soundness, our central claim |
| Secondary metric | Effect-consistency rate of executed prefixes | Tests the load-bearing assumption repair |
| Baseline | CausGround (no belief refinement) | Tests whether monotone refinement over raw grounding helps |
| Baseline | Object-centric latent filter tracker | Tests symbolic lattice vs continuous distributional belief |
| Baseline | Single-observation precondition flip | Tests hysteresis and witness-count necessity |
| Ablation-of-ours | MonoRef without resurrection restriction | Isolates the transition-restriction contribution |
| Ablation-of-ours | MonoRef without effect-consistency gate | Isolates the load-bearing-assumption repair |
| Noise protocol | Per-precondition corruption rates {0.05, 0.15, 0.30}; corruptions independent across steps | Pre-registered; tests independence assumption |
| Correlated-noise protocol | Occlusion-induced correlated errors across consecutive steps | Tests failure mode F from Block D |

### Why this setup validates the claim

The central claim is narrow: restricting transitions on a symbolic precondition lattice preserves committed-prefix soundness under noisy grounding better than either no refinement or a distributional tracker. CausGround without refinement tests the sub-claim that refinement is needed at all — if raw grounding already yields high plan-validity, MonoRef's mechanism is unmotivated. The object-centric latent filter tests the sub-claim that a continuous distribution is the wrong representational choice for a planner that consumes only truth values; if it matches MonoRef, the lattice framing adds nothing. The single-observation flip baseline tests the hysteresis and witness-count claims specifically, since it differs from MonoRef only in those two mechanisms. The resurrection-restriction ablation isolates the one transition rule that most directly protects committed prefixes. The effect-consistency ablation isolates the load-bearing-assumption repair: if removing the effect-consistency gate does not degrade plan-validity, the assumption that retraction-and-replan suffices is unverified. Plan-validity rate under injected grounding noise is the right metric because it is exactly the quantity MonoRef's design targets: it counts how often the planner's committed prefix remains sound, so any gain must come from the transition restriction rather than from better perception. A uniform gain across noise levels would be suspicious; the predicted effect is concentrated where grounding noise is high and where preconditions are genuinely reversible. The correlated-noise protocol tests the independence assumption in Block D: if correlated errors defeat hysteresis, plan-validity should drop relative to the independent-noise condition.

### Expected outcome and causal chain

**vs. CausGround (no belief refinement)** — On a case where a noisy grounding spuriously reports a precondition as false mid-execution, the baseline produces a replan triggered by a single corrupted observation because it has no hysteresis or witness requirement. Our method instead keeps the precondition TRUE until evidence log-odds cross the low threshold with κ independent witnesses because of the monotone status lattice. We expect a noticeable plan-validity gap on the high-noise subset, with near-parity on the low-noise subset.

**vs. Object-centric latent filter tracker** — On a case where a precondition's truth must be committed to for several steps, the baseline produces oscillating belief values that occasionally cross a decision boundary because it maintains a continuous distribution with no symbolic commitment semantics. Our method instead holds a discrete TRUE status with a witness log because the lattice makes commitment explicit and retraction costly. We expect MonoRef to show fewer spurious replans per episode, with the gap widening as episode length grows.

**vs. Single-observation precondition flip** — On a case where a genuine fast-changing scene briefly mis-grounds a precondition, the baseline produces an immediate FALSE and an unnecessary replan because it flips on one observation. Our method instead delays retraction until κ independent observations accumulate because of hysteresis and witness counting. We expect MonoRef to trade a small latency penalty on genuinely changing subsets for a clear plan-validity gain on noisy-but-stable subsets.

**vs. MonoRef without effect-consistency gate (ablation)** — On a case where a precondition is retracted after an action that relied on it has already executed, the ablation reports the prefix as valid because it checks only pending preconditions. Our method instead flags effect-inconsistency and triggers replan because it logs per-action post-conditions. We expect the ablation to show higher plan-validity but lower effect-consistency, revealing the load-bearing-assumption gap.

**Correlated-noise condition** — On a case where occlusion causes consecutive grounding errors, the independence assumption in Block D fails and hysteresis may hold stale TRUE statuses. We expect plan-validity to drop relative to independent noise, and the effect-consistency gate to partially compensate by catching executed-prefix contradictions.

### What would falsify this idea

If MonoRef's plan-validity gain over the single-observation flip baseline were uniform across noise levels and subsets rather than concentrated on noisy-but-stable preconditions, the hysteresis-and-witness mechanism would not be the cause. Likewise, if the resurrection-restriction ablation showed no degradation in committed-prefix soundness, the transition restriction would be doing no work and the central claim would be wrong. If the effect-consistency ablation showed no gap in effect-consistency rate, the load-bearing-assumption repair would be unnecessary.

## References

1. CausGround: Causal Interaction Grounding for Adaptive Embodied Planning

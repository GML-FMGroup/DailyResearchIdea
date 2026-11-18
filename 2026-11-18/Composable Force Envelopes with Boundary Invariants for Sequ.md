# Composable Force Envelopes with Boundary Invariants for Sequenced Contact Primitives

## Motivation

Existing force-regulation systems regulate contact locally but provide no rule that guarantees global contact invariants when primitives are sequenced: ForceHarness's self-terminating primitives each return at a contact event without declaring what force envelope they guarantee across the handoff, CR-DAgger learns residual force corrections without a boundary condition between correction segments, and ForceVLA's force-aware MoE decodes actions without a composition contract between decoded segments. The structural root cause is that force regulation is treated as a per-segment property with no algebra for composing segments, so a plan that chains locally safe primitives can still produce a force spike or discontinuity at the seam where one primitive terminates and the next begins.

## Key Insight

If every primitive declares a force envelope as an interval over its own contact state plus a boundary invariant that constrains the force trace at termination, then adjacent primitives compose soundly exactly when their envelopes overlap and their shared boundary satisfies the no-spike and continuity conditions, making the seam itself a checkable object rather than an emergent risk.

## Method

**What it is.** We introduce **FEC** (Force-Envelope Composition), a contract layer that wraps each self-terminating force primitive with a declared force envelope and a boundary invariant, and a plan-level supervisor that admits a primitive sequence only if every adjacent pair passes an envelope-overlap check and a boundary-invariant check. Inputs: a candidate primitive sequence from a frozen planner, each primitive's declared contract (envelope interval, termination contact event, boundary invariant parameters). Outputs: an admitted sequence, or a rejected seam with the specific violated condition and the primitive index.

**Load-bearing assumption (stated explicitly).** FEC's admission decision is sound only if a primitive's author-declared force envelope is a true bound on its realized force while active. We do not assume this silently: each primitive's contract includes an online envelope-conformance monitor that checks realized force against the declared envelope during execution, and the supervisor conditions seam admission on the preceding primitive having stayed within its envelope. If the monitor flags an envelope violation, the seam is re-checked against the realized force trace (not the declared envelope) and rejected if the realized pre/post force difference exceeds `spike_tol` or the realized continuity check fails. This makes overlap a bound on realized, not merely declared, force sets.

**How it works.**

```pseudocode
# Each primitive p carries contract C_p = (env_p, term_p, inv_p)
#   env_p  = [f_lo, f_hi]  force magnitude interval the primitive guarantees while active
#   term_p = contact event at which p self-terminates
#   inv_p  = (cont_tol, spike_tol)  boundary invariant tolerances
#   monitor_p = online envelope-conformance check: |f(t)| in env_p for all t while p active

SUPERVISOR(sequence S = [p1..pn], measured force trace f, tau=contact sampling period):
  for i in 1..n-1:
    # (0) envelope-conformance of the preceding primitive
    if not CONFORMS(f, active_interval(p_i), env_{pi}):
        # realized force exited declared envelope: re-check seam on realized trace
        f_pre_real  = f[t_seam - tau]
        f_post_real = f[t_seam + tau]
        if |f_pre_real - f_post_real| > inv_{pi}.spike_tol:
            reject(seam=i, reason="realized_spike", delta=|f_pre_real - f_post_real|)
        if not CONTINUOUS(f, t_seam - tau, t_seam + tau, inv_{pi}.cont_tol):
            reject(seam=i, reason="realized_discontinuity")
        # else fall through to declared checks below as a secondary filter
    # (1) envelope overlap at the seam
    if not OVERLAP(env_{pi}, env_{p_{i+1}}):
        reject(seam=i, reason="envelope_disjoint", gap=interval_gap)
    # (2) boundary invariant at the handoff
    t_seam = termination_time(p_i, f)          # last sample before term_{pi}
    f_pre  = f[t_seam - tau]                    # force just before termination
    f_post = f[t_seam + tau]                    # force just after next primitive engages
    if |f_pre - f_post| > inv_{pi}.spike_tol:   # no-spike condition
        reject(seam=i, reason="spike", delta=|f_pre - f_post|)
    if not CONTINUOUS(f, t_seam - tau, t_seam + tau, inv_{pi}.cont_tol):
        reject(seam=i, reason="discontinuity")
  admit(S)

# Repair loop (only if rejected):
#   insert a bridging primitive p_b with env_{p_b} = env_{pi} ∩ env_{p_{i+1}}
#   and inv_{p_b} = (min(cont_tol_i, cont_tol_{i+1}), min(spike_tol_i, spike_tol_{i+1}))
#   re-run SUPERVISOR on the spliced sequence; cap at 2 bridge insertions per seam.

# Semi-automated contract derivation (per primitive):
#   Run K=10 calibration rollouts of the primitive on HiddenPhys.
#   env_p = [min_t f(t) - 2*sigma_f, max_t f(t) + 2*sigma_f] over all rollouts,
#           where sigma_f is the per-primitive force-sensor noise std estimated from a 1-s static hold.
#   spike_tol = 3 * sigma_f (sensor noise floor); cont_tol = 2 * sigma_f.
#   Primitives whose calibration rollouts show realized force outside the derived env_p
#   are flagged for hand-authoring; all others are auto-derived.
```

Hyperparameters live at the seam: `tau` is the contact sampling period used to estimate pre/post forces (default 10 ms, swept over {2, 5, 10, 20, 50} ms in the tau-sensitivity experiment); `spike_tol` and `cont_tol` are declared per primitive by the primitive author (or auto-derived as above), not tuned by the supervisor. The envelope-conformance monitor runs at the same `tau` and flags a violation if any sample in the active interval exits `env_p` by more than `2*sigma_f`.

**Why this design.** We chose to make the envelope and invariant *declared by the primitive author* rather than *estimated online by the supervisor*, because an estimated envelope would make the admission decision depend on the same noisy force trace it is supposed to certify, collapsing the check into a self-fulfilling test; the cost is that primitive authors must commit to conservative envelopes, which may reject some sequences that would in fact have been safe. We chose an *interval-overlap* composition rule over a learned compatibility classifier because overlap is a closed-form condition on declared quantities that a planner can inspect before execution, whereas a classifier would reintroduce the per-segment opacity that ForceHarness already has; the cost is that overlap is a sufficient but not necessary condition, so some composable pairs are rejected. We chose to insert a *bridging primitive* on rejection rather than to re-plan from scratch because re-planning discards the planner's already-validated prefix and re-opens every earlier seam; the cost is that the bridge primitive must itself be authored with a declared contract, which limits repair to the primitive library's coverage. We added the online envelope-conformance monitor because the declared-envelope assumption is load-bearing: without it, a primitive whose realized force exits its declared envelope would make the overlap check certify a contract that no longer holds, so the monitor re-checks the seam against the realized trace and preserves the composition contract under envelope violation. This is not a variant of ForceHarness: ForceHarness supplies self-terminating primitives that regulate force *within* a segment and return at a contact event, but it has no notion of a declared envelope, no seam object, and no admission rule — our contribution is the composition contract and the supervisor that checks it, not the primitive regulator itself.

**Why it measures what we claim.** The envelope-overlap check `OVERLAP(env_{pi}, env_{p_{i+1}})` measures *compositional soundness at the seam* because we assume that a primitive's declared envelope is a true bound on its realized force while active; this assumption fails when a primitive's realized force exits its declared envelope under unmodeled contact dynamics, in which case overlap reflects the declared contracts rather than the actual reachable force sets. The online envelope-conformance monitor `CONFORMS(f, active_interval(p_i), env_{pi})` measures *whether the declared-envelope assumption holds for the preceding primitive* because we assume the monitor's sampling period `tau` is fast enough to catch envelope exits; this assumption fails when contact dynamics evolve faster than `tau`, in which case the monitor reflects aliased samples rather than the true envelope violation. The no-spike check `|f_pre - f_post| <= spike_tol` measures *boundary force continuity* because we assume the force trace sampled at period `tau` around the termination event is representative of the instantaneous handoff (assumption A: contact dynamics evolve slower than `tau`, so aliasing is negligible); this assumption fails when contact dynamics evolve faster than `tau`, in which case the check reflects aliased samples rather than the true seam transient and fast transients can alias into a false pass (failure mode F). The continuity check `CONTINUOUS(f, ...)` measures *absence of a discontinuity at the handoff* because we assume the tolerance `cont_tol` is calibrated to the sensor noise floor; this assumption fails when sensor noise exceeds `cont_tol`, in which case the check reflects noise rather than a genuine discontinuity. The bridging-primitive repair measures *restorability of a rejected seam* because we assume the primitive library contains a bridge whose envelope is contained in the intersection; this assumption fails when the intersection is empty, in which case repair reflects library coverage rather than seam repairability.

## Contribution

(1) A compositional force-regulation contract in which each self-terminating primitive declares a force envelope and a boundary invariant, together with a plan-level supervisor that admits or rejects primitive sequences by checking envelope overlap and boundary-invariant satisfaction at every seam. (2) A bridging-primitive repair rule that splices a contract-bearing primitive into a rejected seam, turning a binary admit/reject into a bounded repair operation. (3) A seam-level diagnostic that reports the specific violated condition (disjoint envelope, spike, discontinuity) and its magnitude, giving primitive authors a per-seam signal for tightening declared envelopes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HiddenPhys seven-task MuJoCo benchmark | Contact-rich tasks with hidden physical variables stress seam checks |
| Primary metric | Admitted-sequence success rate on seam-stressing tasks | Directly measures whether composition contracts predict safe handoffs |
| Secondary metric | Envelope-tightness distribution (realized vs declared bounds) per primitive | Measures the load-bearing assumption rather than assuming it |
| Baseline | ForceHarness frozen-model primitives | Tests whether our contract layer adds value over bare self-terminating primitives |
| Baseline | Expert force controllers | Tests whether declared-envelope admission matches expert seam judgment |
| Baseline | Learned end-to-end force policies | Tests whether contract reasoning beats implicit learned composition |
| Ablation-of-ours | FEC without bridging repair | Isolates whether admission checks or repair drive the gain |
| Ablation-of-ours | FEC without online envelope-conformance monitor | Isolates whether realized-force re-checking drives the gain |
| Ablation-of-ours | FEC with hand-authored vs auto-derived contracts | Quantifies authoring burden and contract quality |

### Why this setup validates the claim

The central claim is that declared force envelopes plus boundary invariants let a supervisor predict, before execution, which primitive sequences will hand off safely — and that this prediction beats mechanisms that lack a seam object. HiddenPhys is the right testbed because its hidden physical variables (mass, friction, stiffness) are exactly what make realized force diverge from nominal, so seams are where failures should concentrate. The ForceHarness baseline tests the sub-claim that self-terminating regulation within a segment is insufficient without a composition contract: if our gain appears only at seams and not within segments, the contract layer is doing the work. The expert-controller baseline tests whether declared-envelope admission approximates expert seam judgment rather than merely outperforming a weak learned policy. The learned-policy baseline tests whether explicit contract reasoning beats implicit composition. The no-bridge ablation isolates whether admission checks alone or repair is responsible. The no-monitor ablation isolates whether the online envelope-conformance check is necessary to preserve the composition contract when realized force exits the declared envelope. The hand-authored vs auto-derived ablation quantifies the practical ceiling on adoption by reporting how many primitives must be hand-authored versus auto-derived from K=10 calibration rollouts. The envelope-tightness distribution directly measures the load-bearing assumption: if realized force frequently exits the declared envelope, the monitor is doing the work and the declared-envelope assumption is weak. Success rate on seam-stressing tasks is the right primary metric because it directly registers the predicted effect — failures at handoffs — rather than averaging over segments where all methods should agree. A tau-sensitivity sweep over {2, 5, 10, 20, 50} ms reports admission decisions across tau values, operationalizing assumption A and failure mode F: if admission decisions change materially with tau, aliasing is affecting the check.

### Expected outcome and causal chain

**vs. ForceHarness frozen-model primitives** — On a case where a plug-insertion primitive terminates at contact and the next primitive engages at a higher force setpoint, ForceHarness hands off with no seam check, so the frozen model's next decision arrives after an unregulated force transient and the insertion either jams or overshoots. Our method instead rejects that seam on the envelope-overlap or spike check before execution and inserts a bridge, so we expect a noticeable gap on seam-stressing tasks (plug insertion, belt assembly) but rough parity on single-segment tasks where no seam exists.

**vs. Expert force controllers** — On a case where contact stiffness is hidden and the expert controller's gains are tuned for nominal stiffness, the expert produces a slow or oscillatory handoff at the seam because it regulates force continuously without a declared handoff contract. Our method instead admits the sequence only if declared envelopes overlap and the boundary invariant holds, so we expect our success rate to approach the expert on tasks where the declared envelopes are conservative and to trail only where library coverage forces rejection of otherwise-safe seams.

**vs. Learned end-to-end force policies** — On a case where two primitives are individually well-learned but their composition was never seen in training, the learned policy produces a force spike at the handoff because it has no explicit seam representation and interpolates between segment behaviors. Our method instead checks the seam against declared quantities and either admits or bridges it, so we expect a concentrated gap on unseen primitive pairs and parity on pairs seen in training.

**vs. FEC without bridging repair** — On a case where a seam is rejected on envelope-overlap but a bridge primitive exists in the library, the no-bridge ablation rejects the sequence and falls back to the frozen planner's next candidate, which may re-open earlier seams. Full FEC inserts the bridge and re-runs the supervisor, so we expect full FEC to outperform the no-bridge ablation on seams with available bridges and to match it on seams with empty intersections (where repair is impossible).

**vs. FEC without online envelope-conformance monitor** — On a case where a primitive's realized force exits its declared envelope under hidden stiffness, the no-monitor ablation admits the seam based on declared overlap even though the realized pre/post force difference exceeds `spike_tol`, producing a spike at the handoff. Full FEC's monitor flags the envelope violation and re-checks the seam on the realized trace, rejecting it, so we expect full FEC to outperform the no-monitor ablation specifically on primitives with low envelope tightness (realized force near or outside declared bounds).

### What would falsify this idea

If our gain over ForceHarness is uniform across seam-stressing and single-segment tasks rather than concentrated at seams, the contract layer is not doing the claimed work and the central claim is wrong. Likewise, if the no-bridge ablation matches full FEC everywhere, admission checks alone are not the mechanism and repair is irrelevant; if the no-monitor ablation matches full FEC everywhere, the declared-envelope assumption holds on HiddenPhys and the online monitor is unnecessary.

## References

1. ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models

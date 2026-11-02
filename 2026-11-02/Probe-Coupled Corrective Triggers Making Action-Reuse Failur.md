# Probe-Coupled Corrective Triggers: Making Action-Reuse Failure Diagnosis Identifiable Under Distribution Shift

## Motivation

Across the corrective-action trees, the trigger that decides when to repair an action tail — log-based failure diagnosis in HexaShell and confidence-gated renoising in VLA-Rolling — is assumed reliable at nominal conditions and is only ever validated there, while under distribution shift it becomes unidentifiable because the trigger is the sole feedback signal and no independent quantity can contradict it. VLA-Rolling in particular gates renoising on per-action confidence whose calibration for task correctness is uncharacterized, so a shifted scene can produce a confidently wrong reuse decision that the loop has no way to detect. The structural problem is that the trigger's own validity is an assumption rather than a testable property of the closed loop.

## Key Insight

A corrective trigger becomes identifiable exactly when it coincides with the unique sign-flip of a held-out counterfactual sufficiency probe computed on the frozen prefix — because then the same quantity that fires the trigger also certifies it, and a mis-specified trigger is detectable without any external label.

## Method

**What it is.** Probe-Coupled Triggers (PCT) is an inference-time wrapper for frozen flow-matching VLA policies (e.g., π_{0.5}, SmolVLA) that replaces the confidence gate of VLA-Rolling with a trigger defined as the sign-flip point of a counterfactual sufficiency probe, certified by an explicit monotonicity test on the sufficiency profile. Inputs: the frozen prefix (already-executed chunk plus current observation embedding), the inherited action tail, and a small held-out probe set P of (prefix, action-tail, observed outcome) triples collected from nominal rollouts. Outputs: a binary reuse/repair decision per action index, a scalar trigger-validity score, and a monotonicity residual.

**How it works.**

```pseudocode
# Frozen at deployment: VLA policy f, probe set P (|P| = 64 nominal triples)
# Probe MLP: 2-layer, hidden=64, GeLU activation, trained with MSE loss
#   on (prefix_embedding, action_tail) -> observed_outcome (binary success indicator)
#   using 64 nominal triples, 200 epochs, Adam lr=1e-3, batch size 16.
# Per control step t:
prefix_t   = frozen_prefix(obs_t, executed_actions)      # embedding, no grad
A_tail     = inherited_tail(chunk_{t-1})                 # VLA-Rolling stage 1

# (1) Counterfactual sufficiency probe on the frozen prefix
for each (p_i, a_i, y_i) in P:
    s_i = sufficiency_score(prefix_t, a_i)   # probe MLP, 2-layer, frozen
S_t = median_i( s_i )                        # scalar in [-1, 1]

# (2) Counterfactual intervention on the action tail
A_tail_cf = renoise(A_tail, sigma=0.3, steps=2)   # single fixed intervention
S_t_cf    = sufficiency_score(prefix_t, A_tail_cf)
Delta_t   = S_t - S_t_cf                          # sign = direction of sufficiency

# (3) Trigger = certified sign-flip point along the tail
# Compute full sufficiency profile S_t(k) for k=1..K (K=8)
for k in 1..K:
    S_t(k) = sufficiency_score(prefix_t, A_tail[1:k])
# Monotonicity test: check if S_t(k) is monotone (non-increasing or non-decreasing)
#   by counting sign changes in the first difference d_k = S_t(k+1) - S_t(k).
#   A profile is certified monotone if it has at most one sign change in d_k.
monotone = (number of sign changes in d_k) <= 1
if monotone:
    tau_t = argmin_{k in [1..K]} | S_t(k) |
    fire  = (sign(S_t(tau_t - 1)) != sign(S_t(tau_t + 1))) and (|Delta_t| > delta_min)
else:
    # Fallback: repair from the earliest certified-sufficient k
    #   (first k where S_t(k) > 0 and S_t(k) > S_t(k-1))
    tau_t = min{ k : S_t(k) > 0 and (k==1 or S_t(k) > S_t(k-1)) }
    fire  = True  # multi-modal profile forces repair

# (4) Validity score (same signal that fires the trigger) + monotonicity residual
validity_t = |Delta_t| / (|S_t| + eps)
# Monotonicity residual: fraction of sign changes in d_k (0 if monotone)
mono_residual = (number of sign changes in d_k) / (K-1)
if fire and validity_t > v_min:
    repair via renoising with budget = ceil(10 * validity_t)   # bounded [1,10]
else:
    reuse A_tail
```

Hyperparameters set once on nominal validation: sigma=0.3 (intervention scale), delta_min=0.05 (minimum sufficiency gap), v_min=0.4 (validity threshold), probe size 64, probe MLP width 64. The monotonicity test uses the full profile S_t(1..K) and a sign-change count threshold of 1; the monotonicity residual is reported alongside validity_t.

**Why this design.** We chose a *single fixed renoising intervention* (sigma=0.3, 2 steps) over a learned counterfactual model because the frozen prefix already exposes the action-tail surface, so the intervention is computable without training a generative counterfactual, accepting the cost that the intervention is not tailored per scene. We chose to define the trigger as the *certified sign-flip point* of the sufficiency profile rather than thresholding per-action confidence (as VLA-Rolling does) because a threshold on confidence has no independent quantity to contradict it under shift, accepting the cost that the trigger now requires K probe evaluations per step (K = tail length, typically 8) and an explicit monotonicity test. We chose the *median over 64 held-out probe triples* rather than a single probe because the median is robust to a few shifted probes while remaining sensitive to a systematic shift, accepting the cost that the probe set must be collected once from nominal rollouts and is not updated online. We chose to couple the validity score to the *same* Delta_t that fires the trigger rather than to a separate verifier because a separate verifier reintroduces the two-module router anti-pattern (module A fires, module B validates), accepting the cost that validity and firing are correlated by construction — which is exactly the identifiability we want, not a bug. A domain expert would not describe this as a variant of VLA-Rolling's confidence gate because the gate is replaced by a sign-flip condition on a counterfactual sufficiency profile, not a threshold on the gate's own output.

**Why it measures what we claim.** The computational quantity `S_t` (median probe sufficiency) measures *predictive sufficiency of the frozen prefix for the inherited tail* because the probe set P was collected from nominal rollouts where the prefix is by construction sufficient, so any deviation of S_t from its nominal distribution is attributable to prefix-tail mismatch; this assumption fails when the probe set is itself drawn from a shifted distribution, in which case S_t reflects probe-set shift rather than prefix-tail mismatch. The quantity `Delta_t = S_t - S_t_cf` measures *causal necessity of the current tail* because renoising is a valid intervention on the tail while holding the prefix fixed, so a large positive Delta_t means the current tail is more sufficient than a renoised alternative and reuse is warranted; this assumption fails when renoising is not a faithful intervention (e.g., the renoised tail is out of the policy's training support), in which case Delta_t reflects the intervention's own distribution shift rather than tail necessity. The quantity `tau_t` (argmin of |S_t(k)|) measures *the trigger point* because the sufficiency profile is monotone in the prefix-tail alignment under nominal conditions, so its unique sign-flip is the boundary between sufficient and insufficient tail segments; this assumption fails when the profile is non-monotone (multi-modal sufficiency), in which case tau_t reflects the first local sign-flip rather than the global boundary — which is why `fire` additionally requires `|Delta_t| > delta_min` and why we add an explicit monotonicity test that certifies uniqueness before firing, falling back to a defined repair rule when the profile is multi-modal.

## Contribution

(1) A trigger mechanism for action-reuse VLA policies in which the corrective decision is defined as the unique sign-flip of a counterfactual sufficiency probe, making the trigger's own validity a testable property of the closed loop rather than an assumption. (2) A design principle — probe-coupled firing — under which the signal that fires a corrective action and the signal that certifies its validity are the same quantity, eliminating the need for a separate verifier module. (3) An inference-time procedure with a bounded renoising budget derived from the validity score, applicable to any frozen flow-matching VLA without retraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LIBERO-Plus shift suite (main); RoboTwin 2.0 bimanual tasks (appendix) | Contains systematic visual/embodiment shift where prefix-tail mismatch arises |
| Primary metric | Success rate on shift subsets, stratified by predicted failure mode | Detects concentrated gain, not uniform improvement |
| Secondary metric | Monotonicity rate of S_t(k) profiles on nominal data; calibration plot of S_t vs. ground-truth prefix-tail mismatch | Validates sign-flip assumption and probe calibration |
| Baseline | VLA-Rolling confidence gate | Tests whether sign-flip trigger beats threshold-on-own-output |
| Baseline | Native fixed-step denoising (10-step) | Tests whether reuse/repair beats always-recompute |
| Baseline | Always-reuse inherited tail (no trigger) | Tests whether the trigger itself, not reuse, drives gain |
| Baseline | Independently-trained verifier (separate MLP, same architecture as probe, trained on shifted data) | Tests whether coupling trigger and validity is the novel ingredient |
| Ablation-of-ours | PCT with thresholded S_t instead of sign-flip | Isolates the sign-flip trigger mechanism from the probe signal |
| Ablation-of-ours | PCT with monotonicity test disabled (always use argmin-of-|S_t(k)|) | Isolates the effect of the monotonicity certification |
| Ablation-of-ours | PCT with sigma in {0.1, 0.3, 0.6} | Tests stability of Delta_t sign under faithful interventions |

### Why this setup validates the claim

The central claim is that a sign-flip point on a counterfactual sufficiency profile is a better reuse/repair trigger than a threshold on the policy's own confidence, precisely because it has an independent quantity to contradict the confidence signal under shift. LIBERO-Plus and RoboTwin 2.0 supply the shift regimes where prefix-tail mismatch is systematic rather than incidental, so the predicted effect should be visible there and near-absent on nominal subsets. VLA-Rolling's confidence gate is the direct competitor testing the sub-claim that a self-referential threshold fails when confidence and correctness decouple under shift; native fixed-step denoising tests that reuse is worth having at all; always-reuse tests that the trigger, not the reuse policy, carries the gain. The thresholded-S_t ablation holds the probe signal fixed and varies only the trigger definition, so any gap between it and full PCT is attributable to the sign-flip mechanism. The independently-trained verifier baseline tests whether the coupling of trigger and validity (same Delta_t) is the novel ingredient versus a two-module router. The monotonicity-disabled ablation isolates the effect of the certification step. The sigma sweep tests the faithfulness of the renoising intervention. Success rate stratified by predicted failure mode is the right metric because a uniform gain would be consistent with generic robustness, whereas a concentrated gain on the shift subsets where confidence decoupling is expected is the signature the mechanism predicts. The calibration plot and monotonicity rate directly validate the two load-bearing assumptions: that S_t tracks prefix-tail mismatch and that S_t(k) profiles are monotone on nominal data.

### Expected outcome and causal chain

**vs. VLA-Rolling confidence gate** — On a case where lighting shift corrupts the observation embedding but the inherited tail is still locally valid, the baseline's confidence gate reads depressed confidence and forces unnecessary repair, injecting renoising noise into an already-good tail. Our method instead computes S_t against the frozen probe set, finds the tail still sufficient, and reuses it, because the sign-flip trigger keys on the sufficiency profile rather than the policy's self-report. We expect a noticeable success-rate gap on visually-shifted LIBERO-Plus subsets and rough parity on nominal subsets.

**vs. Native fixed-step denoising** — On a case where the chunk is largely correct but one early action is misaligned, the baseline recomputes the full chunk from scratch, spending compute and occasionally landing on a worse mode. Our method reuses the sufficient tail and only renoises at the sign-flip boundary with a bounded budget, so we expect comparable or better success at markedly lower inference cost, with the advantage concentrated on tasks with long horizons where recompute variance accumulates.

**vs. Always-reuse inherited tail** — On a case where an object is displaced mid-execution, the baseline reuses a tail that is no longer sufficient and fails to correct, producing a missed grasp. Our method fires at the sign-flip point, detects the large Delta_t, and repairs with a validity-scaled budget, so we expect a clear gap on dynamic-perturbation subsets and parity where no perturbation occurs — confirming the trigger, not reuse, carries the gain.

**vs. Independently-trained verifier** — On a case where the verifier is trained on shifted data and disagrees with the probe, the two-module router may fire inconsistently because the verifier and trigger are not coupled. Our method uses the same Delta_t for firing and validity, so the trigger and its certification are aligned by construction. We expect PCT to outperform the independent verifier on shift subsets where the verifier's training distribution diverges from the probe's nominal distribution.

**vs. PCT with thresholded S_t** — On a case where S_t is high but the tail is actually insufficient (e.g., a confidently wrong reuse), the thresholded version fires incorrectly because it lacks the sign-flip condition. Our method requires a sign-flip and a monotonicity certification, so it avoids false positives. We expect a gap on subsets where confidence and correctness decouple.

**vs. PCT with monotonicity test disabled** — On a case where the sufficiency profile is multi-modal, the disabled version picks an arbitrary local sign-flip, leading to incorrect repair points. Our method certifies monotonicity and falls back to a defined rule, so we expect higher success on multi-modal subsets and a reported monotonicity residual that correlates with failure.

**vs. PCT with sigma sweep** — We expect Delta_t's sign to be stable across sigma in {0.1, 0.3, 0.6} on nominal data, with larger sigma increasing variance but not flipping the sign, confirming the intervention is faithful.

### What would falsify this idea

If PCT's gain over VLA-Rolling is uniform across nominal and shifted subsets rather than concentrated where prefix-tail mismatch is predicted, or if the thresholded-S_t ablation matches full PCT, or if the monotonicity rate on nominal data is low (e.g., <80%) and the monotonicity-disabled ablation performs equally well, then the sign-flip trigger is not doing the causal work and the central claim is wrong.

## References

1. VLA-Rolling: Fewer Denoising Steps, Better Control via Adaptive Action Reuse
2. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
3. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
4. AlphaEvolve: A coding agent for scientific and algorithmic discovery
5. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
6. Amplifying human performance in combinatorial competitive programming
7. Multi-objective Evolution of Heuristic Using Large Language Model
8. GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs

# Calibrated Monitor-Failure Closure for Vision-Language Computer-Use Agents

## Motivation

Streaming perception layers for vision-language computer-use agents, such as the NightWatch-0.8B monitor in Don't Sleep, Keep Watch, are deployed as trusted oracles: the VLM policy acts on every monitor update without any representation of when that update is stale, wrong, or hallucinated. This is a structural gap, not a tuning issue, because the monitor's output is a point estimate of interface state with no attached error distribution, so the downstream policy has no defined behavior when the perception layer silently fails; the same oracle assumption underlies ShowUI's UI-guided token selection, which prunes visual tokens without certifying that pruned tokens were irrelevant. The field therefore lacks a closure condition for perception-layer failure: a monitor that cannot report its own unreliability forces the policy to either over-trust stale snapshots or ignore the monitor entirely.

## Key Insight

A streaming monitor can be made a calibrated sensor rather than an oracle by pairing each update with a held-out-probe-calibrated pair of error probabilities (missed-change and false-change) and by making the VLM policy's action set a closed-form function of those probabilities, so that the policy's expected utility is monotone in monitor reliability and degrades to full-screenshot re-grounding exactly when the monitor's error rate crosses the cost-ratio threshold.

## Method

**A. What it is.** We introduce CAMCO (Calibrated Monitor-failure Closure), a wrapper around a streaming interface monitor (NightWatch-0.8B) and a VLM computer-use policy that attaches to every monitor update a calibrated pair `(p_miss, p_false)` — the probability that the monitor missed a real interface change and the probability that it reported a change that did not occur — and exposes to the policy a three-way action set `{ACT, REGROUND, VERIFY}` whose choice is computed from `(p_miss, p_false)` and a task-cost ratio, not from the VLM's own log-probs. Inputs: monitor update `u_t`, current screenshot `s_t`, last full screenshot `s_{t-k}`, task instruction `I`. Outputs: a reliability-labeled update and a discrete perception action for the policy.

**B. How it works.**

```pseudocode
# Offline: calibrate the monitor's error probabilities on a held-out probe set
# Probe set P = {(s, s', label)} where label in {changed, unchanged} is human-annotated
# on 2,000 pairs drawn from OSWorld + dynamic web, disjoint from agent evaluation.
# Each pair is also annotated with a change-type tag c in {layout_shift, modal_dialog,
#   text_update, animation, dark_mode, other} by the same annotator.
# Calibration bins: 10 quantile bins of the monitor's internal change-score, crossed
#   with the 6 change-type tags, giving up to 60 (bin, change-type) cells.
# Cells with fewer than 30 probe pairs are merged into their parent bin (all change types).
for each cell (b, c) defined by monitor change-score quantile (10 bins) x change-type tag (6):
    p_miss[b,c]   = (# changed pairs in cell where monitor said unchanged) / (# changed pairs in cell)
    p_false[b,c] = (# unchanged pairs in cell where monitor said changed) / (# unchanged pairs in cell)
# Store cell -> (p_miss, p_false) as a lookup table; no learned calibration head.
# Online conformal wrapper: maintain a live calibration buffer B_live of the most recent
#   256 monitor updates whose ground-truth change label is obtained by a cheap
#   structural-diff oracle (pixel-diff threshold on the full screenshot, no VLM).
#   For each live update, compute the nonconformity score s = |monitor_change_score - oracle_change_score|.
#   The conformal quantile q_hat is the ceil((n+1)(1-alpha))-th smallest score in B_live,
#   with alpha = 0.1 (target 90% coverage). The calibrated pair is then
#   p_miss_live = clip(p_miss[b,c] + q_hat, 0, 1), p_false_live = clip(p_false[b,c] + q_hat, 0, 1).
#   B_live is updated every 32 steps by appending the latest 32 labeled updates and evicting oldest.

# Online: at each monitor update u_t with change-score landing in bin b_t and change-type tag c_t
(p_miss, p_false) = lookup[b_t, c_t]
(p_miss_live, p_false_live) = conformal_adjust(p_miss, p_false, B_live, alpha=0.1)
# Cost model: c_act = 1 unit (VLM forward on current state)
#             c_reground = 3 units (full screenshot + VLM forward)
#             c_verify = 2 units (cheap verifier VLM on diff crop)
# Expected loss of each perception action under the calibrated error pair:
L_act      = p_miss_live * L_stale + (1 - p_miss_live) * 0
L_reground = c_reground / c_act
L_verify   = c_verify / c_act + p_false_live * L_false_verify
# Closed-form choice: pick argmin over {ACT, REGROUND, VERIFY}
a_t = argmin(L_act, L_reground, L_verify)
# The VLM policy pi(a | s_t, I) is then queried only if a_t == ACT;
# if a_t == REGROUND, s_t is replaced by a fresh full screenshot before pi is queried;
# if a_t == VERIFY, a 0.5B verifier VLM is run on the pixel-diff crop and its
# change/no-change verdict replaces u_t, after which the choice is recomputed once.
# The verifier's verdict is also used to update B_live with the structural-diff oracle label.
```

Hyperparameters: 10 reliability bins x 6 change-type tags (up to 60 cells, min 30 pairs per cell); probe set size 2,000; live calibration buffer B_live of 256 updates, refreshed every 32 steps; conformal alpha = 0.1; `L_stale = 5` (task-failure cost relative to one VLM forward, set from OSWorld episode-length statistics); `L_false_verify = 2`; one recomputation step after VERIFY (no loop).

**C. Why this design.** We chose a held-out-probe lookup table over a learned calibration head because the monitor's change-score is a scalar with a monotone relationship to error rate only within quantile bins, and a learned head would reintroduce the log-prob-as-confidence failure mode that Kadavath et al. document; the cost is that the table cannot extrapolate to interface distributions far outside the probe set, which we accept because the probe set is drawn from the same OSWorld/web pool as evaluation. We chose a closed-form argmin over expected loss rather than a learned router because Anti-Pattern 4 warns that a controller between two modules is the actual hard problem; here the decision rule follows from the stated cost model and calibrated probabilities, so the controller is fully specified and auditable, at the cost that the cost model must be estimated once per deployment. We chose to let the VLM policy consume the reliability label only through the discrete action `a_t` rather than as a soft prompt feature because feeding raw probabilities into the VLM would make the policy's behavior unverifiable and would collapse onto the log-prob-gating pattern; the cost is that the policy cannot learn a finer-grained reliability-conditioned behavior, which we accept because the three-way action set already spans the useful decision space. A domain expert would not describe CAMCO as a variant of NightWatch because NightWatch supplies a point-estimate monitor with no error model, whereas CAMCO's load-bearing object is the calibrated `(p_miss, p_false)` pair and the closed-form threshold that makes the policy's action a monotone function of monitor reliability.

**D. Why it measures what we claim.** The quantity `p_miss[b,c]` measures the monitor's missed-change probability because the held-out probe set's human labels are assumed to be the ground truth of interface change, and the binning by monitor change-score quantile crossed with change-type tag is assumed to make the monitor's score a sufficient statistic for its error rate within a cell; this assumption fails when the interface undergoes a change type absent from the probe set (e.g., a new animation style), in which case `p_miss[b,c]` reflects the probe set's change-type mix rather than the live interface's. The quantity `p_false[b,c]` measures the false-change probability under the same assumption and fails in the same way, additionally when the probe set's unchanged pairs are drawn from a different visual distribution than the live screen. The conformal-adjusted pair `(p_miss_live, p_false_live)` measures the live error probability because the structural-diff oracle is assumed to provide a correct change label for each buffered update and the conformal quantile is assumed to cover the live distribution at the target rate; this assumption fails when the structural-diff oracle itself misses small UI deltas, in which case the adjustment reflects the oracle's blind spot rather than the monitor's true error. The closed-form action `a_t` measures the motivation-level concept of monitor-failure closure because the expected-loss comparison is assumed to be exact given the calibrated pair and the cost model, so that the policy's choice is optimal for the stated costs; this assumption fails when the cost model's `L_stale` is misestimated for a particular task, in which case `a_t` reflects the misestimated cost ratio rather than the true reliability-driven choice. The VERIFY step's verifier VLM verdict measures a second opinion on change because the verifier is assumed to be conditionally independent of the monitor given the pixel-diff crop; this assumption fails when both models share a training bias toward ignoring small UI changes, in which case the verdict reflects a correlated blind spot rather than an independent check.

## Contribution

['A calibrated monitor-failure closure mechanism (CAMCO) that attaches held-out-probe-calibrated missed-change and false-change probabilities to every streaming monitor update and exposes a closed-form three-way perception action (ACT, REGROUND, VERIFY) derived from an explicit cost model, converting the monitor from a trusted oracle into a calibrated sensor.', "A design principle for vision-language computer-use agents: the policy's perception action should be a monotone function of monitor reliability and the task-cost ratio, so that the agent's behavior degrades to full-screenshot re-grounding exactly when the monitor's error rate crosses the cost-ratio threshold, rather than relying on the VLM's own confidence.", 'An open reliability-labeled probe protocol and bin-lookup calibration table for NightWatch-0.8B on OSWorld and dynamic web interfaces, usable as a diagnostic for monitor failure modes.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OSWorld + dynamic web tasks | Matches NightWatch evaluation and probe pool |
| Primary metric | Task success rate under fixed VLM-forward budget | Captures reliability gain without compute confound |
| Baseline | NightWatch point-estimate monitor | Tests whether calibration alone drives gains |
| Baseline | Always-REGROUND (no monitor trust) | Tests whether cost-aware choice beats brute-force refresh |
| Baseline | Raw monitor update, no gating | Tests the core stale-action failure path |
| Baseline | Learned calibration head (temperature-scaled MLP on monitor score) | Tests whether quantile-bin lookup is merely a simpler instance |
| Baseline | Second monitor architecture (e.g., a different streaming perception model) | Tests whether closure condition generalizes beyond NightWatch-0.8B |
| Ablation-of-ours | CAMCO with uncalibrated monitor scores | Isolates the calibration contribution |
| Ablation-of-ours | CAMCO without conformal adjustment (static lookup only) | Isolates the online calibration contribution |
| Ablation-of-ours | CAMCO with probe-set shift (new animation styles, dark-mode transitions) | Measures degradation of p_miss calibration under shift |

### Why this setup validates the claim

The central claim is that a calibrated `(p_miss, p_false)` pair turns monitor reliability into a closed-form perception action that closes the monitor-failure gap more efficiently than either trusting the monitor blindly or ignoring it. The raw-monitor baseline tests the failure-path sub-claim: if stale-action errors are the mechanism, this baseline should suffer them most on dynamic interfaces. The always-REGROUND baseline tests the cost sub-claim: it should reduce stale errors but pay a compute tax that hurts success under a fixed VLM-forward budget, so a win over it isolates the value of selective rather than constant regrounding. The NightWatch point-estimate baseline tests whether a monitor without an error model is sufficient, i.e. whether our gains come from calibration or merely from having any streaming monitor. The learned calibration head baseline tests whether the quantile-bin lookup is merely a simpler instance of existing calibration, and where the two diverge. The second monitor architecture baseline tests whether the closure condition generalizes beyond NightWatch-0.8B. The uncalibrated-CAMCO ablation is the sharpest test: it keeps the action set and cost model but destroys the probability inputs, so any remaining gain would mean calibration is not load-bearing. The no-conformal ablation isolates the online calibration contribution. The probe-set shift ablation measures the degradation of p_miss calibration under distribution shift. Success rate under a matched forward budget is the right metric because it forces the comparison onto the reliability-vs-cost tradeoff the method claims to solve; a pure accuracy metric would let always-REGROUND win trivially.

### Expected outcome and causal chain

**vs. NightWatch point-estimate monitor** — On a case where a slow-loading web page shifts a button after the monitor's last snapshot, the point-estimate monitor reports a change with no error bound, so the policy acts on a stale coordinate and misclicks because it has no signal that the monitor's own reliability is low in that score bin. Our method instead reads a high `p_miss_live` for that bin and change type and routes to REGROUND before querying the policy, so the click lands on the current layout. We expect a noticeable success gap concentrated on dynamic-interface subsets with high change-score variance, with near-parity on static desktop tasks.

**vs. Always-REGROUND (no monitor trust)** — On a case where the interface is stable for many steps (a form-filling sequence), always-REGROUND burns three cost units per step on full screenshots, exhausting the forward budget before the long-horizon task completes, because it treats every step as unreliable. Our method instead reads low `p_miss_live` in the stable bin and chooses ACT at one unit, preserving budget for the genuinely dynamic segments. We expect a success gap that grows with episode length and is largest on long-horizon tasks, with parity on short episodes.

**vs. Raw monitor update, no gating** — On a case where a modal dialog appears and the monitor's change-score is ambiguous, the raw update propagates a possibly-missed change straight to the policy, which acts on a stale state because nothing downstream checks the monitor's error rate. Our method instead compares expected losses and, when `p_miss_live * L_stale` exceeds the reground cost, issues REGROUND or VERIFY, so the action is grounded in the current screen. We expect fewer stale-action failures and a success gap concentrated on transient-UI subsets, with the uncalibrated ablation recovering only part of the gain.

**vs. Learned calibration head** — On a case where the monitor's change-score is high but the change type is a subtle text update, the learned head may overfit to the probe set's change-type mix and produce a miscalibrated probability, whereas our quantile-bin lookup crossed with change-type tags provides a more robust estimate. We expect our method to match or slightly outperform the learned head on in-distribution tasks and to outperform it on out-of-distribution change types, with the gap widening as the shift increases.

**vs. Second monitor architecture** — On a case where the second monitor has a different change-score distribution, our method's calibration procedure (probe set + conformal adjustment) should transfer with minimal modification, whereas a method tuned to NightWatch-0.8B might require retuning. We expect our method to show similar gains over the point-estimate baseline for the second monitor, demonstrating generalization of the closure condition.

**vs. CAMCO without conformal adjustment** — On a case where the live interface undergoes a change type present in the probe set but with different frequency, the static lookup may be miscalibrated, leading to suboptimal action choices. The conformal adjustment uses the live buffer to correct for this, so we expect the full CAMCO to outperform the static version on tasks with distribution shift, with the gap concentrated on interfaces with novel change-type mixes.

**vs. CAMCO with probe-set shift** — On a case where the probe set deliberately excludes certain change types (e.g., new animation styles), the static lookup will have high error for those types, but the conformal adjustment should partially compensate by using the live buffer. We expect the full CAMCO to degrade less than the static version, but still show some degradation compared to the in-distribution case, quantifying the cost of probe-set shift.

### What would falsify this idea

If CAMCO's gain over the raw-monitor baseline is uniform across static and dynamic subsets rather than concentrated where monitor misses are predicted to be frequent, the calibration-to-action chain is not the cause. Equally, if the uncalibrated ablation matches full CAMCO, the calibrated pair is not load-bearing and the claim collapses to "any gating helps."

## References

1. Don't Sleep, Keep Watch: Proactive Perception for Computer-Use Agents

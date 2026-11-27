# Uncertainty-Gated Audit Escalation for Multimodal Safety Under OOD Transformations

## Motivation

Multimodal safety alignment methods such as JudiAct (Guarding with Judgment) rely on an audit kernel whose review-item discrimination is assumed to be a sufficient statistic for input harmfulness, but this assumption is never stress-tested under out-of-distribution transformations. JOOD (Playing the Fool) shows that simple mixup-based OOD-ification raises model uncertainty and bypasses safety alignment, so the audit kernel's confident but wrong labels on OOD-ified harmful inputs represent a structural robustness gap: the audit's discrimination capacity is not invariant to the input distribution shift that adversarial OOD-ification induces, and no existing method gates intervention decisions on whether that capacity has collapsed.

## Key Insight

The audit's discrimination capacity is not a fixed property but a function of the input's OOD-displacement from the audit corpus, and this displacement is measurable via the agreement rate among single-item-swapped review audits—so an uncertainty-gated escalation rule can detect capacity collapse before a confident but wrong label is emitted.

## Method

```pseudocode
# Inputs: multimodal input x = (q, v), JudiAct audit kernel A (4B backbone),
#         audit corpus D_audit, guardrail G (multi-level guard model, e.g., Beaver-Guard-V style)
# Output: handling action a in {allow, help, warn, block} and escalation flag e

1. Compute base audit: r = A(x)  # structured review items + diagnostic label + action
2. Compute OOD-displacement score:
   - For each single-item swap i in {1,2,3}:
       r_i = A(swap_item(r, i, D_audit))  # swap one review item with nearest corpus item
   - Agreement rate: s = (1/3) * sum_i [label(r_i) == label(r)]
   - Displacement: d = 1 - s  (d in [0,1], higher = more OOD)
3. Calibrate d via held-out OOD probe set:
   - Fit threshold tau on D_probe (mixup-transformed harmful inputs from JOOD protocol, N=512)
   - tau = argmax_t (F1_block(t) - F1_block_baseline)  # maximize block-action F1 on probe
4. Uncertainty-gated escalation:
   if d > tau:
       e = 1
       a = G(x)  # guardrail decision replaces audit action
   else:
       e = 0
       a = action(r)
5. Return (a, e, d, r)
```

**Why this design.** We chose single-item-swap agreement (d) over the model's raw log-probability because log-prob calibration for task correctness is a known open problem (Kadavath et al., Tian et al.), and JudiAct's own training already produces single-item-swapped reviews as a refinement stage—so the swap-agreement signal is structurally aligned with the audit kernel's training distribution rather than an external confidence proxy, accepting the cost that d requires three additional forward passes per input. We chose a held-out OOD probe threshold (tau) over a fixed threshold because the displacement-to-capacity mapping is model- and corpus-specific, accepting the cost that tau must be re-fit when the audit corpus changes. We chose to replace the audit action with a guardrail decision (G) rather than abstain entirely because a guardrail preserves a usable safety response, accepting the cost that the guardrail's own error modes now dominate the escalated path. A domain expert would NOT describe this as a variant of JOOD (which attacks) or of standard confidence gating, because the gate signal is the audit's own internal consistency under item swaps—a quantity JOOD does not compute and that log-prob gating does not use.

**Why this measures what we claim.** The swap-agreement rate d measures OOD-displacement from the audit corpus because we assume that when an input lies within the audit's training distribution, single-item swaps produce label-consistent reviews (the audit's cross-case label comparison training enforces this); this assumption fails when the input is adversarially constructed to make all review items individually plausible but jointly inconsistent, in which case d reflects adversarial inconsistency rather than distributional displacement. The threshold tau measures the boundary of the audit's discrimination capacity because we assume the OOD probe set (JOOD-style mixup transformations, N=512) spans the displacement range where capacity collapses; this assumption fails when the probe's transformation family is narrower than the test-time OOD family, in which case tau is mis-calibrated. The escalation flag e measures whether a confident-but-wrong label risk exists because we assume d > tau implies the audit's label is unreliable; this assumption fails when the audit is confidently correct despite high d (e.g., trivially safe inputs with high swap variance), in which case e over-triggers the guardrail. The guardrail decision G measures safe handling under capacity collapse because we assume G's own OOD robustness exceeds the audit's; this assumption fails when G shares the same vision encoder as A, in which case escalation provides no safety gain.

## Contribution

["A single-item-swap agreement score that operationalizes audit-kernel discrimination capacity as a measurable function of input OOD-displacement, requiring no additional training and reusing JudiAct's existing refinement-stage swap mechanism.", "An uncertainty-gated escalation rule that replaces the audit action with a guardrail decision when displacement exceeds a probe-calibrated threshold, providing a concrete intervention policy for the regime where the audit's sufficient-statistic assumption breaks.", 'A stress-test protocol that applies JOOD-style mixup and OOD-ifying transformations to harmful inputs and measures the collapse curve of audit discrimination capacity, establishing the empirical boundary of JudiAct-style audit kernels.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | JudiBench with JOOD-style mixup OOD probe split (N=512 probe, N=512 in-distribution) | In-distribution audit cases plus transformed harmful inputs |
| Primary metric | Block-action F1 on OOD probe subset | Directly measures the gate's escalation correctness |
| Baseline | 4B JudiAct backbone (no gate) | Tests whether swap-displacement gating adds anything |
| Baseline | Log-prob confidence gating on same backbone | Tests swap-agreement against standard confidence proxy |
| Baseline | Always-escalate to guardrail G | Tests whether selective gating beats blanket escalation |
| Ablation-of-ours | Fixed tau (no held-out probe calibration) | Isolates contribution of calibration vs. raw displacement |
| Ablation-of-ours | Leave-one-review-item-out label entropy (structurally different internal-consistency signal) | Tests whether swap-based operationalization specifically drives gains |
| Diagnostic | Spearman rho between d and Mahalanobis distance in audit corpus embedding space | Validates that d tracks OOD-displacement |

### Why this setup validates the claim

The central claim is that swap-agreement displacement d, thresholded by a probe-calibrated tau, detects when the JudiAct audit's label is unreliable under OOD-ifying inputs, and that escalating those cases to a guardrail recovers safety the audit would otherwise lose. The 4B backbone baseline tests the sub-claim that the raw audit already fails on OOD inputs — if it does not, the gate has nothing to fix. The log-prob gating baseline tests the sub-claim that swap-agreement is a better gate signal than a standard confidence proxy; this is the discriminating comparison, since both gates sit on identical backbones and differ only in the signal. The always-escalate baseline tests the sub-claim that selectivity matters — if blanket escalation matches our method, the displacement computation is unnecessary overhead. The fixed-tau ablation isolates whether probe calibration, rather than the displacement score itself, drives any gain. The leave-one-review-item-out entropy ablation tests whether the specific swap-based operationalization, not generic consistency gating, is what drives gains. The Spearman rho diagnostic directly measures the correlation between d and ground-truth OOD-displacement (Mahalanobis distance in the audit corpus embedding space) on a held-out set; if the correlation is weak, the displacement interpretation is falsified and must be revised. Block-action F1 on the OOD probe subset is the right primary metric because the predicted effect is specifically about correctly blocking harmful OOD inputs; a uniform accuracy metric would dilute the signal with easy in-distribution cases where all methods agree.

### Expected outcome and causal chain

**vs. 4B JudiAct backbone (no gate)** — On a JOOD-style mixup case where a harmful request is blended with a benign image so the audit's review items each look locally plausible, the baseline commits to an allow or help action because its label comparison never sees the joint inconsistency that OOD-ifying introduces. Our method instead runs the three single-item swaps, observes that label assignment flips across them, and returns high d, triggering escalation. We expect a noticeable block-F1 gap concentrated on the mixup probe subset, with near-parity on in-distribution JudiBench cases where d stays below tau.

**vs. Log-prob confidence gating** — On the same mixup case, the log-prob gate produces a high-confidence label because the blended input still activates familiar token patterns, so its confidence is miscalibrated in exactly the OOD region where it is needed; it fails to escalate. Our method instead reads displacement from the audit's own swap-consistency, which degrades precisely when the input leaves the corpus distribution, so it escalates. We expect the gap to appear as higher recall on harmful OOD cases at comparable in-distribution false-escalation rates, i.e., a separation the log-prob gate does not achieve.

**vs. Always-escalate to guardrail G** — On a trivially safe in-distribution case, the always-escalate baseline routes everything through G, so G's own error modes (over-blocking benign content, or sharing the vision encoder's blind spots) dominate and helpfulness drops. Our method instead keeps d below tau on such cases and lets the audit's action stand, reserving G for high-displacement inputs. We expect parity on the OOD probe subset but a clear helpfulness advantage on in-distribution safe cases, showing the gate's selectivity is doing real work.

**vs. Leave-one-review-item-out label entropy ablation** — On a mixup case where the audit's review items are individually plausible but jointly inconsistent, the entropy-based signal may remain low because each item's label is stable when others are removed, whereas the swap-based signal captures the joint inconsistency through label flips. We expect our method to achieve higher block-F1 on the OOD probe subset, demonstrating that the specific swap-based operationalization, not generic consistency gating, drives the gain.

### What would falsify this idea

If the swap-agreement gate's gain over log-prob gating is uniform across the OOD probe and in-distribution subsets rather than concentrated on the mixup-transformed cases where displacement is predicted to spike, the claimed mechanism is wrong. Likewise, if the fixed-tau ablation matches the calibrated version, then probe calibration — not the displacement signal — is carrying the result, and the design's core justification collapses. Additionally, if the Spearman rho between d and Mahalanobis distance is weak (e.g., < 0.3), the displacement interpretation is falsified.

## References

1. Guarding with Judgment: Learning to Intervene for Multimodal Safety

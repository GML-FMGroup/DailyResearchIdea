# Monotone Fidelity–Completion Coupling for Proxy-Conditioned Video Diffusion

## Motivation

Across proxy-, flow-, and layout-conditioned video generators, the field implicitly assumes control-signal fidelity and generative detail are independently improvable, yet ProxyGenBench shows that greater proxy detail can coincide with poorer completion while strong structural adherence does not ensure correct binding; FloVD and IM-Zero condition on flow or layout priors with no mechanism preventing this trade-off. The root cause is structural: conditioning pathways are trained with a fidelity loss and a detail loss that are summed but never constrained to be co-monotone, so the optimizer is free to trade one against the other whenever proxy detail rises. This recurs as a meta-level blocker: every conditioning scheme inherits the same unconstrained two-term objective.

## Key Insight

Coupling the fidelity and completion losses through an explicit monotonicity penalty on their joint response to proxy-fidelity level turns an unconstrained trade-off into a constrained one, so that the conditioning pathway cannot buy structural adherence by sacrificing detail completion.

## Method

**What it is.** We introduce MFC (Monotone Fidelity–Completion), a training objective plus calibration protocol for proxy-conditioned video diffusion. Inputs: a proxy-conditioned diffusion model, a dataset of scenes with proxies rendered at K ordered fidelity levels (K=4: coarse blockout → dense blockout → blockout+masks → blockout+masks+text), and paired text/image/audio conditions. Outputs: a model whose fidelity and completion scores are jointly monotone in proxy fidelity level, and a calibration report verifying this.

**Load-bearing assumption (stated explicitly).** MFC assumes that the K proxy-fidelity levels form a total order on proxy informativeness such that both structural adherence and detail completion should be monotonically non-decreasing in level, so that any observed non-monotonicity is a training-pathway defect that a hinge penalty can remove rather than a genuine property of the proxy ladder. This assumption is load-bearing: if a higher level introduces artifacts or binding ambiguity that make completion legitimately non-monotone, L_mono would penalize a legitimate non-monotonicity. To make the assumption verifiable rather than assumed, MFC includes a per-scene, per-level informativeness calibration (Phase 0) that estimates an empirical 'informative' ordering from held-out checklist scores and applies L_mono only over level pairs whose informativeness is verified to increase, plus a learned per-scene slack term that lets the objective absorb genuine artifact-induced non-monotonicity instead of penalizing it; the slope constraint alpha is then conditioned on the calibrated informativeness gap rather than a fixed constant.

**How it works.**

```pseudocode
# Phase 0: per-scene informativeness calibration (held-out checklist scores)
for each held-out scene s:
    for k in 1..K:
        score f_k^cal, c_k^cal on case-specific checklist   # human or checklist scores
    # empirical informativeness gap between adjacent levels
    delta_f_k = f_{k+1}^cal - f_k^cal
    delta_c_k = c_{k+1}^cal - c_k^cal
    # gate: level pair (k,k+1) is 'informative' iff both checklist scores increase
    gate_k = 1[delta_f_k > 0 and delta_c_k > 0]
    # per-scene slack absorbs genuine artifact-induced non-monotonicity
    slack_k = max(0, -min(delta_f_k, delta_c_k))   # >=0, zero when both increase
    # calibrated slope target: completion gain per unit fidelity gain
    alpha_k = delta_c_k / max(delta_f_k, eps)      # conditioned on informativeness gap

# Phase 1: per-level scoring heads (frozen backbone, trained heads)
# FidelityHead: 2-layer MLP, hidden=256, GeLU, input = [x_k pooled features, p_k embedding]
# CompletionHead: 2-layer MLP, hidden=256, GeLU, input = [x_k pooled features, p_k embedding]
for each training clip with proxy levels p_1 < ... < p_K:
    for k in 1..K:
        x_k = denoise(backbone, p_k, conditions)   # sample
        f_k = FidelityHead(x_k, p_k)               # structural adherence score
        c_k = CompletionHead(x_k, p_k)             # detail-completion score

# Phase 2: monotonicity-coupled training objective
L = L_diffusion + lambda_f * L_fidelity + lambda_c * L_completion
    + lambda_m * L_mono
# L_mono penalizes non-monotone joint response, gated by informativeness and slack:
L_mono = sum_{k=1}^{K-1} gate_k * [ relu(f_k - f_{k+1} - slack_k) + relu(c_k - c_{k+1} - slack_k) ]
         + mu * sum_{k=1}^{K-1} gate_k * relu( (c_{k+1}-c_k) - alpha_k*(f_{k+1}-f_k) )
# second term forbids completion gains that outrun fidelity gains (violating structure)
# hyperparams: lambda_m=0.5, mu=0.25, K=4, alpha_k from Phase 0 (default alpha=1.0 when gate_k=0)

# Phase 3: calibration sweep
for held-out scene s:
    for k in 1..K:
        score f_k, c_k on case-specific checklist
    flag scene if f or c is non-monotone across k
    report per-scene monotonicity violation rate
    report gate fire rate: fraction of level pairs with gate_k=0
```

**Why this design.** We chose a pairwise hinge penalty over the K ordered proxy levels rather than a learned discriminator over (fidelity, completion) pairs because the hinge directly enforces the ordinal invariant that ProxyGenBench's finding violates, accepting the cost that it requires a fixed fidelity ladder and cannot extrapolate to unseen proxy styles. We chose two separate scalar heads (FidelityHead, CompletionHead) over a single joint score because the benchmark's checklists already decompose preservation from completion, accepting the cost that head calibration must be validated independently. We chose to couple the two monotonicity terms with a slope constraint (mu term) rather than penalizing only individual non-monotonicity because a model could otherwise improve completion while degrading structure and still pass the first term, accepting the cost of one extra hyperparameter alpha. We chose to gate L_mono by a per-scene informativeness calibration (Phase 0) and a learned slack term rather than applying it uniformly, because ProxyGenBench reports that greater proxy detail can coincide with poorer completion, so a hard total-order penalty would penalize legitimate non-monotonicity; the cost is that the gate can disable L_mono on scenes where the ladder is not genuinely ordered, which we report as a gate fire rate. Unlike proxy-guided benchmark evaluation (ProxyGenBench), which measures but does not alter the conditioning pathway, MFC modifies the training objective itself; a domain expert would not describe this as a benchmark variant because the mechanism is a constrained optimization over the conditioning response curve, not a new checklist.

**Why it measures what we claim.** The quantity f_k (FidelityHead output at level k) measures structural adherence because we assume the fidelity head is trained on the same preservation checklist used in ProxyGenBench; this assumption fails when the head overfits to proxy artifacts, in which case f_k reflects artifact reproduction rather than intended-structure preservation. The quantity c_k (CompletionHead output) measures detail-completion quality because we assume the completion head is trained on the completion/correction checklist; this assumption fails when missing content is ambiguous, in which case c_k reflects hallucination plausibility rather than correct completion. The quantity L_mono measures the absence of non-monotone trade-offs because we assume the K fidelity levels form a total order on proxy informativeness; this assumption fails when a higher level introduces artifacts that are not strictly more informative, in which case L_mono penalizes a legitimate non-monotonicity — which is why Phase 0 gates L_mono by a per-scene informativeness calibration and adds a slack term to absorb genuine artifact-induced non-monotonicity, and why Phase 3 reports the gate fire rate.

## Contribution

['A monotonicity-coupled training objective (MFC) that adds a pairwise hinge penalty over ordered proxy-fidelity levels, turning the fidelity–completion trade-off into a constrained optimization.', 'A calibration protocol that sweeps proxy fidelity levels and reports per-scene monotonicity violation rates for both fidelity and completion axes.', 'An empirical design principle: that co-monotone fidelity–completion response is achievable without sacrificing peak fidelity, established via the calibration sweep.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ProxyGenBench scenes with K-level proxy ladder | Provides ordered proxy fidelity and checklists |
| Primary metric | Per-scene monotonicity violation rate | Directly measures the ordinal invariant claimed |
| Secondary metric | Gate fire rate (fraction of level pairs with gate_k=0) | Reports how often informativeness calibration disables L_mono |
| Baseline | ProxyGenBench evaluated generators (unmodified) | Tests whether training objective matters, not just measurement |
| Baseline | Single-head joint score variant | Isolates benefit of decomposed fidelity/completion heads |
| Baseline | Monotonicity without slope constraint (mu=0) | Tests whether slope coupling prevents structure-for-completion trades |
| Baseline | Monotonicity without informativeness gate (gate_k=1 always) | Tests whether Phase 0 calibration prevents penalizing legitimate non-monotonicity |
| Ablation-of-ours | MFC with shuffled proxy-level ordering | Checks that gains require a genuine fidelity ladder |

### Why this setup validates the claim

The central claim is that a training objective enforcing joint monotonicity of fidelity and completion across an ordered proxy ladder reduces non-monotone trade-offs relative to models that merely measure them. ProxyGenBench is the right dataset because it already supplies controlled proxy-fidelity variations and checklists that decompose preservation from completion, so the monotonicity violation rate is computable per scene without inventing a new annotation scheme. The unmodified generators test the core sub-claim that measurement alone does not fix the trade-off; the single-head variant tests whether the two-head decomposition is load-bearing; the mu=0 variant tests whether the slope constraint is doing real work beyond individual monotonicity; the no-gate variant tests whether Phase 0 informativeness calibration is load-bearing by showing that ungated L_mono penalizes legitimate non-monotonicity on artifact-heavy scenes. The shuffled-ladder ablation is the sharpest control: if gains persist when level order is destroyed, the mechanism is not the ordinal invariant but something incidental. Together these form a falsifiable test: the predicted effect should appear specifically as reduced violation rates on scenes where the ladder is genuinely informative (gate_k=1), not uniformly everywhere, and the gate fire rate quantifies how often the informativeness assumption holds.

### Expected outcome and causal chain

**vs. ProxyGenBench evaluated generators (unmodified)** — On a case where a dense blockout proxy contains artifacts the generator faithfully reproduces, the baseline produces high structural adherence but poor completion because its conditioning pathway is unchanged and it has no pressure separating artifact reproduction from intended-structure preservation. Our method instead penalizes completion gains that outrun fidelity gains via the slope term, so the heads are pushed apart rather than collapsed, and we expect a noticeable violation-rate gap concentrated on high-fidelity, artifact-heavy scenes with near-parity on coarse blockouts.

**vs. Single-head joint score variant** — On a case where completion improves while structure degrades, the single-head variant produces a blended score that masks the trade-off because one scalar cannot express two opposing monotonicity constraints. Our method instead trains separate FidelityHead and CompletionHead with independent hinge penalties, so opposing movements are visible and penalized, and we expect a larger violation-rate reduction on scenes where preservation and completion pull in opposite directions, with parity on scenes where both improve together.

**vs. Monotonicity without slope constraint (mu=0)** — On a case where a higher proxy level yields slightly better completion but clearly worse structure, the mu=0 variant produces a passing first-term check because each head is individually monotone in the wrong direction being tolerated, since only individual non-monotonicity is penalized. Our method instead couples the two deltas with the alpha_k slope constraint, so completion gains that outrun fidelity gains are rejected, and we expect the gap to concentrate on the subset where completion rises faster than fidelity rather than uniformly across all scenes.

**vs. Monotonicity without informativeness gate (gate_k=1 always)** — On a case where a higher proxy level introduces artifacts that legitimately reduce completion, the no-gate variant penalizes this genuine non-monotonicity because it applies L_mono uniformly, driving the heads to fit the constraint rather than the data. Our method instead gates L_mono by Phase 0 informativeness calibration and adds a slack term, so legitimate artifact-induced non-monotonicity is absorbed rather than penalized, and we expect the no-gate variant to show higher violation rates on artifact-heavy scenes and a larger gap between head scores and checklist scores, with parity on scenes where the ladder is genuinely ordered (gate_k=1).

### What would falsify this idea

If the monotonicity violation-rate reduction is uniform across all scenes and proxy-fidelity subsets rather than concentrated where the predicted artifact-reproduction and structure-for-completion trade-offs occur, the central claim that the ordinal invariant is the causal mechanism would be wrong. Likewise, if the shuffled-ladder ablation shows the same gain as the true ladder, the improvement cannot be attributed to the monotonicity constraint at all.

## References

1. ProxyGenBench: Benchmarking Proxy-Guided Video Generation

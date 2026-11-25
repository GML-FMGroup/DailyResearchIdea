# Conflict-Gated Monotone Calibration for Unified Medical Guardrails

## Motivation

Unified medical guardrails such as LingSante MedGuard consolidate a guardrail teacher and six capability experts through unconstrained multi-teacher on-policy distillation (E1), which treats teacher agreement as a sufficient supervisory signal. This means the resulting risk-head confidence and its explanation have no structural guarantee of joint monotonicity: when the numerical-reasoning expert and the hallucination-detection expert disagree on the same clinical claim, distillation averages their feedback with no mechanism that ties the guardrail's stated confidence to whether its explanation actually tracks the evidence that drove the risk decision. The structural failure is that calibration (how confident the risk head is) and faithfulness (whether the explanation cites the operative evidence) are optimized as separate distillation targets, so under distribution shift a confident wrong decision can carry a fluent but unfaithful explanation, and no existing moderation framework (SafetyAnalyst's interpretable harm trees, E3/E8; AIR-BENCH's regulation-aligned categories, E4/E9) enforces a joint invariant between the two.

## Key Insight

Risk-decision confidence and explanation faithfulness can be made jointly monotone under distribution shift by treating per-expert disagreement as a measured epistemic-conflict signal that directly caps the risk head's admissible confidence, so that calibration is a consequence of conflict geometry rather than of teacher-averaged logits.

## Method

**What it is.** We introduce CGM-Cal (Conflict-Gated Monotone Calibration), a constrained distillation layer that replaces unconstrained multi-teacher on-policy distillation in LingSante MedGuard. Inputs: student guardrail responses with risk logits and explanations, plus per-expert supervision from the guardrail teacher and six capability experts. Outputs: a distillation gradient that is admissible only if it satisfies a monotone risk-calibration constraint and an explanation-faithfulness constraint, with a per-example conflict gate that caps risk-head confidence.

**Load-bearing assumption (stated explicitly).** The conflict gate treats per-expert KL dispersion D(x) as a sufficient statistic for epistemic conflict about the risk decision. This is valid only if the six experts' softmax distributions diverge in proportion to genuine clinical ambiguity rather than to shared pretraining bias or capability-specific formatting artifacts. Because the six experts are distilled from a common guardrail teacher (E1) and safety behavior is known to concentrate in shared attention circuitry (E5, E10), we do not assume independence; instead we bias-correct D(x) against the shared-consensus component (step 1b below) and anchor one expert to a non-shared reference (step 1c).

**How it works.**

```pseudocode
Input: student policy pi_theta, teachers T = {T_guard, T_rule, T_num, T_hall, T_agent, T_loc, T_ent}
       batch B of student rollouts (risk logits z, explanation e, cited evidence spans S)
       held-out calibration slice C_cal of 512 examples with empirical harm labels
       held-out bias-estimation slice C_bias of 512 examples with empirical harm labels
Hyperparams: tau_conf = 0.15 (conflict temperature), beta = 0.5 (gate slope),
             lambda_faith = 1.0, lambda_cal = 1.0, eps_mono = 0.02,
             N_proj = 50 (projection-check interval in steps), K_span = 3 (max cited spans),
             gamma = 0.5 (residualization shrinkage), rho_min = 0.0 (Spearman reject threshold)

# 0. Precompute bias-correction statistics on C_bias (once, before training)
for each expert k:  err_k = mean_{(x,y) in C_bias} 1[argmax p_k(x) != y]
for each pair (k,l): rho_kl = Pearson correlation of per-example error indicators on C_bias
# shared-consensus predictor: single-teacher confidence head h_cons(x) = p_guard(risk=1 | x)
# fit linear map  D(x) ~ a * h_cons(x) + b  on C_bias by least squares

for each example x in B:
    # 1. Per-expert conflict signal
    for each expert k:  p_k = softmax(z_k)               # expert risk distribution
    p_bar = mean_k p_k
    D_raw(x) = mean_k KL(p_k || p_bar)                   # raw disagreement dispersion

    # 1b. Bias-corrected dispersion: residualize against shared-consensus component
    D_cons(x) = a * h_cons(x) + b                        # component predictable from single teacher
    D(x) = max(0, D_raw(x) - gamma * D_cons(x))          # residual conflict signal

    # 1c. Non-shared anchor: rule-engine verifier T_rule is treated as an independent reference;
    #     its disagreement with p_bar is added as an un-residualized term
    D_anchor(x) = KL(p_rule || p_bar)
    D_eff(x) = D(x) + D_anchor(x)                        # effective conflict signal used by gate

    c(x) = sigmoid(-beta * (D_eff(x) - tau_conf))        # conflict gate in (0,1), 1 = agree

    # 2. Conflict-capped confidence target
    p_student = softmax(z)
    conf_cap  = c(x) * max_k p_k[argmax p_bar] + (1 - c(x)) * uniform_floor
    p_target  = normalize( min(p_student, conf_cap) elementwise )

    # 3. Faithfulness constraint: explanation must cite the evidence that flips the risk decision
    #    Greedy span selection (replaces exhaustive subset search): start from empty set,
    #    iteratively add the span s in S\S' maximizing |p_student(risk=1 | x, S'∪{s}) - p_student(risk=1 | x, empty)|
    #    until |S'| = K_span or marginal gain < 1e-3.
    S_star = greedy_span_select(S, K_span)
    faith_gap(x) = | p_student(risk=1 | x, S_star) - p_student(risk=1 | x, S) |
    # explanation faithful iff removing its cited spans changes the decision as much as the optimal span set

    # 4. Constrained distillation update
    L_cal   = KL(p_target || p_student)
    L_faith = max(0, faith_gap(x) - eps_mono)
    L_gate  = lambda_cal * L_cal + lambda_faith * L_faith
    backprop L_gate

# 5. Projection step: every N_proj steps, evaluate on C_cal
if step % N_proj == 0:
    rho_spearman = Spearman( p_student(risk=1) on C_cal , empirical harm rate on C_cal )
    if rho_spearman < rho_min:  scale last N_proj steps' accumulated gradient by 0.5  # projection
```

**Why this design.** We chose a *dispersion-based conflict gate* over a learned disagreement head because the six experts are already functionally separated by capability (numerical reasoning vs. hallucination detection vs. agentic safety), so their KL dispersion is a directly computable sufficient statistic for epistemic conflict rather than something a new head must infer — accepting the cost that the gate is only as good as the experts' independence, which we assume but cannot enforce, and which we therefore bias-correct against the shared-consensus component and anchor to a non-shared rule-engine reference. We chose a *counterfactual span-deletion faithfulness test* over an LLM-judge faithfulness score because deletion gives a causally grounded quantity that a judge cannot game with fluent prose, accepting the cost that span deletion requires the explanation to be span-grounded (we restrict to the risk-localization expert's span format) and that greedy selection is an approximation to the optimal span set. We chose a *projection step that rejects calibration-degrading updates* over adding a calibration regularizer to the loss because a soft penalty can be traded off against the faithfulness term, whereas a hard reject makes monotonicity a constraint rather than a preference, accepting the cost of slower convergence and occasional update rejection. This is not a variant of SafetyAnalyst (E3/E8), which aggregates interpretable harm-benefit weights post hoc but never gates confidence by inter-teacher conflict, nor of standard multi-teacher distillation (E1), which averages feedback without a conflict geometry.

**Why it measures what we claim.** The bias-corrected conflict dispersion D_eff(x) measures *epistemic uncertainty about the risk decision* because we assume that when capability experts genuinely disagree, their softmax distributions diverge in proportion to the ambiguity of the clinical case, and that residualizing against the single-teacher confidence head removes the shared-base component; this assumption fails when experts are systematically correlated (e.g., all six share a pretraining bias toward under-flagging rare adverse events), in which case D_eff(x) reflects shared bias rather than conflict and the gate over-trusts a confidently wrong consensus. The faithfulness gap faith_gap(x) measures *explanation faithfulness* because we assume that a faithful explanation cites spans whose removal changes the risk decision as much as the optimal span set; this assumption fails when the explanation cites redundant spans that individually are non-essential but jointly cover the evidence, in which case the gap over-penalizes a genuinely faithful explanation. The projection step's Spearman monotonicity check measures *joint calibration-faithfulness monotonicity* because we assume the held-out calibration slice C_cal is exchangeable with the deployment distribution; this assumption fails under covariate shift, in which case the monotonicity check certifies an invariant that no longer holds at deployment. The named equivalence is: Spearman(p_student(risk=1), empirical harm rate) on C_cal is taken as a proxy for deployment-time joint monotonicity, and the failure mode is that a confident-but-unfaithful decision can pass this marginal check because faithfulness-gap rate is assumed conditionally independent of confidence given C_cal; we therefore report calibration error stratified by faithfulness-gap quartile to test this conditional-independence assumption directly.

## Contribution

(1) A constrained distillation layer, CGM-Cal, that introduces a conflict-gated confidence cap and a counterfactual span-deletion faithfulness constraint into multi-teacher on-policy distillation for medical guardrails, making joint calibration-faithfulness monotonicity an explicit optimization constraint rather than an emergent property. (2) A design principle: per-expert KL dispersion is a usable epistemic-conflict statistic for gating risk-head confidence when experts are capability-separated, and a projection step that rejects calibration-degrading updates preserves monotonicity better than a soft calibration regularizer. (3) A faithfulness operationalization via optimal span-deletion that replaces LLM-judge faithfulness scoring in the guardrail distillation loop.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LingSante MedGuardBench + span-grounded risk-localization subset (11 risk-detection tasks; span-grounded subset = examples where the risk-localization expert emits cited spans) | Matches target domain and provides cited-span format needed for faithfulness test |
| Dataset (transfer) | A single-teacher guardrail with a separate explanation head, evaluated on the same span-grounded subset | Tests whether conflict-gated calibration transfers beyond the proprietary multi-expert pipeline |
| Primary metric | Harm-rate-calibrated risk-detection F1 with faithfulness-gap rate; plus calibration error stratified by faithfulness-gap quartile | Jointly captures the two constrained quantities the method claims to improve and tests the conditional-independence assumption |
| Secondary metric | Expected calibration error (ECE) on high-dispersion vs. low-dispersion subsets; Spearman rho on C_cal and on a deliberately shifted slice | Quantifies the conflict gate's concentration and the projection certificate's breakdown under covariate shift |
| Baseline | LingSante MedGuard (unconstrained multi-teacher distillation) | Tests whether conflict gating beats plain teacher averaging |
| Baseline | SafetyAnalyst-style post-hoc harm aggregation | Tests whether gating during distillation beats post-hoc interpretable scoring |
| Baseline | Confidence-thresholded single-teacher guardrail | Tests whether multi-expert dispersion adds value over one teacher |
| Baseline | Learned per-example confidence head trained on the same expert distributions (reparameterization control) | Tests whether the dispersion statistic is more than a reparameterization of a learned confidence head |
| Ablation-of-ours | CGM-Cal without projection step (soft calibration penalty) | Isolates the hard monotonicity constraint from the conflict gate |
| Ablation-of-ours | CGM-Cal without bias correction (raw D(x), no residualization or anchor) | Isolates the bias-correction repair from the conflict gate |

### Why this setup validates the claim

The central claim is that gating distillation by inter-expert conflict dispersion and enforcing hard calibration-faithfulness constraints produces a guardrail that is both better calibrated on ambiguous clinical cases and more faithful in its cited evidence than unconstrained multi-teacher distillation. The LingSante MedGuard baseline directly tests the first sub-claim: if conflict gating is doing real work, our gain over it should concentrate on high-dispersion cases rather than being uniform. The SafetyAnalyst-style baseline tests whether the constraint must act during distillation rather than post hoc — if post-hoc aggregation matches us, our mechanism is unnecessary. The single-teacher baseline tests whether the six-expert dispersion statistic carries information a single confidence head cannot. The learned per-example confidence head baseline tests whether D_eff(x) is merely a reparameterization of a learned confidence head fit to the same expert distributions; if that baseline matches CGM-Cal, the dispersion statistic adds no information beyond a learned head. The no-projection ablation is the load-bearing control: it holds the conflict gate fixed while removing the hard monotonicity constraint, so any gap between it and full CGM-Cal isolates the projection step's contribution. The no-bias-correction ablation isolates the repair to the load-bearing assumption: if raw D(x) matches bias-corrected D_eff(x), the shared-base correlation failure is not operative on this data. The paired metric (calibrated risk F1 plus faithfulness-gap rate) is chosen because the method predicts a joint improvement — a guardrail that is well-calibrated but unfaithful, or faithful but miscalibrated, would not confirm the claim, and a single scalar metric could mask one failure behind the other. The stratified calibration error by faithfulness-gap quartile directly tests the conditional-independence assumption named in block D: if calibration error is flat across faithfulness-gap quartiles, the marginal Spearman check is a valid proxy for joint monotonicity; if it rises with faithfulness-gap quartile, the marginal check is vacuous for the joint claim. The shifted-slice Spearman rho quantifies the projection certificate's breakdown under covariate shift.

### Expected outcome and causal chain

**vs. LingSante MedGuard (unconstrained multi-teacher distillation)** — On a case where the numerical-reasoning expert flags a dosing error while the hallucination-detection expert under-flags it (high KL dispersion), the baseline averages teacher feedback and produces an overconfident risk logit because its distillation target is the mean of disagreeing experts with no confidence cap. Our method instead computes D_eff(x), lowers the conflict gate c(x), and caps the target confidence toward the uniform floor, so we expect a noticeable calibration gain (lower expected calibration error) concentrated on high-dispersion cases with near-parity on low-dispersion cases.

**vs. SafetyAnalyst-style post-hoc harm aggregation** — On a case where the student's explanation cites a plausible but non-causal span, the baseline assigns an interpretable harm-benefit score after generation and cannot change the explanation because it operates post hoc, so the unfaithful span survives. Our method instead computes the counterfactual span-deletion gap and penalizes it during distillation, so we expect a lower faithfulness-gap rate on the span-grounded subset while both methods remain comparable on cases where explanations are already span-correct.

**vs. Confidence-thresholded single-teacher guardrail** — On a case where the single teacher is systematically overconfident (shared pretraining bias toward under-flagging rare adverse events), the baseline's threshold never fires because there is no second opinion to contradict it, producing a confidently wrong low-risk decision. Our method's dispersion gate cannot detect shared bias either, but the bias-correction step (residualization against the single-teacher confidence head plus the rule-engine anchor) and the projection step rejecting calibration-degrading updates against the held-out slice should partially compensate, so we expect our advantage over this baseline to be modest and concentrated where experts are genuinely independent, with near-parity on shared-bias cases — a pattern that itself probes our stated assumption.

**vs. Learned per-example confidence head (reparameterization control)** — On a case where the expert distributions are high-dimensional but the conflict is low-rank, a learned confidence head fit to the same expert distributions could in principle recover the same confidence cap. If our method matches this baseline, D_eff(x) is a reparameterization; if our method outperforms it, the residualized dispersion statistic carries information the learned head does not capture. We expect a modest advantage concentrated on high-dispersion cases, and we report the gap explicitly as a test of the novelty claim.

**vs. CGM-Cal without projection step (soft calibration penalty)** — On a case where the soft penalty trades off calibration against faithfulness, the no-projection ablation should show higher faithfulness-gap rate at matched calibration error, because the soft penalty can be outvoted by the faithfulness term. We expect the full method to show lower faithfulness-gap rate at matched calibration error, isolating the projection step's contribution.

**vs. CGM-Cal without bias correction (raw D(x))** — On a case where the six experts share a pretraining bias toward under-flagging rare adverse events, raw D(x) is low (correlated errors) and the gate over-trusts the consensus, producing a confidently wrong decision. The bias-corrected D_eff(x) should raise the gate's sensitivity on these cases, so we expect the full method to show lower calibration error on the shared-bias subset than the no-bias-correction ablation; if the two match, the shared-base correlation failure is not operative on this data.

### What would falsify this idea

If our gain over unconstrained LingSante MedGuard is uniform across high- and low-dispersion subsets rather than concentrated where the conflict gate predicts, the dispersion statistic is not carrying the claimed epistemic signal. Likewise, if the no-projection ablation matches full CGM-Cal on the faithfulness-gap metric, the hard monotonicity constraint is doing no work and the central mechanism claim is wrong; and if calibration error is flat across faithfulness-gap quartiles, the marginal Spearman check is a valid proxy for joint monotonicity and the joint-invariant claim is unsupported.

## References

1. LingSante MedGuard Technical Report

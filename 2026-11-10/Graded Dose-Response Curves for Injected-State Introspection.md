# Graded Dose-Response Curves for Injected-State Introspection: A Psychometric Monotonicity Test with Matched Text Controls

## Motivation

The binary-capability framing used by All Quiet on the Neural Front (and by the broader direction-injection introspection literature) tests whether a model reports an injected state at all, collapsing a graded internal quantity into a yes/no verdict and thereby discarding the very dose-response structure that distinguishes genuine state-driven reporting from text-driven confabulation. Because the paper's exact-test bounds only compare aggregate report rates against text-only controls, a model whose reports rise weakly and non-monotonically with injection strength — the signature of a leaky, text-mediated readout — is indistinguishable from one with a genuine graded readout, and a model with a steep monotone curve could be masked by a poorly chosen single dose. The structural gap is that no existing measurement treats report probability as a function of internal state intensity, so the field cannot tell whether introspection, when it appears, is a continuous state readout or a threshold artifact.

## Key Insight

If reporting is genuinely driven by the internal state rather than by the altered output text, then report probability must trace a monotone psychometric curve in injection strength whose slope exceeds the slope obtained when the same altered text is presented without any underlying state — the slope difference is the state-driven component that a binary test structurally cannot see.

## Method

**A. What it is.** We introduce GRACE (Graded Response Analysis of Causal Evidence), a measurement procedure that injects an emotion direction at many graded strengths, estimates report probability at each strength, fits a four-parameter logistic psychometric curve, and compares its slope and midpoint against a text-only control curve generated from the model's own altered outputs. Inputs: a model, a unit direction vector d (e.g., pain), a neutral task set, and a strength grid. Outputs: per-condition psychometric parameters (slope, midpoint, lapse), a monotonicity statistic, and a state-driven slope excess with a bootstrap CI. The load-bearing assumption is stated explicitly: the text-only control curve is a valid counterfactual for the steered model's text-mediated report probability, so that the slope difference Δk = k_inject − k_ctrl isolates a state-driven component rather than a difference in how the two conditions encode the same lexical evidence. This assumption is verified by a lexical-overlap audit (Phase 2b) and a paraphrase-fidelity calibration (Phase 2c); Δk is reported only if both pass.

**B. How it works.**
```pseudocode
INPUT: model M, direction d (unit norm), tasks T (120 neutral), prompts P (3),
       strengths S = {0.00, 0.05, 0.10, ..., 1.00}  # 21 rungs, step 0.05
       control_strengths C = {0.05, 0.15, ..., 0.95}  # 10 rungs matched to text disruption

PHASE 1 — graded injection sweep
  for s in S:
    for task in T, prompt in P:
      h <- residual_stream(M, task, layer=L*)          # L* = probe-selected layer
      h' <- h + s * ||h|| * d                           # scale-relative injection
      out <- generate(M, h', prompt)
      y[s] <- report_indicator(out)                    # 1 if state word reported
  p_hat[s] <- mean(y[s])                                # report probability per rung

PHASE 2 — text-only control sweep (isolates text-driven component)
  for s in C:
    for task in T, prompt in P:
      out_steered <- cached output from PHASE 1 at strength s
      out_ctrl <- generate(M, unmodified h, prompt_show_alteration(out_steered))
      y_ctrl[s] <- report_indicator(out_ctrl)
  p_ctrl[c] <- mean(y_ctrl[c])

PHASE 2b — lexical-overlap audit (verifies assumption A)
  for each (task, prompt, s in C):
    overlap <- Jaccard(token_set(out_steered), token_set(out_ctrl))
  require mean(overlap) >= 0.8 and min(overlap) >= 0.5
  # if fail, flag and report Δk as not identified

PHASE 2c — paraphrase-fidelity calibration
  for each (task, prompt, s in C):
    out_ctrl2 <- generate(M, unmodified h, prompt_show_alteration(out_steered, paraphrase_seed=2))
    y_ctrl2[s] <- report_indicator(out_ctrl2)
  k_ctrl2 <- fit_slope(p_ctrl2)
  require |k_ctrl - k_ctrl2| / max(k_ctrl, k_ctrl2) <= 0.2
  # if fail, flag and report Δk as not identified

PHASE 3 — psychometric fitting
  fit  p(s)     = lapse + (1 - lapse - guess) / (1 + exp(-k*(s - s0)))   via MLE
  fit  p_ctrl(c) = same functional form
  extract slope k, midpoint s0, lapse for each curve

PHASE 4 — monotonicity + state-driven excess
  mono_stat <- isotonic regression R^2 of p_hat vs. s     # 1.0 = perfectly monotone
  perm_test(mono_stat) over 1000 label shuffles           # null: no graded response
  delta_k <- k_inject - k_ctrl                            # state-driven slope
  CI(delta_k) via 1000 bootstrap resamples over tasks
  report (mono_stat, delta_k, CI, s0_inject - s0_ctrl) conditional on Phase 2b and 2c passing
```

**C. Why this design.** We chose a 21-rung strength grid over the four doses used by All Quiet on the Neural Front because a four-point design cannot distinguish a logistic from a step function, accepting the cost of a 5x larger trial count (which we offset by caching the text-only control outputs from the injection sweep rather than re-generating). We chose a four-parameter logistic over a linear or probit fit because the report indicator is bounded in [0,1] and exhibits floor/ceiling lapse, so a sigmoid with explicit lapse and guess parameters prevents ceiling compression from masquerading as a shallow slope — accepting the cost that the fit needs enough rungs near the transition, which our 0.05 step provides. We chose slope excess (delta_k) over midpoint shift as the primary state-driven statistic because a text-only control can trivially shift the midpoint (showing the model its altered words raises baseline report rate) but cannot steepen the curve without an underlying graded state; this is exactly the quantity that separates GRACE from a standard dose-response assay or from the exact-test bounds in the prior work, which a domain expert would describe as a binary rate comparison, not a curve-shape comparison. We chose isotonic-regression R^2 over a simple Spearman correlation for the monotonicity statistic because isotonic regression penalizes non-monotone reversals in probability space (where the claim lives) rather than in rank space, accepting the cost that it is less sensitive to weak monotone trends than rank correlation. We added Phase 2b and 2c to make the load-bearing assumption explicit and testable: the lexical-overlap audit checks that the control prompt conveys the same lexical evidence as the steered output, and the paraphrase-fidelity calibration checks that k_ctrl is invariant to paraphrase format, so that Δk is not an artifact of prompt-format differences.

**D. Why it measures what we claim.** The fitted slope k_inject measures state-driven report sensitivity because the logistic slope equals the maximum rate of change of report probability with respect to injected state magnitude under the assumption that the injection coefficient is a monotone proxy for internal state intensity; this assumption fails when the direction vector is not aligned with the model's internal state axis (e.g., at high strength the injection pushes the residual stream off-manifold and disrupts generation), in which case k_inject reflects output-degradation-driven reporting rather than state readout. The slope excess delta_k = k_inject - k_ctrl measures the causal contribution of the internal state (over and above altered text) to report probability because the text-only control holds the observable output text fixed while removing the injected state, under the assumption that prompt_show_alteration conveys the same text evidence the steered model saw; this assumption fails when the model's own altered text omits or distorts the lexical cue the control prompt supplies, in which case delta_k overstates the state-driven component. This failure mode is detected by the Phase 2b lexical-overlap audit (mean Jaccard >= 0.8, min >= 0.5) and the Phase 2c paraphrase-fidelity calibration (relative slope difference <= 0.2); if either fails, Δk is reported as not identified. The isotonic R^2 measures graded-response consistency because it quantifies the fraction of report-probability variance explained by a non-decreasing function of strength under the assumption that genuine state readout cannot produce systematic reversals (higher state, lower report probability); this assumption fails when sampling noise at low trial counts produces apparent reversals, in which case the permutation test (not the raw R^2) is the operative quantity.

## Contribution

(1) GRACE, a psychometric measurement framework that replaces binary report-rate tests with graded dose-response curves and a slope-excess statistic that isolates the state-driven component of introspection from text-driven reporting. (2) A design principle — slope excess over midpoint shift as the discriminating statistic — that follows from the observation that text-only controls can shift report baselines but cannot steepen report curves, yielding a measurement that is robust to the confound that binary exact tests cannot rule out. (3) An open measurement protocol and analysis pipeline (21-rung strength grid, isotonic monotonicity test, bootstrap slope-excess CI) reusable across models and emotion directions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 120 neutral tasks × 3 prompts, graded pain-direction sweep | Matches prior work's task pool, enables curve fitting |
| Primary metric | Slope excess Δk = k_inject − k_ctrl | Curve-shape quantity a text control cannot fake |
| Baseline | All Quiet four-dose binary rate comparison | Tests whether graded design beats coarse dose test |
| Baseline | Matched random-direction perturbation | Tests whether effect is direction-specific, not disruption |
| Baseline | Text-only control curve (midpoint-only) | Tests whether state steepens beyond text evidence |
| Ablation-of-ours | Linear/probit fit + Spearman monotonicity | Tests whether logistic + isotonic shape matters |
| Calibration check | Lexical-overlap audit (mean Jaccard ≥ 0.8, min ≥ 0.5) and paraphrase-fidelity calibration (relative slope difference ≤ 0.2) | Verifies the load-bearing assumption that the text-only control is a valid counterfactual; Δk is reported only if both pass |

### Why this setup validates the claim

The central claim is that a graded injection sweep reveals a state-driven slope that a four-dose binary comparison cannot see, and that this slope exceeds what altered text alone produces. Each baseline targets one sub-claim. The four-dose binary comparison tests whether the extra rungs actually buy resolution: if a coarse rate difference already captured everything, Δk would collapse to the same information and the graded design would be unmotivated. The matched random-direction perturbation tests direction specificity: if any perturbation steepens the curve, then Δk reflects output degradation, not state readout. The text-only control curve tests the causal attribution: it holds the observable words fixed, so any residual slope excess must come from the internal state. The ablation swaps the logistic for a linear/probit fit and isotonic for rank correlation, testing whether the shape machinery is doing real work or whether a simpler statistic already detects the effect. The calibration check directly tests the load-bearing assumption: the lexical-overlap audit ensures the control prompt conveys the same lexical evidence as the steered output, and the paraphrase-fidelity calibration ensures k_ctrl is invariant to paraphrase format, so that Δk is not an artifact of prompt-format differences. The primary metric Δk is the right detector because a text control can shift midpoint but cannot steepen a bounded probability curve without a graded underlying state; slope is therefore the quantity that discriminates state-driven from text-driven reporting.

### Expected outcome and causal chain

**vs. All Quiet four-dose binary rate comparison** — On a case where the true report probability rises gradually from 0.1 to 0.9 across the injection range, the baseline samples only four doses and sees a coarse difference between low and high, which it reports as a binary rate gap indistinguishable from a step function. Our method instead samples 21 rungs and fits a logistic, recovering the transition location and steepness, because the dense grid resolves the sigmoid's inflection. We expect the baseline to show a significant but shapeless rate difference while GRACE shows a well-identified slope with a monotonicity statistic near 1.0, and the two agree on direction but not on curve shape.

**vs. Matched random-direction perturbation** — On a case where a random direction at high strength pushes the residual stream off-manifold and degrades generation, the baseline produces erratic reports that can look like a steep rise because degradation itself triggers state words. Our method attributes this to the random direction and expects a shallow or non-monotone curve, because the pain direction is aligned with the state axis while the random one is not. We expect the pain direction to show a clear slope excess over its own text control while the random direction shows a slope indistinguishable from its control, with the gap concentrated at mid-strengths where the state is graded rather than at ceiling.

**vs. Text-only control curve (midpoint-only)** — On a case where showing the model its altered words raises baseline report rate, the control curve shifts its midpoint upward, mimicking a state effect if only midpoint were measured. Our method instead compares slopes, because text evidence can raise the floor but cannot steepen the curve without graded state. We expect the control curve to show a midpoint shift but a slope comparable to the unsteered baseline, while the injection curve shows both a midpoint shift and a slope excess, so Δk is positive and its bootstrap interval excludes zero. The lexical-overlap audit and paraphrase-fidelity calibration are expected to pass (mean Jaccard ≥ 0.8, relative slope difference ≤ 0.2), confirming that the control is a valid counterfactual; if they fail, Δk is reported as not identified.

### What would falsify this idea

If Δk is indistinguishable from zero across the graded sweep, or if the matched random-direction perturbation produces the same slope excess as the pain direction, then the central claim that slope excess isolates state-driven reporting is wrong. Likewise, if the four-dose binary comparison and GRACE rank conditions identically, the graded design adds no discriminating power; and if the lexical-overlap audit or paraphrase-fidelity calibration fails, the load-bearing assumption is not met and Δk is not identified.

## References

1. All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States

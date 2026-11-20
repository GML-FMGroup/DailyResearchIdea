# Shift-Robust Conformal Recalibration of System-1 Fallback Thresholds in Compiled Agent Workflows

## Motivation

AnyS1 (Thinking Fast, Anywhere) and AgentSquare both calibrate a single gate — a confidence threshold for System-1 fallback, or a predictor skip rate — on a fixed mining/training distribution, so the reported 75–91% episode-cost reduction and 17.2% gain are point estimates with no structural guarantee once the task distribution departs from training. The root cause is that calibration is treated as a one-shot offline fit rather than a distribution-dependent quantity: when covariate shift changes the frequency or difficulty of enumerable decision points, the threshold's cost/error trade-off silently drifts, and neither work provides a mechanism that detects or corrects it at deployment time.

## Key Insight

The System-1/System-2 fallback decision is a selective-prediction problem whose risk is exchangeability-dependent, so replacing a fixed threshold with a Mondrian split-conformal quantile computed over decision-point strata and updated online restores a distribution-free, shift-conditional risk guarantee on the System-1 acceptance set without retraining the 4B pointer.

## Method

**What it is.** SROC (Shift-Robust Online Conformal fallback) takes as input a compiled AnyS1 workflow, a stream of runtime decision points each with a System-1 candidate set and calibrated confidence, and a target System-1 error budget α (default 0.05). It outputs, per decision point, a binary accept-System-1 / escalate-to-System-2 action, together with an online-updated per-stratum threshold that maintains the error budget under covariate shift.

**Load-bearing assumption (stated explicitly).** SROC's error-budget guarantee holds only if, within each stratum, the nonconformity score is monotone in empirical decision correctness — i.e., higher score means more likely wrong — and this monotonicity is stable under the shift being corrected. We do not assume the raw System-1 confidence satisfies this; we fit a per-stratum score that does, and we verify it on held-out data before trusting q_s.

**How it works.**

```pseudocode
# Offline (once, on Any-DP held-out split)
# Step 0: fit a proper nonconformity score per stratum
for each stratum s:
    fit isotonic regression f_s: System1_confidence -> P(correct | confidence, stratum=s)
    on Any-DP_calib points with stratum(d)=s (minimum 200 points; else pool with parent stratum)
    score(d) = 1 - f_s(System1_confidence(d))   # calibrated nonconformity score

# Step 0b: per-stratum monotonicity / separation check
for each stratum s:
    compute Spearman rho_s between score(d) and 1[correct(d)] on Any-DP_calib
    compute AUC_s of score(d) separating correct vs incorrect
    if rho_s < 0.3 or AUC_s < 0.65:
        mark stratum s as UNRELIABLE -> escalate-all policy for s (q_s = -inf)

# Step 1: Mondrian split-conformal quantiles on the calibrated score
for each reliable stratum s:
    q_s = Quantile(1 - α, {score(d) : stratum(d)=s, d in Any-DP_calib})

# Online (per deployed workflow episode)
for each runtime decision point d_t:
    s = stratum(d_t)
    if s is UNRELIABLE: escalate to System-2; continue
    if score(d_t) <= q_s:  accept System-1
    else:                  escalate to System-2
    # after observing ground-truth correctness y_t of the accepted/rejected decision
    # (y_t may arrive with delay k steps; see delayed-feedback protocol below)
    if d_t was accepted and y_t == WRONG:
        push score(d_t) into sliding window W_s (length L=200)
        q_s = Quantile(1 - α, W_s ∪ {q_s})   # online recalibration
    # drift detector: two-sample KS between W_s and calibration scores
    if KS_pvalue(W_s, calib_s) < 0.01:
        flag stratum s as shifted; tighten q_s by factor γ=0.9 until KS clears
```

**Why this design.** We chose Mondrian (per-stratum) conformal quantiles over a single global threshold because the AnyS1 decision-point distribution is heterogeneous across skill families and candidate-set sizes — a global quantile would over-accept easy strata and under-accept hard ones under shift — accepting the cost that sparse strata need pooling fallbacks. We chose an online sliding-window recalibration over periodic full recalibration because the cost guarantee must hold between retraining cycles, accepting that the window length L trades responsiveness against variance in q_s. We chose a KS-based drift detector over a fixed schedule because covariate shift is episodic and non-stationary, accepting the false-positive cost of occasional unnecessary tightening. We chose a per-stratum isotonic recalibration of System-1 confidence over the raw confidence score because raw softmax confidence is miscalibrated and non-monotone under shift (Guo et al. 2017; Ovadia et al. 2019), and a conformal quantile over a non-monotone score bounds the wrong quantity; the isotonic fit is a fixed function of the input-label pair on held-out data, restoring the exchangeability the quantile needs within each stratum. We added the monotonicity/separation check because the isotonic fit can still fail on a stratum with too few points or a genuinely novel candidate-set structure, and in that case the only safe action is escalate-all rather than trusting q_s. This is not a variant of standard split conformal prediction (Vovk et al.) nor of AnyS1's original calibration: standard conformal assumes exchangeability and produces a static threshold, whereas SROC explicitly breaks exchangeability via stratum-conditional online updates and a drift-triggered tightening rule, which a domain expert would recognize as a selective-prediction-with-shift mechanism rather than a calibration wrapper.

**Why it measures what we claim.** The nonconformity score `1 - f_s(System1_confidence(d))` measures System-1 decision unreliability because f_s is an isotonic fit of empirical correctness against confidence within stratum s, so the score is monotone in the observed error rate by construction on Any-DP_calib; this assumption fails when a new skill family produces out-of-vocabulary candidate sets for which f_s was never fit, in which case the score reflects extrapolated isotonic values rather than decision correctness, and the monotonicity check flags the stratum as UNRELIABLE. The stratum definition `(skill_family, candidate_set_size_bucket, entropy_bucket)` measures covariate-shift relevance because decision-point difficulty is exchangeable within these buckets but not across them; this assumption fails when a shift changes difficulty without changing any bucket label, in which case the online KS detector — not the stratum — carries the correction. The KS p-value measures distribution shift because it tests equality of the accepted-score distribution against calibration; this assumption fails under adversarial (non-covariate) shift, in which case the tightening factor γ provides a conservative fallback rather than a calibrated one. The sliding-window quantile q_s measures the true stratum-conditional System-1 error rate because accepted scores within window length L are assumed exchangeable and L is large enough for quantile stability; this assumption fails under rapid non-stationary shift shorter than L, or in sparse strata where L exceeds available samples, in which case q_s reflects a stale or high-variance estimate and coverage error grows — we report coverage error as a function of L and stratum sparsity.

## Contribution

["SROC, a stratum-conditional online conformal recalibration layer that replaces AnyS1's single fixed fallback threshold with a drift-aware, per-decision-point acceptance rule for System-1/System-2 routing.", 'A drift-triggered tightening mechanism that couples a two-sample KS test on accepted-decision scores to threshold adjustment, giving a deployable procedure for maintaining the System-1 error budget when the task distribution departs from the Any-DP mining distribution.', 'An empirical characterization of how the 75–91% cost reduction reported by AnyS1 degrades under controlled covariate shift, and how much of that degradation SROC recovers relative to a fixed-threshold baseline.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Three unseen agent benchmarks with Any-DP decision points (same three as AnyS1) | Matches AnyS1 evaluation, exposes shift strata |
| Primary metric | Cost per episode at fixed error budget α=0.05 | Directly tests efficiency-under-reliability claim |
| Secondary metric | Per-stratum System-1 error rate vs. α (coverage gap) | Tests whether error budget holds where shift occurs |
| Secondary metric | Spearman rho and AUC of score vs. correctness per stratum | Validates the load-bearing monotonicity assumption |
| Baseline | Uncompiled System-2 agent | Upper-reliability, high-cost reference point |
| Baseline | AnyS1 with static calibrated threshold | Tests whether online shift-robustness adds value |
| Baseline | Global split-conformal fallback (no strata) | Tests whether Mondrian stratification is necessary |
| Baseline | SROC with raw confidence score (no isotonic recalibration) | Isolates contribution of proper nonconformity score |
| Ablation-of-ours | SROC without KS drift detector | Isolates contribution of drift-triggered tightening |
| Ablation-of-ours | SROC with window length L ∈ {50, 100, 200, 500, 1000} | Tests sensitivity of online quantile to L |
| Ablation-of-ours | SROC with delayed feedback k ∈ {0, 1, 5, 10} steps | Tests coverage degradation under label delay |

### Why this setup validates the claim

The central claim is that SROC preserves AnyS1's reliability guarantee under covariate shift while retaining most of its cost savings, and that this margin comes specifically from stratum-conditional online recalibration plus drift detection. The uncompiled System-2 baseline establishes the reliability ceiling and the cost floor the method must beat; AnyS1 with a static calibrated threshold tests whether shift-robustness is needed at all, since under exchangeability it should match SROC — a gap in SROC's favor on shifted subsets is the key signal. The global-conformal baseline isolates stratification: if per-stratum quantiles matter, the global threshold should over-accept easy strata and under-accept hard strata under shift, producing error-budget violations concentrated in specific strata. The raw-confidence baseline isolates the isotonic recalibration: if the load-bearing monotonicity assumption is doing real work, SROC with raw confidence should show larger coverage gaps on shifted strata than SROC with the calibrated score, and the per-stratum Spearman/AUC table should show which strata fail the monotonicity check. The no-drift-detector ablation isolates the tightening mechanism, so any gain attributable to episodic shift detection is separated from the base online recalibration. The L-sweep and delayed-feedback sweep test the stationarity assumption behind the sliding-window quantile: if coverage error grows sharply as L increases or as feedback delay k increases, the online recalibration is not tracking the true error rate at the claimed timescale. Cost per episode at a fixed error budget α is the right primary metric because it directly operationalizes "efficiency subject to reliability": a method that cuts cost but violates α is not efficient in the claimed sense, and a method that holds α but saves nothing is not efficient at all. Together these choices make the claim falsifiable: the predicted pattern is cost savings with α held on shifted subsets, not uniform savings everywhere.

### Expected outcome and causal chain

**vs. Uncompiled System-2 agent** — On a case where a routine decision point selects among a small known candidate set but the workflow is deployed on a benchmark with a novel skill family, the uncompiled agent invokes System-2 at every step because it has no compiled decision-point structure, so it pays full model-call cost and latency on every trivial choice. Our method instead accepts System-1 at those decision points because the stratum-conditional quantile q_s was calibrated on held-out Any-DP points of the same skill family, and only escalates when score exceeds q_s. We expect a large cost-per-episode reduction (in the range AnyS1 reports) with reliability parity on the benchmark's overall success rate, since escalation still covers the hard decisions.

**vs. AnyS1 with static calibrated threshold** — On a case where the deployed episode drifts toward a harder sub-distribution (e.g., longer candidate sets than calibration), the static-threshold baseline keeps accepting System-1 at the original rate because its quantile never updates, so its System-1 error rate climbs above α on that sub-distribution while overall cost stays low. Our method instead pushes misclassified accepted scores into the sliding window, raises q_s for the affected stratum, and escalates more often there, because the online recalibration tracks the observed error rate rather than assuming exchangeability. We expect SROC to show a visible gap in error-budget adherence on the shifted subset, at a modest cost increase confined to that subset, with near-parity elsewhere.

**vs. Global split-conformal fallback (no strata)** — On a case where easy and hard strata are mixed, the global-threshold baseline sets one quantile that over-accepts easy strata (wasting reliability headroom) and under-accepts hard strata (escalating unnecessarily), because a single quantile cannot represent heterogeneous difficulty. Our method instead assigns per-stratum quantiles via Mondrian calibration, so easy strata accept aggressively and hard strata escalate conservatively, because the stratum label captures the difficulty variation. We expect the global baseline to show error-budget violations concentrated in hard strata and excess cost in easy strata, while SROC shows flatter error rates across strata and lower total cost at the same α.

**vs. SROC with raw confidence score (no isotonic recalibration)** — On a case where a stratum's raw System-1 confidence is miscalibrated under shift (e.g., a novel skill family with out-of-vocabulary candidate sets), the raw-score baseline computes q_s over a score that is not monotone in correctness, so its conformal quantile bounds the wrong quantity and its System-1 error rate on that stratum exceeds α even with online updates. Our method instead uses the per-stratum isotonic score f_s, which is fit to empirical correctness on Any-DP_calib and therefore monotone by construction on the calibration distribution, and the monotonicity check flags strata where even f_s fails to separate correct from incorrect decisions, falling back to escalate-all. We expect the raw-score baseline to show larger coverage gaps on shifted strata and the calibrated-score SROC to hold α more closely, with the per-stratum Spearman/AUC table showing which strata triggered the UNRELIABLE fallback.

**vs. SROC without KS drift detector** — On a case where covariate shift is episodic and short-lived (shorter than the window length L), the no-detector ablation relies only on the sliding-window quantile, which lags the shift because it needs L accepted-and-wrong samples to move q_s; during the lag its error rate exceeds α. Our method instead triggers tightening by factor γ=0.9 as soon as the KS p-value between W_s and calibration scores falls below 0.01, so q_s drops before the window fills. We expect the no-detector ablation to show a transient error-budget violation at shift onset that SROC avoids, at the cost of occasional unnecessary tightening (false-positive KS) visible as a small cost increase on stationary subsets.

### What would falsify this idea

If SROC's cost savings and error-budget adherence are uniform across strata and subsets rather than concentrated where the predicted shift occurs, or if the no-drift-detector ablation matches SROC even on episodically shifted benchmarks, then the claimed shift-robustness mechanism is not doing the work and the central claim is wrong. A further falsifier is SROC matching the static-threshold AnyS1 on shifted subsets, which would mean online recalibration adds nothing beyond the original calibration; and if the per-stratum Spearman/AUC table shows the calibrated score fails the monotonicity check on most strata, the load-bearing assumption is false and the method reduces to escalate-all.

## References

1. Thinking Fast, Anywhere: One System-1 for the Decision Points of Any Agent Workflow

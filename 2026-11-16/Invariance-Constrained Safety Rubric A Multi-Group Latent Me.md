# Invariance-Constrained Safety Rubric: A Multi-Group Latent Measurement Model for Multi-Agent Alignment Diagnostics

## Motivation

The Aligning AI Teams diagnostic (arxiv:2611.17820) decomposes the single-vs-team safety gap into four findings, but its central construct — diffusion of responsibility, operationalized via proactive-safety-check rate and rationalization/deflection events — is calibrated jointly on two software-engineering tasks and ten consulting-proposal tasks without ever testing whether those rubric items measure the same construct on the same scale in both task families. The frontier mitigated the gap (safety-lead and coordinator interventions) rather than validating the instrument, so any intervention effect is confounded with task-specific rubric sensitivity: a 'gap' on sepsis triage and a 'gap' on proposal generation may be different latent quantities wearing the same aggregate score, making cross-task comparisons and pooled intervention estimates structurally uninterpretable. This is the meta-level bottleneck — the field's safety-gap claims across multi-agent work (e.g., MACPO's weak-to-strong gains, ICR's common-ground convergence) are all reported on raw aggregate behavioral rates with no measurement-invariance check, so construct validity is assumed rather than established.

## Key Insight

Diffusion of responsibility is only a task-invariant latent quantity if the rubric's factor loadings and intercepts are equal across task families, and that equality is a testable, falsifiable constraint on the measurement model rather than an assumption baked into aggregate scores.

## Method

**What it is.** InvariantScore is a multi-group confirmatory factor analysis (MG-CFA) layer that sits between raw agent-team transcripts and any safety-gap or intervention estimate. Inputs: per-episode rubric event counts from the Aligning AI Teams task families (sepsis triage, news recommender, ten consulting-proposal tasks), each episode labeled with task family g ∈ {SE, CP} and intervention arm a ∈ {none, mutual-critique, group-chat, safety-lead, coordinator-spec}. Outputs: task-invariant factor scores θ̂_i per episode, invariance test statistics, and (if invariance fails) a re-specified per-family rubric with a partial-invariance model.

**Load-bearing assumption (stated explicitly).** The model assumes that the four rubric indicators (proactive-safety-check rate, rationalization count, deflection count, attribution entropy) are conditionally independent reflections of a single continuous latent responsibility-allocation factor within each task family, so that a unidimensional MG-CFA is the correct measurement model and scalar-invariance rejection can be attributed to construct drift rather than to misspecification of the factor structure itself. This assumption is load-bearing because the entire inferential move — rejection of scalar invariance ⇒ construct drift — is valid only if the configural model is correctly specified. The source evidence (arxiv:2611.17820) describes proactive checking and responsibility deflection as two distinct behaviors, and the method's own block D concedes that consulting tasks elicit rationalization through a family-specific mechanism (client-facing hedging). A unidimensional CFA fit to indicators with family-specific generating mechanisms will reject scalar invariance for reasons of structural misspecification (cross-loadings, method factors) rather than latent-construct non-equivalence. To make this assumption testable and to separate misspecification from construct drift, we add three concrete calibration steps inside the existing MG-CFA structure: (i) a formal test of unidimensionality before imposing cross-group constraints; (ii) a bifactor re-specification if unidimensionality fails; and (iii) an opportunity-structure covariate to absorb intercept drift attributable to base-rate differences.

**How it works.**

```pseudocode
# Phase 0: unidimensionality test (new, inside existing structure)
# Fit exploratory bifactor model to pooled indicators (all families, no intervention)
# Use WLSMV estimator for ordinal/count indicators (see Phase 2)
# Criteria:
#   - parallel analysis: retain factors with eigenvalues > 95th percentile of random data
#   - bifactor fit: compare 1-factor vs 2-factor (proactive-checking vs. responsibility-deflection)
#     via ΔCFI, ΔTLI, and χ² difference (α=0.01)
#   - if 2-factor model fits better (ΔCFI > 0.01 or ΔTLI > 0.01), adopt bifactor model:
#       y_i = Λ_g * [θ_general, θ_specific] + τ_g + ε_i
#     where θ_general is the shared responsibility factor and θ_specific is a
#     family-specific method factor (e.g., client-facing hedging for CP)
#   - if 1-factor model is retained, proceed with unidimensional MG-CFA as originally planned

# Phase 1: rubric operationalization
for each episode i, task family g:
    indicators y_i = [proactive_safety_check_rate,      # continuous, [0,1] -> ordinalized into 5 bins for WLSMV
                     rationalization_event_count,       # Poisson-count -> ordinalized into 5 bins
                     deflection_event_count,            # Poisson-count -> ordinalized into 5 bins
                     responsibility_attribution_entropy] # continuous -> ordinalized into 5 bins
    # all indicators scored by two independent LLM raters; keep Krippendorff-alpha >= 0.67
    # opportunity_structure_i = count of system-level failure surfaces per episode (covariate)

# Phase 2: baseline single-group CFA (configural) with WLSMV
fit  θ_i ~ N(0,1); y_i = Λ_g θ_i + τ_g + ε_i,  ε_i ~ N(0, Ψ_g)
     with Λ_g free per family, τ_g free per family  -> configural model M0
     estimator = WLSMV (mean-and-variance adjusted weighted least squares for ordinal indicators)
     include opportunity_structure_i as a covariate predicting τ_g (intercept) to absorb base-rate differences

# Phase 3: invariance ladder (χ² difference tests, α=0.01, plus ΔCFI <= 0.01 rule)
M1: constrain Λ_SE = Λ_CP            # metric invariance
M2: constrain Λ_SE = Λ_CP and τ_SE = τ_CP   # scalar invariance
M3: constrain Ψ_SE = Ψ_CP            # strict (residual) invariance
if M2 not rejected:  use pooled θ̂ for all downstream estimates
if M2 rejected but M1 holds:  fit partial-scalar model, free the offending τ_j per family
if M1 rejected:  fall back to per-family θ̂ (configural only) and flag construct non-equivalence
# If Phase 0 adopted bifactor model, apply invariance constraints to the general factor loadings
# and intercepts; the specific factor is allowed to vary by family by design.

# Phase 4: downstream re-analysis with invariant scores
re-run safety-lead / coordinator intervention effect:
    Δθ = E[θ̂ | a=safety_lead] - E[θ̂ | a=none]   within family, then pooled
compare to raw aggregate rubric score Δ_raw from the original diagnostic
report invariance-robust vs. raw effect sizes with bootstrap CIs (B=2000)
```

**Why this design.** We chose a multi-group CFA over the original paper's raw aggregate rubric scores because the aggregate implicitly assumes scalar invariance (equal intercepts and loadings across task families) while never testing it, accepting the cost that we must collect two-rater event labels and pay a χ²-testing budget per model. We chose the ordered invariance ladder (configural → metric → scalar → strict) over a single joint test because a rejected scalar model still localizes which indicator's intercept drifts, accepting the cost of four sequential fits but gaining a diagnostic that tells us whether proactive-check rate or deflection count is the non-invariant item. We chose partial-scalar re-specification over discarding the pooled analysis entirely because it preserves cross-family comparison on the invariant indicators, accepting the cost that the resulting factor scores are only comparable on a subset of the construct. A domain expert would not describe this as a variant of standard psychometric CFA applied to a new domain, because the load-bearing move is the explicit separation of measurement invariance from intervention effect: the original Aligning AI Teams diagnostic (and MACPO, ICR) report intervention deltas on raw rates, and InvariantScore makes the invariance test a precondition for the delta to be interpretable at all — a constraint no prior multi-agent alignment paper imposes. The addition of Phase 0 (unidimensionality test) and the opportunity-structure covariate does not change this core move; it makes the load-bearing assumption testable and separates misspecification from construct drift, as required by the adversarial alert.

**Why it measures what we claim.** The factor score θ̂ measures diffusion-of-responsibility because the MG-CFA model assumes that proactive-safety-check rate, rationalization count, deflection count, and attribution entropy are conditionally independent given a single latent responsibility-allocation factor with task-family-specific loadings; this assumption fails when, e.g., consulting-proposal tasks elicit rationalization through a mechanism (client-facing hedging) that sepsis triage does not share, in which case θ̂ reflects a task-specific mixture rather than the shared construct — which is exactly what the metric-invariance test is designed to detect and flag. The scalar-invariance test measures construct equivalence because it assumes equal indicator intercepts across families reflect equal baseline event rates under no intervention; this assumption fails when the two task families have different base rates of safety-check opportunities (sepsis triage has more system-level failure surfaces than proposal generation), in which case a rejected scalar model reflects opportunity-structure differences rather than construct drift. The intervention delta Δθ measures the safety-lead/coordinator effect because it assumes random assignment of intervention arm within family; this assumption fails when the safety-lead arm is run only on the ten consulting tasks, in which case Δθ reflects a family-by-arm confound rather than the intervention's causal effect. The Phase 0 unidimensionality test and the opportunity-structure covariate directly address the first two failure modes: if the indicators are not unidimensional, the bifactor model separates the shared factor from the family-specific method factor; if base rates differ, the covariate absorbs the intercept drift before the invariance verdict is issued.

## Contribution

(1) A measurement-model layer (InvariantScore) that formalizes diffusion of responsibility as a latent variable and imposes a testable multi-group invariance ladder on the Aligning AI Teams rubric, converting an assumed construct into an identified one. (2) An empirical design principle: intervention effects on multi-agent safety gaps should be reported on task-invariant factor scores with invariance diagnostics, not raw aggregate rubric rates, because the latter conflate construct drift with treatment effect. (3) A re-specified per-task-family rubric derived from partial-invariance diagnostics, identifying which indicators (proactive-check vs. deflection vs. attribution-entropy) carry the non-invariant signal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Aligning AI Teams task families (sepsis triage, news recommender, ten consulting proposals) | Matches the source multi-agent alignment diagnostic directly |
| Primary metric | Invariance-robust intervention delta Δθ vs. raw rubric delta Δ_raw | Isolates measurement artifact from true effect |
| Baseline | Raw aggregate rubric score (original diagnostic) | Implicitly assumes scalar invariance, untested |
| Baseline | Per-family configural-only θ̂ | Discards cross-family comparability |
| Baseline | MACPO-style preference optimization | Alignment-training-transfer comparison |
| Ablation-of-ours | MG-CFA without invariance ladder (single joint fit) | Tests whether the ordered ladder is load-bearing |
| Ablation-of-ours | MG-CFA without Phase 0 unidimensionality test (force unidimensional model) | Tests whether the unidimensionality check is load-bearing |
| Ablation-of-ours | MG-CFA without opportunity-structure covariate | Tests whether base-rate adjustment is load-bearing |

### Why this setup validates the claim

The central claim is that the safety-gap and intervention deltas reported on raw aggregate rubric scores are not interpretable until measurement invariance across task families is tested, and that InvariantScore separates measurement artifact from genuine intervention effect. The raw-aggregate baseline tests the first sub-claim directly: if raw deltas are inflated or reversed relative to invariant deltas, the aggregate was measuring partly task-family opportunity structure rather than the responsibility construct. The configural-only baseline tests the second sub-claim: it shows what is lost when cross-family comparison is abandoned, so the partial-invariance path can be shown to recover comparability the configural model forfeits. MACPO tests whether the construct we isolate is the same alignment construct that alignment-training methods target, guarding against a metric that is internally clean but externally irrelevant. The no-ladder ablation tests whether the ordered configural-metric-scalar-strict sequence is doing real diagnostic work or whether a single joint test would suffice. The no-unidimensionality ablation tests whether the Phase 0 check is necessary to avoid attributing misspecification to construct drift. The no-covariate ablation tests whether opportunity-structure differences alone can explain any observed scalar rejection. The primary metric is right because it is defined as the difference between two effect estimates on the same episodes; only a metric that holds the episodes fixed while varying the measurement model can attribute a discrepancy to invariance rather than to sampling.

### Expected outcome and causal chain

**vs. Raw aggregate rubric score** — On a case where sepsis triage and consulting proposals differ in baseline safety-check opportunity rate, the raw aggregate produces a pooled intervention delta that is partly a family-mix artifact because it silently assumes equal intercepts and loadings across families. Our method instead tests scalar invariance first and, on rejection, fits a partial-scalar model freeing the offending intercept, because the ladder localizes drift to a specific indicator. We expect a noticeable divergence between Δ_raw and Δθ concentrated on the non-invariant indicator (likely proactive-check rate), with near-parity on deflection count.

**vs. Per-family configural-only θ̂** — On a case where the safety-lead arm is run only on consulting tasks, the configural-only baseline produces per-family deltas that cannot be pooled, so the intervention effect is reported as ten task-specific numbers with no cross-family statement, because it forfeits any shared metric. Our method instead retains metric invariance and pools on the invariant indicators, because partial-scalar re-specification preserves comparison on the subset that is construct-equivalent. We expect the pooled Δθ to recover a directional effect that the configural baseline leaves unresolved, with the largest gain on consulting tasks and attenuated but same-sign estimates on sepsis.

**vs. MACPO-style preference optimization** — On a case where rationalization arises through client-facing hedging in proposals but not in triage, the MACPO baseline produces an alignment improvement measured on its own preference objective that does not map onto the responsibility construct, because its supervision is weak-teacher preference data rather than rubric events. Our method instead measures the same construct across families with an explicit invariance check, because θ̂ is anchored to rubric event indicators. We expect MACPO to show flat or family-inconsistent movement on θ̂ despite its own objective improving, revealing that its alignment gain is not the same latent factor.

**vs. MG-CFA without invariance ladder (single joint fit)** — On a case where the configural model is misspecified due to family-specific residual covariance (client-facing hedging), the single joint fit produces a single rejection of all constraints without localizing the source, because it does not sequence the tests. Our method instead uses the ordered ladder to isolate whether metric, scalar, or strict invariance fails, because each step adds one constraint. We expect the ladder to attribute rejection to scalar invariance on proactive-check rate, while the joint fit only reports a global misfit.

**vs. MG-CFA without Phase 0 unidimensionality test** — On a case where the four indicators are not unidimensional (two-factor structure: proactive checking vs. responsibility deflection), the no-Phase-0 baseline forces a unidimensional model and rejects scalar invariance for reasons of structural misspecification, because it ignores the two-factor structure. Our method instead tests unidimensionality first and adopts a bifactor model if needed, because Phase 0 compares 1-factor vs. 2-factor fits. We expect the no-Phase-0 baseline to show a spurious scalar rejection that disappears when the bifactor model is used, with the general factor showing invariance while the specific factor varies by family.

**vs. MG-CFA without opportunity-structure covariate** — On a case where sepsis triage has more system-level failure surfaces than proposal generation, the no-covariate baseline attributes intercept drift to construct non-equivalence, because it does not adjust for base-rate differences. Our method instead includes opportunity_structure_i as a covariate predicting intercepts, because it absorbs base-rate differences. We expect the no-covariate baseline to reject scalar invariance on proactive-check rate, while the covariate model shows that the rejection is explained by opportunity structure, leaving the construct equivalence intact.

### What would falsify this idea

If scalar invariance is never rejected across the task families, then the raw aggregate was already interpretable and the invariance layer adds nothing — the central claim collapses. Equally fatal: if the invariance-robust and raw deltas agree within noise on every indicator, the predicted measurement-artifact pathway does not exist and the separation of measurement from effect is unmotivated. Additionally, if Phase 0 shows that the indicators are not unidimensional and the bifactor model's general factor also fails scalar invariance, then the construct itself is not equivalent across families and the method's diagnostic value is limited to flagging non-equivalence rather than enabling pooled comparison.

## References

1. Aligning AI Teams

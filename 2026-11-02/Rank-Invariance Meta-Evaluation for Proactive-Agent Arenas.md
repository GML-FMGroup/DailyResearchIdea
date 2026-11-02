# Rank-Invariance Meta-Evaluation for Proactive-Agent Arenas

## Motivation

Arena-style benchmarks such as HALO report capability rankings from a fixed panel of models (HALO-8K uses 10 models, with best proprietary at 72.88% and best open-source at 56.63% acceptability), yet the field treats these rankings as if they were model-independent properties of the benchmark. HALO in particular is designed around a single frontier model family and its gpt-oss-20b documentation assumption, so its reported difficulty ordering is never checked for invariance across capability tiers or domains. The structural root cause is that benchmark validity is asserted via one model family's documentation rather than measured as an invariance property of the arena itself, which means any downstream claim that 'model A is better than model B on proactive assistance' silently depends on an untested panel-composition assumption.

## Key Insight

If a benchmark's difficulty calibration is a genuine property of its scenarios rather than an artifact of which models happen to be in the panel, then the per-scenario difficulty ordering must be rank-invariant across capability tiers and domains; measuring the rank correlation of item difficulty across tiers turns that invariance into a falsifiable, reportable validity statistic instead of a documentation assumption.

## Method

**What it is.** RIM (Rank-Invariance Meta-evaluation) is a protocol that takes an existing agent arena (HALO-Arena's 400 scenarios, 4 need types, 10 domains, 50 personal environments) plus a capability-tiered model panel, and outputs a per-scenario difficulty vector, a cross-tier rank-correlation matrix, a per-scenario panel-sensitivity score, an additivity/DIF diagnostic, and a calibrated validity verdict. It does not score agents; it scores the benchmark.

**Load-bearing assumption (stated explicitly).** RIM's centering step assumes *additive separability*: that per-model difficulty decomposes as d(m,s) = alpha_m + beta_s + epsilon(m,s), i.e., a model-level capability offset alpha_m plus a tier-invariant scenario ordering beta_s, with no capability-by-scenario interaction. If this holds, subtracting the per-tier mean removes alpha_m and leaves beta_s comparable across tiers. This assumption is exactly the one IRT/DIF research shows routinely fails (Holland & Wainer, 1993; Swaminathan & Rogers, 1990): bounded [0,1] acceptability produces ceiling/floor saturation, so low-tier models saturate at 0 on hard scenarios and high-tier models saturate at 1 on easy ones, compressing or tie-dominating the residual ordering in the very strata being compared. HALO's own spread (72.88% vs 56.63% acceptability) implies such saturation, and its over-proactivity finding is a direct statement of tier-by-scenario interaction. RIM therefore does not merely assume additivity; it tests it (Phase 2b) and reports the interaction magnitude as a primary quantity, treating the centered Spearman as a derived descriptive.

**How it works.**

```pseudocode
INPUT: arena S (400 scenarios), panel M = {m_1..m_K} partitioned into T=3 capability tiers
       (tier assignment fixed by an external capability index, e.g. published MMLU/agentic
       composite; tiers are NOT inferred from arena performance)
OUTPUT: difficulty vector d per scenario, cross-tier Spearman matrix R, per-scenario
        panel-sensitivity score Psi, additivity/DIF diagnostic A, validity verdict V

PHASE 1 - Per-scenario difficulty estimation (per model, no cross-model pooling):
  for each m in M:
    for each s in S:
      run m on s -> acceptability a(m,s) in {0,1} (HALO expert rubric, 3 annotators,
        majority vote; ties -> reject)
      run m on s R=5 times at temperature 0.7 -> a_rep(m,s) in [0,1]
      d(m,s) = 1 - mean_r a_rep(m,s)          # per-model difficulty in [0,1]

PHASE 2 - Tier aggregation with capability-stratified centering:
  for each tier t in {1,2,3}:
    d_t(s) = mean_{m in tier t} d(m,s)
    center: d~_t(s) = d_t(s) - mean_s' d_t(s')   # remove tier-level offset
  # centering is what makes cross-tier comparison legitimate ONLY IF the additive
  # separability assumption holds; Phase 2b tests that assumption directly

PHASE 2b - Additivity / DIF diagnostic (tests the load-bearing assumption):
  # (i) Two-way additivity test on the per-model difficulty matrix D[K x 400]:
  fit D[m,s] = alpha_m + beta_s + epsilon[m,s] by least squares (K models x 400 scenarios)
  compute Tukey additivity statistic: SS_interaction / SS_error, with df=(K-1)(400-1)
  report interaction variance fraction V_int = SS_interaction / SS_total
  # (ii) Per-scenario DIF test across tiers:
  for each scenario s:
    fit ordinal cumulative-link model on the 5 repeated binary outcomes per model
      with tier as a factor: logit P(a=1) = theta_m - beta_s - gamma_{t(s),s}
    test H0: gamma_{t,s} = 0 for all t (likelihood-ratio test, df=T-1=2)
    record p_s and interaction magnitude |gamma_{t,s}|_max
  A = { V_int, fraction of scenarios with p_s < 0.05 (DIF rate), median |gamma|_max }
  # if V_int is large or DIF rate is high, the centering premise is violated and
  # the centered Spearman is reported as a descriptive, not as the validity verdict

PHASE 3 - Rank-invariance statistics (derived descriptive):
  for each tier pair (t,u):
    R[t,u] = Spearman(d~_t, d~_u)               # global rank invariance
  for each domain D in the 10 domains:
    R_D[t,u] = Spearman(d~_t restricted to D, d~_u restricted to D)
  for each need type N in the 4 types:
    R_N[t,u] = Spearman over N
  bootstrap B=1000 resamples over scenarios -> 95% CI on every R
  # per-scenario panel-sensitivity score:
  for each scenario s:
    Psi(s) = variance over tiers t of d~_t(s)   # high Psi = tier-sensitive scenario

PHASE 4 - Validity verdict:
  V = PASS if (A.V_int <= 0.10 AND A.DIF_rate <= 0.10) AND
              min over all (t,u),D,N of lower-CI(R) >= tau_rank (tau_rank = 0.70)
      else FAIL, and report the (tier-pair, domain) cells that break first
      and whether the failure is attributable to additivity violation (Phase 2b)
  also report difficulty-calibration drift: Delta_cal = max_{t,u} |mean_s d~_t - mean_s d~_u|
  also report Psi distribution and list the top-20 highest-Psi scenarios for downstream filtering
```

**Why this design.** We chose a fixed external capability index to assign tiers over inferring tiers from arena performance, because inferring tiers from the same data being validated would make rank invariance circular; the cost is that tier assignment inherits whatever noise the external index has, which we accept because the external index is independent of HALO. We chose per-model repeated sampling (R=5) with majority-vote acceptability over a single annotation pass, because single-pass difficulty estimates are dominated by annotation noise and would deflate cross-tier correlations; the cost is 5x inference and 3x annotation budget on 400 scenarios, which is tractable at HALO's scale. We chose capability-stratified centering over raw correlation, because raw per-tier difficulty vectors share a large common offset (low-tier models are uniformly worse) that inflates Spearman; centering removes that main effect and forces the statistic to measure ordering, not level. We chose a per-domain and per-need-type breakdown over a single global correlation, because HALO's own finding that over-proactivity is the dominant failure mode predicts that difficulty ordering may be preserved globally but break inside specific domains; the cost is more hypothesis tests, which we handle with bootstrap CIs rather than p-values. We chose to add Phase 2b (two-way additivity test and per-scenario DIF test) because the centering step is only valid under additive separability, and IRT/DIF research shows that assumption routinely fails under ceiling/floor saturation; the cost is an extra model fit per scenario, which is tractable at 400 scenarios. A domain expert would not describe RIM as a variant of standard benchmark-agreement work (e.g., Cohen's-kappa inter-annotator studies or HELM-style multi-metric aggregation) because those measure agreement among raters or aggregate scores across metrics, whereas RIM measures whether the difficulty *ordering* of items is invariant across independently-assigned capability strata — a property of the arena, not of the raters or the aggregate — and it explicitly tests the additive separability that makes that ordering comparable.

**Why it measures what we claim.** The centered Spearman R[t,u] measures scenario-intrinsic difficulty ordering comparability across tiers because it assumes additive separability (d(m,s) = alpha_m + beta_s + epsilon); this assumption fails when capability-by-scenario interaction is present (ceiling/floor saturation, shortcut exploitation by high-capability models), in which case R reflects the interaction structure rather than tier-invariant scenario difficulty. The per-scenario panel-sensitivity score Psi(s) measures how much a scenario's centered difficulty varies across tiers because it assumes that under additivity Psi(s) is sampling noise around zero; this assumption fails when a scenario exhibits DIF, in which case Psi reflects genuine tier-specific difficulty rather than noise. The additivity diagnostic A (V_int and DIF rate) measures the magnitude of the interaction term because it assumes the two-way model captures the dominant structure; this assumption fails when higher-order interactions dominate, in which case A understates the violation. The validity verdict V measures whether the arena's difficulty ordering is a tier-invariant property because it requires both additivity (A) and rank invariance (R) to pass; this assumption fails when either component is violated, in which case V correctly returns FAIL and attributes the failure.

## Contribution

['RIM, a meta-evaluation protocol that converts benchmark validity into a measurable rank-invariance statistic over capability tiers, need types, and domains, with bootstrap confidence intervals and an explicit PASS/FAIL verdict rule.', 'The design principle that capability-stratified centering is required before cross-tier rank correlation, because uncentered per-tier difficulty vectors are dominated by a capability main effect that inflates apparent invariance.', "An open protocol and reference implementation applicable to any arena-style agent benchmark that reports per-item acceptability, demonstrated on HALO-Arena's 400 scenarios and 10-model panel."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HALO-Arena (400 scenarios, 10 domains, 4 need types) | Native granularity for per-domain/per-need breakdown |
| Primary metric | Validity verdict V = PASS iff (V_int ≤ 0.10 AND DIF rate ≤ 0.10) AND min lower-CI cross-tier Spearman ≥ 0.70 | Jointly tests additivity and rank invariance |
| Secondary metric | Additivity diagnostic A: interaction variance fraction V_int and DIF rate (fraction of scenarios with p_s < 0.05) | Directly tests the load-bearing additive-separability assumption |
| Secondary metric | Per-scenario panel-sensitivity score Psi(s) = variance over tiers of centered difficulty | Enables downstream filtering of tier-sensitive scenarios |
| Baseline | Raw Spearman (no centering) | Tests whether centering is doing the work |
| Baseline | Tier-inferred-from-arena grouping | Tests circularity of tier assignment |
| Baseline | Single-pass difficulty (R=1, 1 annotator) | Tests whether noise deflation is the real driver |
| Ablation-of-ours | RIM without per-domain/need breakdown | Tests whether global correlation masks localized breaks |
| Ablation-of-ours | RIM without Phase 2b (additivity diagnostic omitted) | Tests whether the additivity test changes the verdict |

### Why this setup validates the claim

The central claim is that HALO-Arena's difficulty *ordering* is invariant across independently assigned capability strata, and that this invariance is a property of the arena rather than an artifact of annotation noise, tier circularity, level effects, or unmodeled capability-by-scenario interaction. Each baseline isolates one threat to that claim. The raw-Spearman baseline tests whether capability-stratified centering is essential: if raw correlation already passes τ_rank, the centering mechanism is unnecessary and the claim collapses into trivial level agreement. The tier-inferred baseline tests circularity: if tiers inferred from arena performance produce high rank invariance while external-index tiers do not, the invariance is self-fulfilling. The single-pass baseline tests noise deflation: if R=1 with one annotator already yields high invariance, the repeated-sampling machinery earns nothing. The ablation without per-domain/need breakdown tests localization: a global PASS with per-domain FAILs would mean the verdict is masking exactly the breakdown HALO's over-proactivity finding predicts. The ablation without Phase 2b tests whether the additivity diagnostic is load-bearing: if omitting it changes PASS to FAIL or vice versa, the additive-separability assumption is doing real work and must be reported. The primary metric — a joint verdict requiring both additivity (V_int ≤ 0.10, DIF rate ≤ 0.10) and minimum lower-CI Spearman ≥ 0.70 across all tier-pair/domain/need cells — is chosen because it is the strictest falsifiable form of the claim: it fails if *any* stratum breaks or if the additive model is violated, and the bootstrap CI makes the failure attributable to sampling rather than point-estimate luck.

### Expected outcome and causal chain

**vs. Raw Spearman (no centering)** — On a case where low-tier models fail uniformly across easy and hard scenarios, the raw baseline produces inflated cross-tier correlation because the shared capability offset dominates the covariance. Our method instead subtracts the per-tier mean before correlating because the statistic must measure ordering, not level, so we expect raw Spearman to sit visibly above the centered estimate on the global aggregate but to converge with ours inside domains where tier offsets are small — a gap concentrated exactly where level effects are largest. Additionally, the raw baseline does not test additivity, so it cannot distinguish level agreement from ordering agreement; our Phase 2b diagnostic will report V_int and DIF rate, which the raw baseline ignores.

**vs. Tier-inferred-from-arena grouping** — On a case where arena performance itself defines the tier split, the baseline produces near-perfect rank invariance because the grouping is a function of the same difficulty vector being validated. Our method instead uses an external capability index fixed before HALO is touched because validation must be independent of the object validated, so we expect the tier-inferred baseline to pass τ_rank almost by construction while the external-index verdict is the only one that can fail — and if the two diverge, the divergence is the signal that circularity was doing the work. The tier-inferred baseline will also show artificially low DIF because the grouping absorbs interaction, so our Phase 2b diagnostic provides a check that the external-index tiers do not.

**vs. Single-pass difficulty (R=1, 1 annotator)** — On a borderline scenario where one annotator's majority vote flips on a single ambiguous action, the single-pass baseline produces a difficulty estimate dominated by annotation noise, deflating cross-tier Spearman toward zero. Our method instead averages R=5 repeated samples with 3-annotator majority vote because the difficulty signal must exceed the annotation noise floor, so we expect the single-pass baseline to under-correlate by a noticeable margin on the noisiest domains (those with subjective acceptability) while matching ours on domains with crisp rubric outcomes. The single-pass baseline will also inflate the DIF rate because annotation noise masquerades as interaction, so our Phase 2b diagnostic on the full R=5 data provides a cleaner test of additivity.

**vs. RIM without per-domain/need breakdown** — On a case where global rank invariance is high but a specific domain (e.g., over-proactivity-heavy domains) shows low invariance, the ablation without breakdown reports a global PASS and misses the localized break. Our method instead computes R_D and R_N for every domain and need type because the verdict must fail if any stratum breaks, so we expect the ablation to pass while ours fails, with the failure concentrated in the domains where over-proactivity is the dominant failure mode — exactly the structural property HALO's own findings predict.

**vs. RIM without Phase 2b (additivity diagnostic omitted)** — On a case where ceiling/floor saturation produces large interaction variance (V_int > 0.10) but centered Spearman still exceeds 0.70, the ablation without Phase 2b reports PASS while our full method reports FAIL because the additive-separability assumption is violated. We expect this divergence to occur in the domains with the widest capability spread (where saturation is most severe), demonstrating that the additivity diagnostic is load-bearing and that centered Spearman alone can mask interaction-driven ordering.

### What would falsify this idea

If the per-domain and per-need breakdowns show rank invariance that is *uniformly* high or uniformly low across all strata — rather than concentrated breaks in the domains where over-proactivity is the dominant failure mode — then the localization mechanism is not detecting a real structural property and the central claim that HALO's difficulty ordering is stratum-invariant is wrong. Equivalently, if the tier-inferred baseline and the external-index verdict agree on every cell, the independence argument is doing no work and RIM reduces to ordinary agreement measurement; and if the additivity diagnostic (Phase 2b) shows V_int ≤ 0.10 and DIF rate ≤ 0.10 uniformly across all domains, then the load-bearing additive-separability assumption holds and the centering step is justified, but if it fails in the same domains where rank invariance breaks, the failure is attributable to interaction rather than to noise or circularity.

## References

1. HALO: Human-Grounded Granular Evaluation of Proactive Agents

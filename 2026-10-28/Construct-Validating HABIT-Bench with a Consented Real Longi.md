# Construct-Validating HABIT-Bench with a Consented Real Longitudinal Habit Corpus and Cross-Domain Transfer Calibration

## Motivation

HABIT-Bench and other agent-memory benchmarks (e.g., MemoryAgentBench, LongMemEval) establish habit/memory competence almost entirely on synthetic or curated histories, so no existing work can tell whether a memory method that scores well on controlled habit graphs would also induce habits from real longitudinal user behavior; HABIT-Bench's own best macro accuracy of 33.61% is reported without any evidence that this number tracks real-world habit induction. The root cause is structural: synthetic habit graphs encode habits as clean, low-noise, fully-observable latent variables, whereas real user logs contain unlabeled, drifting, and partially-observed routines, so a benchmark score conflates 'model can read a controlled graph' with 'model can induce a habit' — and no prior benchmark measures the mapping between the two.

## Key Insight

A synthetic habit benchmark is construct-valid exactly when the rank-ordering of memory methods it induces is preserved on real consented longitudinal logs, and this rank agreement can be estimated by anchoring both datasets to a shared set of method-agnostic, graph-identifiable habit probes whose difficulty is matched by an explicit noise-and-sparsity intervention rather than by surface similarity.

## Method

**What it is.** We introduce HABIT-VALID, a validation protocol that pairs HABIT-Bench's synthetic habit graphs with a small consented real longitudinal corpus (HABIT-Real) and reports a *rank-transfer coefficient* between the two, computed over memory methods and over difficulty-matched probe strata. Inputs: HABIT-Bench synthetic histories + graphs, HABIT-Real interaction logs (with per-user consent and anonymization), and a set of memory methods under test. Output: per-method synthetic vs. real macro accuracy, a rank-transfer coefficient with bootstrap CI, and a per-stratum breakdown showing where transfer breaks.

**Load-bearing assumption (stated explicitly).** HABIT-VALID assumes that within a difficulty stratum, synthetic and real probes are exchangeable with respect to the habit-induction competence being measured, i.e., that the difficulty descriptor d is a sufficient statistic for probe difficulty. This is a strong ignorability / no-unmeasured-confounding claim. It is structurally at risk because real logs are unlabeled, drifting, and partially observed while synthetic graphs are clean, low-noise, and fully observable, so the corpora differ on axes (label noise, drift, partial observability, domain vocabulary, templated phrasing) not in d. We therefore do not treat exchangeability as a premise: we test it via the difficulty-invariance calibration below before computing RTC, and if the scales fail to link we report that as the result.

**How it works.**

```pseudocode
# Phase 0: pilot axis selection (n≈8 users, 3 weeks)
for probe p in pilot_probes:
    err[p] = 1 - acc(panel_method, p)          # panel = 5 held-out memory methods
for axis a in {evidence_count, session_gap, exception_rate, distractor_overlap, vocabulary, drift}:
    R2[a] = OLS(err ~ a + user_fixed_effect).R2  # per-axis variance explained
keep_axes = {a : R2[a] >= 0.05}                  # drop negligible axes
report residual_latent_variance = 1 - sum(R2[keep_axes])

# Phase 1: HABIT-Real construction (consented, n~40 users, 6-10 weeks)
for user u in consented_users:
    logs_u = anonymize(interaction_logs(u))          # strip PII, keep timestamps+intent labels
    habits_u = two_annotator_induction(logs_u)        # independent labelers write habit graph
    habits_u = adjudicate(habits_u)                   # disagreements resolved by 3rd labeler
    probes_u = instantiate_probes(habits_u, n=20)     # 4-choice, same schema as HABIT-Bench

# Phase 2: empirical difficulty calibration (replaces fixed 4-axis binning)
# Fit 2PL IRT item-difficulty/discrimination jointly across corpora on held-out panel
for probe p in HABIT-Bench ∪ HABIT-Real:
    (b_p, a_p) = fit_2PL(responses[panel_methods, p])   # b=difficulty, a=discrimination
# Differential-item-functioning (DIF) test per kept axis: does b_p depend on corpus?
for axis a in keep_axes:
    DIF[a] = likelihood_ratio_test(b_p ~ corpus + a + corpus:a)  # p-value + effect size
if any DIF[a] significant (p<0.05, |delta_b|>0.3):
    report scale_link_failure = True; return {DIF, scale_link_failure}  # result, not RTC
# If scales link, define strata on the linked empirical difficulty scale
for probe p:
    p.bin = quantize(b_p, n_bins=4)                  # 4 bins on linked scale (coarser than 192)

# Phase 3: run all memory methods M on both corpora
for m in M:
    for p in probes:
        acc[m][p.corpus][p.bin] = evaluate(m, p)

# Phase 4: rank-transfer estimation
for bin b with >=k probes in both corpora (k=8):
    rho[b] = spearman(rank_M acc[m][synth][b], rank_M acc[m][real][b])
RTC = weighted_mean(rho, weights=n_probes_real[b])
CI  = bootstrap_users(RTC, B=2000)                    # resample users, not probes

# Phase 5: calibration check with control methods
# Inject two control methods into M:
#   format-artifact control: matches templated dialogue phrasing; predicted RTC ≈ 0
#   competence control: induces habit graph from evidence; predicted RTC high
# Report RTC for each control; if format-artifact control RTC > 0.3, flag instrument failure

# Phase 6: report + diagnose
return {RTC, CI, per_stratum_rho, DIF_results, control_RTCs, worst_methods_by_transfer_gap}
```

**Why this design.** We chose a *rank-transfer* statistic over absolute-accuracy comparison because the field's question is whether method A beats method B, not whether the synthetic number equals the real number — accepting the cost that RTC is blind to a uniform bias shift affecting all methods equally. We chose difficulty-matched strata over global correlation because synthetic and real probes differ systematically in evidence sparsity and exception rate; without matching, any rank correlation is confounded by probe difficulty, accepting the cost of discarding probes that fall in sparse strata (k<8). We chose user-level bootstrap over probe-level bootstrap because probes within a user are correlated through the same underlying habit graph, accepting wider CIs in exchange for honest uncertainty. We chose human-adjudicated real habit graphs over LLM-inferred habits as ground truth because using the same class of model to label and to be evaluated would make the validation circular, accepting the annotation cost. We chose empirical IRT difficulty calibration with a DIF test over a fixed four-axis grid because the fixed grid assumes d is a sufficient statistic — an untested ignorability claim — whereas the DIF test makes exchangeability a tested hypothesis and reports scale-link failure as a result; we accept the cost of a pilot study and a coarser 4-bin design to keep k≥8 achievable. This is not a variant of LongMemEval or MemoryAgentBench: those benchmarks *report* scores on curated histories, whereas HABIT-VALID measures the *mapping* between two corpora via a difficulty-conditioned rank statistic — a quantity no prior memory benchmark computes.

**Why this measures what we claim.** The rank-transfer coefficient RTC measures *construct validity of HABIT-Bench* because it assumes that if synthetic probes and real probes in the same stratum tap the same underlying habit-induction competence, then method rankings should agree within that stratum; this assumption fails when a memory method exploits a synthetic-specific artifact (e.g., templated dialogue phrasing), in which case RTC reflects *format transfer* rather than *competence transfer* and the per-stratum rho will be near zero in high-distractor strata while high in low-distractor strata. The empirical IRT difficulty b_p measures *probe difficulty* because it assumes the held-out panel's response pattern is driven by a unidimensional difficulty latent; this fails when a method's errors are driven by a latent axis not captured by the panel (e.g., domain vocabulary), in which case the DIF test flags scale-link failure and we report that instead of RTC. The two-annotator adjudicated habit graph measures *the user's true habit* because it assumes independent labelers converge on the same latent routine; this fails when a user's behavior is genuinely ambiguous, in which case the probe tests the annotators' consensus rather than the user's habit — we flag such probes via inter-annotator disagreement and exclude them from RTC.

## Contribution

(1) HABIT-VALID, a validation protocol that computes a difficulty-stratified rank-transfer coefficient between a synthetic habit benchmark and a consented real longitudinal corpus, with user-level bootstrap uncertainty. (2) A concrete empirical design principle: that construct validity of an agent-memory habit benchmark is a per-stratum property, so aggregate benchmark scores can be valid in easy strata and invalid in hard ones — a finding that follows from the stratified RTC and is invisible to single-number benchmark reporting. (3) HABIT-Real, a small consented and anonymized longitudinal habit corpus with adjudicated habit graphs and schema-compatible four-choice probes, released for future benchmark validation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | HABIT-Bench + HABIT-Real (n≈40, 6–10 weeks; pilot n≈8, 3 weeks) | Synthetic scale plus consented real longitudinal probes; pilot powers axis selection. |
| Primary metric | Rank-transfer coefficient (RTC) with user-bootstrap CI, conditioned on passing DIF scale-link test | Tests method-ranking agreement across corpora on a linked difficulty scale, not absolute scores. |
| Baseline 1 | Best HABIT-Bench method (33.61% macro acc) | Tests whether top synthetic rank transfers to real. |
| Baseline 2 | LongMemEval-style long-context LLM | Tests whether curated-history competence transfers to habit induction. |
| Baseline 3 | LLM-inferred habit graphs as ground truth | Tests whether human-adjudicated labels are necessary for non-circular validation. |
| Ablation-of-ours | Global correlation (no difficulty strata) | Isolates the effect of difficulty matching on RTC. |
| Control method A | Format-artifact control (matches templated dialogue phrasing) | Predicted RTC ≈ 0; calibration check that RTC separates format from competence. |
| Control method B | Competence control (induces habit graph from evidence) | Predicted high RTC; calibration check that RTC detects competence transfer. |

### Why this setup validates the claim

The central claim is that HABIT-Bench's synthetic habit graphs have construct validity for habit-induction competence, meaning method rankings on synthetic probes should agree with rankings on real consented probes *within difficulty-matched strata on a scale that has passed a DIF link test*. The best HABIT-Bench method baseline tests the sub-claim that synthetic rank is meaningful: if it also ranks high on real probes within strata, RTC is high; if it collapses, RTC is near zero. The LongMemEval-style baseline tests the sub-claim that this is about habit induction, not generic long-context recall — it should show a different transfer profile across strata. The LLM-inferred-graph baseline tests the circularity sub-claim: if RTC is similar when ground truth is LLM-inferred, human adjudication is unnecessary. The global-correlation ablation isolates the stratum-matching mechanism: if RTC without strata is much lower or noisier, difficulty matching is doing the work. The two control methods make the instrument's validity testable: a format-artifact control with predicted RTC ≈ 0 and a competence control with predicted high RTC must be separated by RTC for the statistic to be credible. The pilot axis selection and DIF test make the exchangeability assumption a tested hypothesis rather than a premise. RTC is the right metric because the field's question is comparative (does method A beat B?), not absolute accuracy, and RTC directly operationalizes rank agreement while being robust to uniform bias shifts. The user-level bootstrap correctly accounts for within-user probe correlation, so the CI is honest about uncertainty in the transfer estimate.

### Expected outcome and causal chain

**vs. Best HABIT-Bench method (33.61% macro acc)** — On a case where a synthetic probe has many distractor sessions with templated phrasing, the baseline's memory mechanism may latch onto surface dialogue templates and rank high on synthetic but fail on real probes where phrasing varies, because its mechanism gap is format-specific pattern matching rather than habit induction. Our method instead conditions rank comparison within empirically calibrated difficulty strata, so a method that exploits synthetic-specific artifacts will show low per-stratum rho in high-distractor strata while high in low-distractor strata, producing a moderate overall RTC (e.g., 0.4–0.6) rather than near 1.0. We expect a noticeable gap on high-distractor strata but parity on low-distractor strata.

**vs. LongMemEval-style long-context LLM** — On a case where a real user's habit requires aggregating weak evidence across many sessions with an exception, the baseline produces wrong answers because its long-context recall mechanism retrieves salient recent sessions rather than inducing a stable habit graph, so its ranking on real probes diverges sharply from synthetic probes where salient evidence is denser. Our method instead stratifies by the linked empirical difficulty scale, exposing that the baseline's rank collapses specifically in sparse-evidence, long-gap strata. We expect low per-stratum rho concentrated in sparse-evidence strata and near-zero contribution to RTC from dense-evidence strata, showing the transfer failure is mechanism-specific.

**vs. LLM-inferred habit graphs as ground truth** — On a case where a user's behavior is genuinely ambiguous, the LLM-inferred graph produces a confident but arbitrary habit label, so probes built from it test the LLM's prior rather than the user's routine, inflating apparent transfer for methods that share the LLM's inductive biases. Our method instead uses two-annotator adjudication and flags high-disagreement probes for exclusion, so RTC computed on human-adjudicated graphs should be lower than RTC computed on LLM-inferred graphs for methods that share LLM biases. We expect a systematic RTC inflation of roughly 0.15–0.3 when using LLM-inferred graphs, concentrated in ambiguous-habit strata.

**vs. Format-artifact control (predicted RTC ≈ 0)** — On a case where the control matches templated dialogue phrasing, it should rank high on synthetic probes but near-chance on real probes, so its per-stratum rho should be near zero across strata. If the control's RTC exceeds 0.3, the instrument is not separating format from competence and the construct-validity claim is unsupported.

**vs. Competence control (predicted high RTC)** — On a case where the control induces a habit graph from evidence, it should rank consistently across synthetic and real probes within linked strata, so its RTC should be high (e.g., >0.6). If the competence control's RTC is near zero, the instrument is not detecting competence transfer and the central claim fails.

### What would falsify this idea

If RTC is uniformly high (e.g., >0.8) across all strata even for the format-artifact control, or if the global-correlation ablation yields RTC indistinguishable from the stratified version, then difficulty matching is not doing the work and the claimed construct validity is unsupported. If the DIF test fails to link the synthetic and real difficulty scales, the exchangeability assumption is rejected and RTC is not computed — that scale-link failure is itself the reported result.

## References

1. HABIT-Bench: Evaluating Habit Induction from Longitudinal Weak Evidence in Agent Memory
2. Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions
3. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory

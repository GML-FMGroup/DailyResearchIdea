# Capability-Gap-Conditioned Validation Certificates for LLM Judges at the Evaluation Frontier

## Motivation

AltRA-Test validates an LLM judge by testing non-inferiority of mixed human-LLM panel agreement against an all-human panel, but this guarantee holds only for the fixed judge and fixed validation corpus observed at validation time; it says nothing about whether the same judge remains valid for models released later that may exceed the judge's own accuracy. The frontier paper (venue:iclr:2025:ea7fd6842187) shows that when the judge is no more accurate than the evaluated model, no debiasing method can reduce ground-truth label requirements by more than half, so a validation certificate that ignores the judge-evaluated capability gap can silently overstate judge validity exactly where reliable evaluation matters most. Both branches share the same unresolved structural assumption — a single validation corpus with a fixed judge — which makes the guarantee non-transferable across the capability gap.

## Key Insight

The non-inferiority guarantee of AltRA-Test is not a scalar certificate but a function of the judge-evaluated capability gap, and this function can be anchored at the frontier using the twice-the-data bound as a worst-case calibration point, so the certificate degrades monotonically and predictably as the evaluated model surpasses the judge rather than failing abruptly.

## Method

**What it is.** GapCert is a validation certificate for an LLM judge that outputs, for any target evaluated model M_e, an interval [τ_lo(g), τ_hi(g)] on the non-inferiority margin τ at capability gap g = acc(M_e) − acc(J), where acc is measured on a small held-out capability probe. Inputs: paired annotations from AltRA-Test's bootstrap panel construction, a capability probe set, and a frontier anchor from the twice-the-data bound. Output: a gap-conditioned certificate stating the maximum admissible capability gap at which the judge's mixed-panel IAA remains non-inferior to the all-human panel IAA at level α.

**Load-bearing assumption (stated explicitly).** GapCert assumes that the judge's accuracy on a small, disjoint capability probe is a sufficient statistic for how its ranking/non-inferiority reliability degrades as the evaluated model's capability gap grows, so that a single scalar g_e = acc(M_e) − acc(J) measured on the probe can index a margin schedule τ(g) anchored at the frontier twice-the-data bound. This assumption is load-bearing and is not verified by the probe alone; the calibration procedure below is designed to test it directly on the validation corpus and to fall back to a conservative lower envelope when it fails.

**How it works.**

```pseudocode
# Phase 1: AltRA-Test panel construction (inherited)
for b in 1..B:                       # B = 1000 bootstrap replicates
    items_b = resample(items, n)
    human_panel_b = draw_k_annotations(items_b, k=3)
    mixed_panel_b = replace_one_annotation(human_panel_b, judge=J)
    IAA_human[b] = agreement(human_panel_b)   # Krippendorff alpha
    IAA_mixed[b] = agreement(mixed_panel_b)

# Phase 2: Capability-gap probe (new)
probe = stratified_sample(probe_pool, n_probe=200)   # disjoint from validation corpus
acc_J  = accuracy(J, probe)
for each candidate evaluated model M_e in frontier_suite:
    acc_e = accuracy(M_e, probe)
    g_e   = acc_e - acc_J                            # signed capability gap

# Phase 2b: Per-regime, per-bias calibration on the validation corpus (new)
# Stratify the validation corpus by judge-strength strata using reference models
# already available (weaker/stronger than J), not by probe gap alone.
strata = {s_weak, s_match, s_strong}   # reference models bracketing J
for s in strata:
    IAA_mixed_s[b] = agreement(mixed_panel_b restricted to items in s)
    IAA_human_s[b] = agreement(human_panel_b restricted to items in s)
    delta_s[b] = IAA_mixed_s[b] - IAA_human_s[b]
    lower_s = quantile(delta_s[b], alpha)          # one-sided lower bound per stratum
# Fit IAA-vs-gap relationship empirically on these strata (not on probe gap):
#   delta_s ~ f(g_s) with g_s = acc(ref_s, probe) - acc_J, monotone-constrained
# Report fitted g* with 95% bootstrap CI and sensitivity to functional form
# (linear vs. monotone spline).

# Phase 3: Gap-conditioned non-inferiority test (new)
# AltRA null: IAA_mixed >= IAA_human - tau  (non-inferiority at margin tau)
# GapCert null: IAA_mixed(g) >= IAA_human - tau(g)
# where tau(g) = tau_0 * max(0.5, 1 - g / g_star) is the frontier-aware margin
# tau_0 = baseline non-inferiority margin (0.05)
# g_star = gap at which twice-the-data bound saturates (calibrated below)

g_star = calibrate_frontier_anchor(J, probe, frontier_suite)
   # g_star solves: required_labels(g) = 2 * required_labels(0)
   # operationalized by fitting label-savings vs g from the frontier paper's
   # reported empirical curve, anchored at the theoretical twice-the-data limit
   # AND cross-checked against the empirical f(g_s) fit from Phase 2b;
   # if the two disagree, g_star is set to the more conservative (smaller) value
   # and the certificate uses the lower envelope over strata.

for each g_e:
    tau_e = tau_0 * max(0.5, 1 - g_e / g_star)
    delta_b = IAA_mixed[b] - IAA_human[b]       # per-bootstrap difference
    lower_b = quantile(delta_b, alpha)          # one-sided lower bound
    # Conservative lower envelope over strata (repair for load-bearing assumption):
    lower_env = min(lower_b, min_s lower_s)
    certified[e] = (lower_env >= -tau_e)        # non-inferiority at gap g_e

# Phase 4: Certificate output
return {g_star, tau(g), certified[e], max_certified_gap}
   # max_certified_gap = sup{ g : certified(g) = True }
```

**Why this design.** We chose to condition the non-inferiority margin on the capability gap rather than re-running AltRA-Test per evaluated model because re-running requires fresh human annotations for every new model, which defeats the purpose of a reusable certificate; the cost is that the certificate is only as good as the capability probe's estimate of acc(M_e). We chose a piecewise-linear margin schedule τ(g) = τ_0 · max(0.5, 1 − g/g*) over a learned regression of IAA on gap because the frontier paper's twice-the-data bound supplies a principled worst-case anchor at g*, giving a schedule whose endpoints are externally justified rather than fit; the cost is that intermediate gaps may be conservative. We chose Krippendorff alpha as the IAA statistic over Cohen's kappa because AltRA-Test's panel construction with k=3 annotators and one replacement produces a variable number of raters per item, which kappa does not handle; the cost is reduced comparability with prior judge-validation work that reports kappa. We added the per-stratum calibration and lower-envelope certification because the load-bearing assumption (probe accuracy as a sufficient statistic for reliability degradation) is not guaranteed by the frontier bound alone; the cost is that the certificate is more conservative than a pure probe-gap schedule, but the benefit is that it does not depend on the probe being distribution-matched to the evaluation corpus. A domain expert would not describe GapCert as a variant of AltRA-Test because AltRA-Test tests a fixed judge on a fixed corpus, whereas GapCert produces a function-valued certificate whose argument is a property of models not yet released at validation time — the capability gap — and whose calibration point comes from the frontier bound, not from the validation corpus itself.

**Why it measures what we claim.** The bootstrap lower quantile lower_b measures the judge's non-inferiority to human panels because the paired-panel construction holds the item distribution fixed and only swaps one annotator, so any systematic shift in IAA is attributable to the judge under the assumption that the judge's errors are exchangeable with the replaced human's errors; this assumption fails when the judge has a self-preferencing bias correlated with response style, in which case lower_b reflects style agreement rather than shared signal. The capability gap g_e = acc(M_e) − acc(J) on the probe measures the evaluated model's superiority over the judge because probe accuracy is a sufficient statistic for the judge's ranking reliability only under the assumption that probe difficulty is distribution-matched to the evaluation corpus; this assumption fails when the probe over-represents easy items, in which case g_e understates the true gap and the certificate is optimistic. The frontier anchor g* measures the gap at which the twice-the-data bound saturates because the bound's label-savings curve is monotone in judge accuracy under the assumption that the judge's accuracy on the evaluation distribution tracks its accuracy on the probe; this assumption fails when the evaluated model's errors are adversarially concentrated on items the judge handles well, in which case g* is mislocated and the margin schedule is miscalibrated for that model family. The per-stratum lower envelope lower_env measures the judge's worst-case non-inferiority across judge-strength strata because it takes the minimum of the overall bootstrap bound and each stratum's bound under the assumption that the strata bracket the judge's capability range; this assumption fails when no reference model in the strata is close to the evaluated model's capability, in which case lower_env is conservative but may under-certify.

## Contribution

["A gap-conditioned validation certificate (GapCert) that extends AltRA-Test's non-inferiority test from a fixed-judge scalar decision to a function of the judge-evaluated capability gap, with the margin schedule anchored at the frontier paper's twice-the-data bound.", 'A calibration procedure that locates the frontier anchor g* from a capability probe and a frontier model suite, yielding a certificate that degrades monotonically as the evaluated model surpasses the judge rather than failing abruptly.', "An empirical characterization of how much of AltRA-Test's non-inferiority guarantee survives capability-gap shift, reported as a max-certified-gap statistic per judge family."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AltRA-Test validation corpus + disjoint capability probe pool (n_probe=200) | Paired-panel construction plus clean gap estimation |
| Primary metric | Certification coverage at fixed α with realized non-inferiority | Detects over/under-certification directly |
| Baseline | AltRA-Test fixed-margin test | Tests whether gap-conditioning adds validity |
| Baseline | Learned IAA-on-gap regression | Tests whether frontier anchor beats fitted schedule |
| Baseline | Oracle re-run AltRA-Test per model | Upper bound on achievable certification |
| Ablation-of-ours | GapCert with constant margin τ(g)=τ_0 | Isolates contribution of the gap schedule |
| Ablation-of-ours | GapCert without per-stratum lower envelope | Isolates contribution of the calibration repair |
| Diagnostic | Probe-estimated g_e vs. oracle-estimated g_e on frontier suite | Direct test of load-bearing assumption |

### Why this setup validates the claim

The central claim is that a reusable, gap-conditioned certificate can correctly state which evaluated models remain safely judged by J, without re-collecting human annotations per model. The design makes this falsifiable by pairing a certification decision with a ground-truth check: for each evaluated model in the frontier suite we independently run the expensive oracle (fresh human annotations, full AltRA-Test) and record whether the judge is truly non-inferior. Coverage — the fraction of models where GapCert's certificate agrees with this oracle verdict — is the right primary metric because it directly measures whether the margin schedule is calibrated to real non-inferiority, not merely whether IAA is high. AltRA-Test with a fixed margin tests the sub-claim that conditioning on gap matters at all: if fixed-margin coverage matches GapCert, the gap machinery is unnecessary. The learned-regression baseline tests the sub-claim that the frontier anchor supplies principled endpoints rather than fitted ones, since a regression can fit the same data but has no external calibration point. The oracle baseline bounds what any reusable certificate could achieve, so we can distinguish "GapCert is wrong" from "no reusable certificate is possible." The constant-margin ablation isolates the schedule's contribution from the panel-construction inheritance, ensuring any gain is attributable to gap-conditioning rather than to AltRA-Test's bootstrap machinery. The per-stratum lower-envelope ablation isolates the contribution of the calibration repair, and the probe-vs-oracle diagnostic directly tests the load-bearing assumption that probe-estimated g_e matches oracle-estimated g_e on the frontier suite.

### Expected outcome and causal chain

**vs. AltRA-Test fixed-margin test** — On a case where the frontier suite contains a model clearly stronger than J (large positive g), the fixed-margin baseline certifies the judge because its single τ_0 ignores that the judge's IAA advantage evaporates at the frontier, producing false non-inferiority. Our method instead shrinks the admissible margin as g grows, refusing certification where the twice-the-data bound predicts label savings collapse, so we expect the fixed-margin baseline to over-certify specifically on the high-gap subset while both agree on low-gap models.

**vs. Learned IAA-on-gap regression** — On a case where the validation corpus has few high-gap models, the regression extrapolates the IAA-vs-gap curve from sparse high-g samples, producing an unstable schedule that swings with the fitted functional form. Our method instead pins the schedule's endpoint to the externally justified g* from the frontier bound, so its margin at large g is not data-starved; we expect the regression to show higher variance in certification decisions across bootstrap replicates at high g, with GapCert matching it at low g.

**vs. Oracle re-run AltRA-Test per model** — On a case where a newly released model sits near the decision boundary, the oracle re-collects human annotations and resolves the verdict with fresh data, which is correct but costs a full annotation round per model. Our method instead reuses the bootstrap panel and only re-estimates g_e on the cheap probe, so we expect GapCert's coverage to approach the oracle's on models well inside or outside the margin, with the gap concentrated in a narrow boundary band where probe noise dominates.

**vs. GapCert without per-stratum lower envelope** — On a case where the probe over-represents easy items (so g_e understates the true gap), the no-envelope variant certifies optimistically because it relies solely on the overall bootstrap bound. Our method instead takes the minimum over judge-strength strata, so we expect the no-envelope variant to over-certify on the high-gap subset while GapCert remains conservative, at the cost of some under-certification on low-gap models.

**Diagnostic: probe-estimated g_e vs. oracle-estimated g_e** — On the frontier suite, we compare g_e from the probe to g_e estimated from the oracle's fresh human annotations. If the two agree within bootstrap CI, the load-bearing assumption is supported; if the probe systematically understates the gap (e.g., on easy-item-heavy probes), the certificate is optimistic and the lower envelope is necessary.

### What would falsify this idea

If GapCert's coverage gain over the fixed-margin baseline is uniform across the gap spectrum rather than concentrated on the high-gap subset where the frontier bound predicts failure, then the gap-conditioning is not doing the causal work we claim and the certificate is merely a noisier AltRA-Test. Equally, if the constant-margin ablation matches GapCert's coverage, the entire schedule is inert and the central claim is wrong; and if the probe-vs-oracle diagnostic shows probe-estimated g_e diverges from oracle-estimated g_e on the frontier suite, the load-bearing assumption is false and the certificate's optimism is not repaired by the lower envelope alone.

## References

1. AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators

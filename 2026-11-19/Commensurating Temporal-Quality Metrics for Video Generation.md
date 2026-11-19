# Commensurating Temporal-Quality Metrics for Video Generation via Monotone Model-Pair Calibration

## Motivation

The field now has several evaluation signals for temporal quality in generated video — StrideWorld's stride-based action consistency, JEDi's JEPA embedding distance, and MJ-Video's Coherence & Consistency criterion — but no work establishes that these signals order the same model pairs identically, so cross-benchmark aggregation of temporal quality is unestablished. Each signal is validated only against human judgments on its own benchmark, which is exactly the structural condition under which two metrics can each align with humans yet disagree with each other on model pairs (StrideWorld reports human agreement on 140 start views; JEDi reports +34% human alignment over FVD; MJ-Video reports fine-grained preference gains). The root cause is that no shared model-pair set has been evaluated under all three signals with an explicit commensurability test, so the field cannot tell whether these are three views of one temporal invariant or three different constructs.

## Key Insight

If two temporal-quality metrics both track a single underlying temporal invariant, then on a shared model-pair set their induced orderings must be monotone-equivalent; a fitted monotone calibration with a rank-agreement test therefore turns 'do these metrics agree?' from an untestable claim into a measurable, falsifiable property of the metric pair.

## Method

**What it is.** **TEMPCO** (Temporal Metric Pair Commensuration) takes as input a shared set of generated-video model pairs, each pair consisting of two models evaluated on the same prompt/start-view suite, and outputs for each metric pair (StrideWorld stride-consistency, JEDi JEPA distance, MJ-Video Coherence & Consistency) a fitted monotone calibration curve plus a commensurability verdict (commensurate / partially commensurate / incommensurate) with a rank-agreement statistic. The verdict is computed on the **temporal component** of each metric after partialling out a non-temporal nuisance baseline, and is conditioned on a per-metric variance-ratio diagnostic that certifies the inter-model gap exceeds the inter-seed noise floor.

**Load-bearing assumption (stated explicitly).** TEMPCO assumes that after removing each metric's non-temporal nuisance component, the three residual temporal signals are monotone functions of a single shared latent temporal invariant, so that on a shared model-pair set their induced orderings must be monotone-equivalent; any residual rank disagreement is then attributable to construct divergence rather than to nuisance or generation noise. This assumption is not taken for granted: it is operationalized by (i) a nuisance-residualization step (Phase 2b) that removes the non-temporal component of each metric before ordering, and (ii) a variance-ratio diagnostic (Phase 2c) that excludes pairs whose inter-model gap does not exceed the bootstrap-derived noise floor, so that sign flips counted in tau reflect construct rather than noise.

**How it works.**

```pseudocode
INPUT: model set M (|M| >= 8 open video models), prompt/start-view suite S (shared, |S| = 140 start views as in StrideWorld), metrics g in {Stride, JEDi, MJ-Coherence}
OUTPUT: for each metric pair (g1, g2): monotone map f, Kendall tau, verdict, variance-ratio diagnostic

Phase 1 — Shared evaluation
  for each model m in M, each prompt/start-view s in S:
      generate K=4 videos per (m,s) with fixed seeds
      compute g1, g2, g3 per video; aggregate per (m,s) by median over K
  form model-pair set P = {(m_i, m_j, s) : i<j, s in S}  # paired on same prompt

Phase 2 — Pairwise ordering extraction
  for each (m_i, m_j, s) in P and each metric g:
      sign_g(i,j,s) = +1 if g(m_i) < g(m_j)  (lower-is-better convention per metric)
                      -1 if g(m_i) > g(m_j)
                       0 if |g(m_i)-g(m_j)| < epsilon_g  (epsilon_g = 0.05 * IQR of g on S)
  build agreement matrix A_g1,g2 over P: fraction of pairs with sign_g1 == sign_g2 (ties excluded)

Phase 2b — Nuisance residualization (temporal-component extraction)
  # non-temporal nuisance baselines, computed on the same videos:
  #   n_JEDi = FVD (distributional visual-fidelity distance) on the same (m,s) clips
  #   n_Stride = static-frame quality score (first-frame FID to the start view)
  #   n_MJ = MJ-Video Fineness sub-score (non-temporal fine-grained criterion)
  for each metric g with nuisance baseline n_g:
      fit OLS  g ~ a + b * n_g  on per-video scores over all (m,s)
      g_tilde = g - (a + b * n_g)   # temporal residual used for all downstream ordering
  replace g by g_tilde in sign_g(i,j,s) and in Phase 3

Phase 2c — Variance-ratio diagnostic (noise floor)
  for each metric g, each model pair (m_i, m_j):
      sigma_seed_g(i,j) = std over K seeds of |g_tilde(m_i) - g_tilde(m_j)| per (m,s), pooled over S
      gap_g(i,j)        = median_s |g_tilde(m_i) - g_tilde(m_j)|
      VR_g(i,j)         = gap_g(i,j) / sigma_seed_g(i,j)
  noise_floor_g = 1.0  # require VR >= 1.0 to count a pair
  exclude (m_i, m_j, s) from P_g if VR_g(i,j) < noise_floor_g
  report per-metric: fraction of pairs excluded, median VR_g

Phase 3 — Monotone calibration fit
  for each metric pair (g1, g2):
      fit isotonic regression f: g1_tilde -> g2_tilde on per-video scores (monotone non-decreasing, no parametric form)
      residual R = mean |f(g1_tilde) - g2_tilde| / IQR(g2_tilde)
      bootstrap over S (B=1000 resamples of start views) to get CI on Kendall tau between f(g1_tilde) and g2_tilde

Phase 4 — Commensurability verdict
  tau = Kendall tau(sign_g1, sign_g2) over P (ties excluded, pairs failing Phase 2c excluded)
  verdict = COMMENSURATE if tau >= 0.8 and CI lower bound > 0.7 and R < 0.25 and median VR_g1 >= 1.0 and median VR_g2 >= 1.0
            PARTIALLY COMMENSURATE if 0.5 <= tau < 0.8
            INCOMMENSURATE if tau < 0.5
  # for triples: test transitivity of the three pairwise verdicts
```

**Why this design.** We chose isotonic regression over a parametric linear/rank map because commensurability is an ordinal claim (does the ordering agree?), and a parametric fit would smuggle in a cardinality assumption the metrics do not license, accepting the cost that isotonic fits are less sample-efficient and require the 140-start-view suite to be reasonably dense. We chose a shared model-pair set paired on identical prompts/start views over independently sampled evaluations because metric disagreement can be driven entirely by prompt-mix differences rather than by construct differences — this is the same confound that makes StrideWorld's and JEDi's separate human-agreement numbers non-comparable — accepting the cost that we must regenerate videos under a single harness, which is expensive. We chose a rank-agreement statistic (Kendall tau with bootstrap CI) over Pearson correlation because the downstream use is model selection (which model is better?), which is ordinal, accepting the cost that tau discards magnitude information that a distribution-level metric like JEDi encodes. We added the Phase 2b nuisance residualization because JEDi is a distributional distance that may track visual fidelity more than temporal consistency, so without removing its non-temporal component tau would reflect nuisance-driven disagreement rather than construct incommensurability; we added the Phase 2c variance-ratio diagnostic because K=4 median aggregation can leave residual inter-seed variance comparable to the inter-model gap, so without a noise floor sign flips would conflate generation noise with construct disagreement. A domain expert would not describe TEMPCO as 'applying isotonic regression to metrics': the contribution is the shared-pair experimental design plus the explicit falsification verdict, which no prior metric paper (StrideWorld, JEDi, MJ-Video) constructs — each validates only against humans, never against another metric on shared pairs.

**Why this measures what we claim.** The computational quantity `sign_g(i,j,s)` measures the motivation-level concept *metric-induced model ordering* because it assumes the paired generation (same prompt, same seeds) removes prompt-mix confounds so that any sign disagreement reflects the metric's construct rather than sampling noise; this assumption fails when a model's output is stochastic enough that K=4 median aggregation leaves residual variance comparable to the inter-model gap, in which case sign flips reflect generation noise, not construct disagreement — which is why Phase 2c excludes pairs with VR_g < 1.0 and reports the excluded fraction per metric. The computational quantity `Kendall tau` over P measures *commensurability* because it assumes that if both metrics track one temporal invariant, their induced orderings must be monotone-equivalent (tau near 1); this assumption fails when one metric is dominated by a non-temporal nuisance (e.g., JEDi's JEPA distance may track visual fidelity more than temporal consistency), in which case tau reflects nuisance-driven disagreement rather than construct incommensurability — which is why Phase 2b residualizes each metric on its nuisance baseline before ordering, and the verdict should be read as 'these metrics measure different things' rather than 'one is wrong'. The computational quantity `isotonic residual R` measures *calibration quality* because it assumes a monotone map exists between the two scores if they are commensurate; this assumption fails when the relationship is non-monotone (e.g., U-shaped), in which case R is large even for a commensurate pair, and the verdict should be downgraded to partial rather than incommensurate.

## Contribution

(1) A commensurability-calibration framework (TEMPCO) that, for the first time, evaluates StrideWorld stride-consistency, JEDi JEPA distance, and MJ-Video Coherence & Consistency on a shared model-pair set and outputs a falsifiable commensurate/partially/incommensurate verdict per metric pair with bootstrap rank-agreement CIs. (2) An empirical finding about which temporal-quality metrics are monotone-equivalent and which are not, providing the field's first cross-benchmark ordering-consistency evidence for video-generation temporal evaluation. (3) A reusable shared model-pair evaluation harness and calibration artifacts (isotonic maps, agreement matrices) that downstream benchmark papers can adopt to report cross-metric consistency instead of only human agreement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Shared 140-start-view suite, 8+ open video models, K=4 seeds | Paired generation removes prompt-mix confound |
| Primary metric | Kendall tau on model-pair orderings with bootstrap CI (B=1000 start-view resamples) | Ordinal agreement matches model-selection use |
| Diagnostic metric | Variance-ratio VR_g(i,j) = median inter-model gap / inter-seed std, per metric per model pair; noise floor VR >= 1.0 | Quantifies fraction of sign flips attributable to generation noise vs construct |
| Diagnostic metric | Nuisance-residual R² of g ~ a + b*n_g per metric (n_JEDi = FVD, n_Stride = first-frame FID, n_MJ = MJ-Video Fineness) | Quantifies how much of each metric is non-temporal before residualization |
| Baseline | Per-metric human-agreement validation (as in StrideWorld/JEDi) | Tests whether isolated human alignment suffices |
| Baseline | Independent-sampling tau (unpaired prompt mixes) | Tests whether shared pairing drives the verdict |
| Baseline | Parametric linear/rank calibration map | Tests whether isotonic monotone fit is necessary |
| Baseline | Raw-metric tau (no Phase 2b residualization) | Tests whether nuisance residualization changes the verdict |
| Ablation-of-ours | TEMPCO without epsilon tie-band | Tests sensitivity of sign extraction to noise |
| Ablation-of-ours | TEMPCO without Phase 2c noise floor (all pairs counted) | Tests whether noise-driven sign flips inflate or deflate tau |

### Why this setup validates the claim

The central claim is that commensurability between temporal metrics is a property of shared-pair orderings of the temporal component, and that metrics validated only against humans (StrideWorld, JEDi, MJ-Video) cannot establish it. The per-metric human-agreement baseline tests the sub-claim that isolated human alignment is insufficient: if that baseline already yields high cross-metric tau, TEMPCO adds nothing. The independent-sampling baseline tests the sub-claim that shared pairing is the causal lever: if unpaired tau matches paired tau, prompt-mix confounding is not the driver and our design rationale collapses. The parametric baseline tests whether the ordinal isotonic fit matters versus a cardinality-smuggling linear map. The raw-metric baseline tests the load-bearing assumption directly: if raw-metric tau equals residualized tau, the nuisance component is not driving disagreement and Phase 2b is inert; if residualized tau is systematically higher on pairs where the nuisance R² is high, the residualization is doing the claimed work. The variance-ratio diagnostic grounds the noise caveat quantitatively: it reports, per metric, the fraction of model pairs whose inter-model gap does not exceed the inter-seed noise floor, so the claim that sign disagreement measures construct is supported by a measured exclusion rate rather than assumed. The epsilon-band and noise-floor ablations test whether the verdict is fragile to tie handling and to noise-driven sign flips. Kendall tau is the right primary metric because the downstream question is ordinal model selection; a Pearson-style statistic would reward magnitude agreement that the metrics do not license and would mask the ordering disagreements we predict. Together these form a falsifiable test: the claim predicts paired tau clearly exceeds unpaired tau, residualized tau exceeds raw tau where nuisance R² is high, and isotonic residuals stay low only where verdicts are commensurate.

### Expected outcome and causal chain

**vs. Per-metric human-agreement validation** — On a case where StrideWorld and JEDi each report strong human agreement but on disjoint prompt mixes, the baseline treats both as validated and implicitly comparable because it never places them on shared pairs. Our method instead forces both metrics onto identical start views and seeds, so any sign disagreement is attributable to construct rather than sampling. We expect a noticeable tau gap between the StrideWorld/JEDi pair and the StrideWorld/MJ-Coherence pair, with the latter higher because both target coherence rather than distributional fidelity.

**vs. Independent-sampling tau** — On a case where two metrics are evaluated on different prompt mixes with different difficulty distributions, the baseline produces depressed or inflated tau because prompt-mix variance leaks into the ordering. Our method pairs on identical prompts and seeds, removing that variance. We expect paired tau to exceed unpaired tau by a clear margin specifically on metric pairs whose underlying constructs overlap, with near-parity on pairs where constructs genuinely diverge — the divergence being the signal, not noise.

**vs. Parametric linear/rank calibration** — On a case where the true relation between StrideWorld stride and JEDi distance is monotone but strongly nonlinear, the baseline produces a large residual and may falsely flag incommensurability because the linear form cannot bend. Our method's isotonic fit absorbs monotone nonlinearity, so residual R drops while tau is unchanged. We expect the parametric baseline to over-flag incommensurate verdicts on nonlinear pairs, while TEMPCO's verdicts stay stable and its residual stays below the 0.25 threshold exactly on the commensurate pairs.

**vs. Raw-metric tau (no Phase 2b residualization)** — On a case where JEDi's JEPA distance is dominated by visual-fidelity variance (high nuisance R² against FVD), the raw-metric baseline orders models largely by fidelity and produces low tau against StrideWorld even though the temporal components agree. Our method residualizes JEDi on FVD before ordering, so the temporal components are compared directly. We expect residualized tau to exceed raw tau by a margin that scales with the nuisance R² of the metric pair, and we expect the variance-ratio diagnostic to show that the excluded-pair fraction is low for StrideWorld (per-video geometric score, low seed variance) and higher for JEDi on short clips — quantifying the noise caveat rather than assuming it away.

### What would falsify this idea

If paired and unpaired Kendall tau were statistically indistinguishable across all metric pairs, the shared-pair design would not be doing the causal work and the central claim would be wrong. Likewise, if residualized tau were indistinguishable from raw tau on pairs with high nuisance R², the temporal-component claim would fail; and if isotonic residuals were uniformly low regardless of verdict, the calibration and verdict machinery would be measuring noise rather than construct commensurability.

## References

1. StrideWorld: Benchmarking Scene Rigidity and Action Consistency in Interactive Video World Model

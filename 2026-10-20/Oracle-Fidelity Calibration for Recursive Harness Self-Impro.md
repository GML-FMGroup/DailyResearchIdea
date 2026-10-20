# Oracle-Fidelity Calibration for Recursive Harness Self-Improvement

## Motivation

HarnessPro and similar recursive harness-improvement loops select candidate harnesses by their pass rates on LLM-generated tests, but the mapping from generated-test outcomes to true harness quality is never validated against official tests. Because the loop is closed over generated tests only, Pareto selection can systematically favor harnesses that exploit generator artifacts (e.g., brittle assertions, leaked solution hints) rather than genuine capability, and this assumption is inherited unchanged from the evaluation design.

## Key Insight

A per-candidate affine calibration between generated-test pass rate and held-out official-test pass rate, estimated on a stratified fidelity probe, is monotone in true harness quality whenever the probe covers the generated-test difficulty distribution, so reweighting candidates by calibrated rather than raw generated-test scores removes generator artifacts that are uncorrelated with official outcomes.

## Method

```pseudocode
Input: harness population H_0, generated-test suite G, official-test suite O (held-out, never shown to agents), probe size k (default k=120), calibration strata S, residual threshold tau (default tau=0.10)
Output: selected harnesses H* under cost budgets

0. PROBE BUDGET AND ORACLE CACHE
   - Fixed probe budget: k=120 generated tests per iteration, allocated as k/|S| per stratum.
   - Oracle-test cache: P_t^O is drawn once from O and cached; the same cached oracle tests are reused across all iterations t (no new oracle draws after t=0), so total oracle-test evaluations = k for the whole run.
   - Total generated-test evaluations per iteration = |H_t| * |G| (unchanged from HarnessPro).

1. STRATIFIED FIDELITY PROBE
   For each iteration t:
     S = cluster(G, by=assertion signature + difficulty bin)   # |S| strata, default |S|=6
     P_t = sample k/|S| = 20 generated tests per stratum s in S
     P_t^O = corresponding official tests (oracle, from cache)
     Split each stratum's probe into two disjoint halves:
       P_t^fit  (10 tests per stratum, used to fit a_h)
       P_t^eval (10 tests per stratum, used to compute r_h out-of-sample)

2. PER-CANDIDATE CALIBRATION (cross-fitted, per-stratum vector)
   For each candidate h in H_t:
     For each stratum s in S:
       g_h^s = pass rate of h on P_t^fit ∩ stratum s   (fit half)
       o_h^s = pass rate of h on P_t^O ∩ stratum s     (fit half, oracle)
     Fit affine map a_h: g -> o by least squares on the |S| per-stratum pairs {(g_h^s, o_h^s)}_{s∈S}
       (|S|=6 points per candidate, not a single aggregate pair)
     Calibrated score: c_h = a_h( pass_rate(h, G) )
     Out-of-sample fidelity residual:
       r_h = mean_{s∈S} | a_h( pass_rate(h, P_t^eval ∩ stratum s) ) - pass_rate(h, P_t^O ∩ stratum s) |
       (residual computed on the eval half, never used to fit a_h)
     Fit residual variance: v_h = variance of the |S| fit residuals on P_t^fit (reported as a diagnostic; used to flag candidates with |S| < 3 usable strata, which fall back to a global monotone map fit on all candidates' pooled per-stratum pairs)

3. ARTIFACT FILTER
   Reject h if r_h > tau (tau = 0.10) OR if c_h < median(c) - 2*MAD(c)
   (r_h is out-of-sample, so the filter can fire on artifact-exploiters whose fit-half calibration is overfit)

4. PARETO SELECTION ON CALIBRATED SCORES
   For each budget b in {lite, medium, heavy}:
     Select h*_b = argmin_h  w_b * (1 - c_h) + (1-w_b) * cost(h)
     (w_b from HarnessPro normalized weighted squared-distance, re-tuned on calibrated scores)

5. RECURSIVE LOOP
   H_{t+1} = H_t ∪ {mutate(h*_b) for b in budgets}
   Repeat until convergence or budget exhausted (max 10 iterations)
```

**Why this design**
We chose a stratified probe over uniform random sampling because generated tests are imbalanced across assertion types and difficulty bins, and uniform sampling would under-represent rare artifact-prone strata, accepting the cost of needing a stratum definition. We chose a per-candidate affine calibration over a global linear correction because different harnesses interact with generator artifacts differently (e.g., heavy harnesses with more tools may trigger more brittle assertions), accepting the cost of needing at least two probe points per candidate. We chose hard filtering on the fidelity residual over soft down-weighting because a candidate whose calibrated score diverges from its oracle score is structurally untrustworthy, accepting the risk of discarding a candidate that is genuinely good but poorly probed. We chose to keep HarnessPro's normalized weighted squared-distance selection intact on calibrated scores rather than replacing the selector, because the selector's cost-performance trade-off is orthogonal to the fidelity problem, accepting the cost that the selector's weights must be re-tuned on calibrated rather than raw scores. A domain expert would not describe this as a variant of HarnessPro because HarnessPro has no oracle anchor at all; the fidelity probe and per-candidate calibration are new mechanisms that change what the selector optimizes.

**Why this measures what we claim**
The per-candidate affine calibration c_h = a_h(pass_rate(h,G)) measures true harness quality because we assume the generated-test pass rate is a monotone (affine) function of official-test pass rate within the probe strata; this assumption fails when a harness exploits a generator artifact that is absent from official tests, in which case c_h reflects artifact-adjusted capability rather than raw generated-test performance. The out-of-sample fidelity residual r_h measures generator-artifact susceptibility because we assume the probe strata are representative of the full generated-test distribution (assumption A: probe-stratum artifact exposure is proportional to full-suite artifact exposure); this assumption fails when a rare artifact-prone stratum is not sampled (failure mode F), in which case r_h underestimates the artifact gap. The stratified probe P_t measures distributional coverage because we assume clustering by assertion signature and difficulty bin captures the relevant variation; this assumption fails when a new assertion type appears only in the full suite, in which case the probe misses that stratum and calibration is biased. The cross-fitted split (fit half vs eval half) measures fidelity rather than in-sample fit error because we assume the eval half is exchangeable with the fit half within each stratum; this assumption fails when the eval half happens to share the same artifact as the fit half, in which case r_h is small for an artifact-exploiter and the filter does not fire.

## Contribution

['A fidelity-calibrated evaluation layer that fits a per-candidate affine map from generated-test pass rate to held-out official-test pass rate on a stratified probe, and uses the calibrated score plus a fidelity residual to filter and reweight candidates before Pareto selection.', 'A design principle: recursive self-improvement loops closed over generated tests require an explicit oracle-fidelity certification step whose probe is stratified by assertion signature and difficulty, because uniform probes under-sample artifact-prone strata and inflate apparent harness quality.', 'An open-source calibration harness that plugs into HarnessPro-style loops, exposing per-candidate fidelity residuals and calibrated scores for audit.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-Bench Pro + Terminal-Bench 2 with held-out official suite O | Oracle anchor exists; generated/official split enables fidelity probe |
| Primary metric | Official-test pass rate of selected harness at matched cost budget | Directly measures artifact-adjusted capability, not generated-test inflation |
| Baseline | HarnessPro (raw-score Pareto selection) | Tests whether calibration changes what the selector optimizes |
| Baseline | Uniform-random probe instead of stratified probe | Tests whether stratification is needed for residual fidelity |
| Baseline | Global linear correction instead of per-candidate affine | Tests whether per-candidate calibration matters |
| Baseline | Agreement-based ensembling (no oracle anchor) | Tests whether oracle anchoring is replaceable by cheaper proxy-validation |
| Baseline | Learned artifact classifier on generated-test features (no oracle anchor) | Tests whether oracle anchoring is replaceable by a learned proxy |
| Ablation-of-ours | Ours minus artifact filter (soft down-weighting only) | Tests whether hard residual filtering is load-bearing |
| Ablation-of-ours | Ours with single-pair affine fit (original) instead of per-stratum vector fit | Tests whether per-stratum identifiability is load-bearing |
| Ablation-of-ours | Ours with in-sample residual instead of cross-fitted out-of-sample residual | Tests whether cross-fitting is load-bearing |

### Why this setup validates the claim

The central claim is that generated-test pass rates are a biased proxy for official-test quality, and that per-candidate affine calibration anchored on a stratified oracle probe removes that bias so Pareto selection picks genuinely better harnesses. HarnessPro is the right baseline because it isolates exactly the mechanism we add: it uses the same selector and the same generated suite but optimizes raw scores, so any gap between our method and HarnessPro is attributable to calibration and filtering rather than to the search procedure. The uniform-probe baseline tests the stratification sub-claim, and the global-correction baseline tests the per-candidate sub-claim; if either degrades toward HarnessPro, the corresponding mechanism is not doing work. The agreement-based ensembling and learned artifact classifier baselines test whether the oracle anchor is replaceable by cheaper proxy-validation tricks: if either matches our method, the oracle anchor is not load-bearing. The ablation (no hard filter) tests whether the residual threshold is load-bearing or merely cosmetic. The single-pair-fit and in-sample-residual ablations test whether the cross-fitted per-stratum vector fit is what makes r_h a genuine fidelity signal rather than an in-sample fit error. Official-test pass rate at matched cost is the correct primary metric because it is the only quantity not contaminated by generator artifacts, and it is exactly what the calibration is supposed to track; a method that improves generated-test scores without moving official-test scores would be detected as a null result. This combination is falsifiable because each mechanism has a dedicated contrast that can fail independently.

### Expected outcome and causal chain

**vs. HarnessPro (raw-score Pareto selection)** — On a case where a heavy harness with many tools passes a brittle generated assertion that official tests never check, HarnessPro scores it high on G and promotes it into the heavy budget because it has no oracle anchor to notice the mismatch. Our method fits a per-candidate affine map on the |S|=6 per-stratum (g_h^s, o_h^s) pairs, sees the out-of-sample calibrated score diverge from the oracle probe pass rate on the eval half, and the residual r_h exceeds tau, so the harness is filtered before selection. We expect a noticeable official-test gap concentrated on artifact-prone strata (brittle assertion types, high tool-count harnesses) with near-parity on clean strata, and a lower variance in selected-harness official scores.

**vs. Uniform-random probe** — On a case where a rare assertion signature (say, a specific timeout or ordering assertion) is the only place a harness cheats, uniform sampling draws few such tests, so the probe pass rate is dominated by common strata and the residual r_h stays small, letting the artifact pass the filter. Our stratified probe allocates k/|S|=20 tests to every stratum including the rare one, so the artifact shows up in the probe and inflates the out-of-sample residual. We expect our method to catch a larger fraction of artifact-exploiting candidates, visible as fewer artifact-prone harnesses surviving into H* and a wider official-vs-generated gap closed on the rare-stratum subset specifically.

**vs. Global linear correction** — On a case where two harnesses have the same generated pass rate but different artifact susceptibility (one tool-heavy, one lean), a single global map applies the same correction to both, so their calibrated scores stay tied and the selector cannot distinguish them. Our per-candidate affine map absorbs each harness's own interaction with generator artifacts, separating their calibrated scores. We expect the selected harness set to differ from the global-correction set mainly among harnesses with divergent tool counts, with our method's picks showing higher official pass rates on the tool-heavy subset while tying on the lean subset.

**vs. Agreement-based ensembling (no oracle anchor)** — On a case where multiple generated tests agree on a harness's quality but all share the same generator artifact, agreement-based ensembling cannot detect the artifact because it has no oracle anchor. Our method's out-of-sample residual r_h against the cached oracle probe detects the divergence. We expect our method to show higher official-test pass rates on artifact-prone strata while tying on clean strata, confirming the oracle anchor is not replaceable by agreement alone.

**vs. Learned artifact classifier on generated-test features (no oracle anchor)** — On a case where the artifact signature is novel and not represented in the classifier's training features, the classifier mislabels the artifact-exploiting harness as clean. Our method's per-candidate calibration against the oracle probe does not depend on pre-specified artifact features. We expect our method to catch a larger fraction of artifact-exploiters on novel artifact types, visible as fewer artifact-prone harnesses surviving into H*.

**vs. Ours minus artifact filter (soft down-weighting only)** — On a case where an artifact-exploiting harness has a high calibrated score c_h but a large out-of-sample residual r_h, soft down-weighting may still promote it if its calibrated score is high enough. Hard filtering removes it. We expect the hard filter to yield higher official-test pass rates on artifact-prone strata, with the gap concentrated on candidates whose r_h exceeds tau.

**vs. Ours with single-pair affine fit (original)** — On a case where a harness's single aggregate (g_h, o_h) pair yields a near-degenerate fit, the in-sample residual is mechanically small and the filter does not fire. Our per-stratum vector fit (|S|=6 points) and cross-fitted out-of-sample residual make r_h a genuine fidelity signal. We expect our method to catch more artifact-exploiters, visible as fewer artifact-prone harnesses surviving into H*.

**vs. Ours with in-sample residual instead of cross-fitted out-of-sample residual** — On a case where a 2-parameter fit on few points overfits, the in-sample residual is small even for an artifact-exploiter. Our cross-fitted residual on the eval half is not mechanically small. We expect our method to fire the filter on artifact-exploiters whose in-sample residual is below tau, visible as a larger fraction of artifact-prone harnesses rejected.

### What would falsify this idea

If our official-test gains over HarnessPro are uniform across all strata rather than concentrated on the artifact-prone strata the fidelity residual is designed to detect, or if the ablation without hard filtering matches our full method, or if the agreement-based ensembling or learned artifact classifier baselines match our method, then calibration and filtering are not the causal mechanisms and the central claim is wrong. If the single-pair-fit or in-sample-residual ablations match our full method, then the cross-fitted per-stratum vector fit is not load-bearing and the identifiability repair is unnecessary.

## References

1. HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-offs

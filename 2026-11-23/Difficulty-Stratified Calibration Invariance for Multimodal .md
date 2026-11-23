# Difficulty-Stratified Calibration Invariance for Multimodal Benchmark Evaluation

## Motivation

Existing multimodal benchmarks report a single accuracy or ELO number at one fixed difficulty regime and treat it as a sufficient summary of evaluation signal quality — PURE curates 1,457 fixed MCQs whose difficulty is baked in at construction time and reports overall scores of 47.1%–86.0% without ever testing whether the signal behaves consistently as difficulty shifts, while VideoAutoArena's fault-driven evolution raises question complexity but imposes no invariant tying model confidence or judge agreement to accuracy across the resulting regimes. Structurally this fails because difficulty is treated as a nuisance to be averaged over rather than as a conditioning variable that can expose signal inversion: a model can be perfectly ranked at one stratum and anti-ranked at another while its aggregate score is unchanged, so the benchmark cannot distinguish a genuinely competent model from one whose apparent competence is an artifact of where the difficulty mass happens to sit.

## Key Insight

If difficulty is made an explicit, independently measured conditioning variable rather than a property implicit in curated items, then the monotonicity of confidence-to-accuracy and judge-agreement-to-accuracy relations across difficulty strata becomes a testable invariant, and violations of it identify evaluation-signal failure that no single-stratum aggregate score can reveal.

## Method

**A. What it is.**
We introduce CALIBRA-PURE, a difficulty-stratified calibration-invariance audit layered on PURE's 14 tasks. Inputs: PURE's 1,457 multiple-choice items with model responses and per-item confidence, plus judge agreement scores. Outputs: per-model per-stratum calibration curves, a Calibration-Invariance Violation (CIV) flag per model, and a decomposition of aggregate accuracy into stratum-weighted components.

**B. How it works.**

```pseudocode
INPUT: items I (1457), tasks T (14), models M (10), judges J (3)
# Phase 1: difficulty measurement, independent of the model under test
for each item i in I:
    d_i = 1 - mean_{m in M \ {m_test}} [ correct(m, i) ]   # leave-one-model-out pass rate
    # difficulty is a property of the item, estimated from OTHER models
strata = quantile_bin(d_i, K=5)   # K=5 equal-mass difficulty strata per task

# Phase 2: per-stratum signal extraction
for each model m, stratum k:
    acc[m,k]  = mean_{i in k} correct(m,i)
    conf[m,k] = mean_{i in k} confidence(m,i)         # elicited 0-100 verbalized + logit-margin average
    agree[m,k]= mean_{i in k} mean_{j in J} [ judge_j(i) == correct(m,i) ]

# Phase 3: monotonicity tests (the invariant)
for each model m:
    rho_conf[m]  = Spearman(conf[m,1..K], acc[m,1..K])   # require rho > 0
    rho_agree[m] = Spearman(agree[m,1..K], acc[m,1..K])  # require rho > 0
    CIV[m] = 1 if (rho_conf[m] <= 0) OR (rho_agree[m] <= 0) else 0

# Phase 4: stratum-weighted decomposition
acc_agg[m] = sum_k w_k * acc[m,k]        # w_k = stratum mass
acc_shift[m] = max_k acc[m,k] - min_k acc[m,k]

OUTPUT: CIV flags, rho vectors, acc_shift, per-stratum curves
```

**C. Why this design.**
We chose leave-one-model-out pass rate over a learned difficulty predictor because difficulty must be measured without contaminating it with the model being audited — a learned predictor trained on the same ten models would let a model's own errors define its difficulty strata and manufacture apparent monotonicity, accepting the cost that strata are only as stable as the ten-model population. We chose Spearman rank monotonicity over a parametric calibration fit (e.g., ECE) because ECE conflates miscalibration magnitude with difficulty shift and cannot distinguish a uniformly overconfident model from one whose confidence inverts across strata, accepting the cost that rank tests are insensitive to absolute calibration offset. We chose verbalized-plus-logit-margin averaged confidence over log-prob alone because the anti-pattern of treating raw log-probs as gateable is well documented, accepting the cost of a noisier but more robust signal. A domain expert would not describe this as a variant of PURE or VideoAutoArena: PURE reports one accuracy per model with difficulty fixed at construction, and VideoAutoArena's fault-driven evolution changes complexity without ever conditioning signal quality on the resulting regime — CALIBRA-PURE's load-bearing object, the difficulty-conditioned monotonicity invariant, has no counterpart in either.

**D. Why it measures what we claim.**
`d_i` (leave-one-model-out pass rate) measures item difficulty because the assumption that aggregate cross-model failure rate is a sufficient statistic for the cognitive demand of an item holds; this assumption fails when the model population is homogeneous in a shared blind spot, in which case `d_i` reflects population-specific error rather than intrinsic difficulty. `rho_conf` (Spearman rank correlation between stratum-mean confidence and stratum-mean accuracy) measures calibration invariance because the assumption that a calibrated model's confidence must track its accuracy monotonically as difficulty varies holds; this assumption fails when confidence is uniformly saturated at ceiling, in which case `rho_conf` reflects a degenerate tie rather than genuine inversion. `rho_agree` measures judge-signal stability because the assumption that a reliable judge's agreement with ground truth should degrade in step with model accuracy holds; this assumption fails when judges share a systematic bias with the model under test, in which case `rho_agree` reflects correlated judge-model bias rather than evaluation-signal quality. `CIV[m]` operationalizes the motivation-level concept of evaluation-signal robustness because the conjunction of both monotonicity tests failing is assumed to be necessary and sufficient for signal inversion; this assumption fails when a model is genuinely competent but has a single noisy stratum, in which case `CIV` reflects sampling variance rather than structural signal failure — which is why `acc_shift` is reported alongside as a magnitude check.

**Load-bearing assumption (stated explicitly).** The audit assumes that stratum-mean confidence and stratum-mean judge agreement must rank-correlate positively with stratum-mean accuracy across difficulty strata for a model to be considered well-behaved — i.e., that monotonicity of these aggregate relations across difficulty is a necessary property of a valid evaluation signal rather than an artifact of how confidence and difficulty are jointly aggregated. This assumption is load-bearing and is not verified by the rank test alone; it is tested by the permutation null and synthetic injection described in the experiment section.

## Contribution

(1) A difficulty-stratified calibration-invariance audit protocol (CALIBRA-PURE) that conditions evaluation-signal quality on independently measured difficulty strata and emits a per-model CIV flag, turning difficulty from a nuisance averaged over into an explicit test variable. (2) The design principle that evaluation-signal validity requires monotonicity of both confidence-to-accuracy and judge-agreement-to-accuracy relations across difficulty strata, with stratum-weighted decomposition separating aggregate accuracy from stratum-shift magnitude. (3) A reusable audit harness over PURE's 14 tasks and 1,457 items that reports per-stratum calibration curves and exposes models whose aggregate scores conceal stratum-level signal inversion.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PURE (1,457 MCQs, 14 tasks) | Real multi-granularity items with reported accuracy spread |
| Primary metric | Calibration-Invariance Violation (CIV) rate | Directly tests the monotonicity invariant we claim |
| Baseline | PURE aggregate accuracy ranking | Tests whether difficulty-blind accuracy hides signal failure |
| Baseline | PURE per-capability-dimension breakdown | Tests whether dimension slicing substitutes for difficulty slicing |
| Baseline | ECE-based calibration diagnostic | Tests whether magnitude calibration captures inversion |
| Ablation-of-ours | CALIBRA-PURE with random strata | Isolates whether difficulty stratification drives the effect |
| Null calibration | Permutation null (label-shuffled and difficulty-shuffled) | Sets a per-model significance threshold for CIV so flags are defined relative to the null, not relative to zero |
| Synthetic injection | Known injected difficulty and confidence-inversion, varying M from 5 to 50 | Bounds the regime where leave-one-out difficulty is a valid proxy and measures CIV true-positive / false-positive rates |

### Why this setup validates the claim

The central claim is that aggregate accuracy on PURE hides a structural failure: some models' evaluation signals (confidence and judge agreement) invert across difficulty strata, and this inversion is invisible to the accuracy ranking that PURE currently reports. The falsifiable test is therefore a contrast between two orderings of the same ten models — the PURE accuracy ordering and the CIV/rho ordering — and whether they disagree in a patterned way. The aggregate-accuracy baseline tests the sub-claim that accuracy rank is not a sufficient summary; the capability-dimension baseline tests the sub-claim that slicing by capability is not a substitute for slicing by difficulty, since a model can be uniformly accurate across dimensions yet invert across strata within each. The ECE baseline tests the sub-claim that magnitude calibration is the wrong instrument: ECE can flag a model as miscalibrated while its rank ordering is perfectly monotone, and can pass a model whose confidence inverts between easy and hard strata but averages out. The random-strata ablation is the load-bearing control: if CIV flags appear at the same rate under random stratification, the invariant is measuring noise, not difficulty-conditioned structure. The permutation null and synthetic injection directly test the load-bearing assumption stated in the method: that joint failure of rho_conf and rho_agree cannot arise from sampling variance alone. CIV rate is the right primary metric because it is exactly the quantity the motivation names — an evaluation-signal robustness failure — and it is diagnostic rather than aggregate, so it can register a failure that leaves mean accuracy untouched.

### Expected outcome and causal chain

**vs. PURE aggregate accuracy ranking** — On a case where a model scores mid-pack overall (say ~70%) because it is strong on perception items and weak on evidence-grounded judgment, the baseline reports a single number and treats the model as uniformly middling, because its mechanism collapses all difficulty regimes into one mean and has no per-stratum confidence or judge signal. Our method instead recomputes confidence and judge agreement per difficulty stratum and tests their rank monotonicity, because it conditions signal quality on the regime the item occupies. We expect a noticeable CIV flag rate concentrated among mid-accuracy models and near-zero among the top and bottom models, with accuracy rank and CIV rank disagreeing on a minority but non-trivial subset.

**vs. PURE per-capability-dimension breakdown** — On a case where a model is uniformly competent across the four capability dimensions but its confidence saturates at ceiling on easy strata and drops sharply on hard strata while accuracy falls only mildly, the baseline reports four similar dimension scores and concludes stable capability, because dimension slicing holds difficulty roughly constant within each cell and never inspects the confidence-accuracy relationship. Our method instead bins by leave-one-model-out difficulty within each task and tests rho_conf across the five strata, because it treats difficulty as the axis along which signal stability must be checked. We expect the dimension breakdown to show near-parity across dimensions while our per-stratum curves show a clear monotonicity break, so the observable signal is a divergence between flat dimension profiles and non-flat rho vectors.

**vs. ECE-based calibration diagnostic** — On a case where a model is overconfident by a roughly constant offset across all strata, the baseline reports a large ECE and flags the model as badly calibrated, because ECE aggregates absolute confidence-accuracy gaps and cannot distinguish a uniform offset from a rank inversion. Our method instead asks only whether confidence rank-orders accuracy across strata, so a constant offset leaves rho_conf near 1 and produces no CIV flag. We expect the two diagnostics to disagree systematically: high-ECE models with monotone rho and low-ECE models with inverted rho, meaning the observable signal is a weak or negative association between ECE and CIV across the ten models.

**vs. CALIBRA-PURE with random strata (ablation)** — On a case where a model has a genuine inversion between the easiest and hardest difficulty strata, random stratification scatters easy and hard items across bins, so each bin's mean confidence and accuracy regress toward the overall mean and the rank correlation is attenuated toward zero, because random bins destroy the difficulty contrast the invariant depends on. Our method instead uses leave-one-model-out pass-rate quantiles, which preserve the contrast and keep strata comparable across models. We expect the ablation to show a markedly lower and noisier CIV flag rate, with the gap between difficulty-stratified and random-stratified CIV concentrated on the models our full method flags, which is the signal that stratification rather than sampling noise drives the result.

**vs. permutation null (label-shuffled and difficulty-shuffled)** — On a case where a model has a genuine inversion, the null distribution of rho under label-shuffled and difficulty-shuffled permutations provides a per-model threshold: CIV is flagged only when the observed rho falls below the 5th percentile of the null. We expect the observed CIV rate to exceed the null CIV rate by a clear margin, and the flagged models to be those whose rho lies outside the null envelope, confirming that the invariant is not measuring sampling variance.

**vs. synthetic injection (known difficulty and confidence-inversion, M from 5 to 50)** — On a synthetic population with injected difficulty and known confidence-inversion, we measure CIV true-positive and false-positive rates as a function of M. We expect true-positive rate to rise with M and false-positive rate to stay near the nominal 5% threshold, bounding the regime where leave-one-out difficulty is a valid proxy.

### What would falsify this idea

If CIV flags under difficulty stratification appear at essentially the same rate and on the same models as under random stratification, or if flagged models are indistinguishable from unflagged ones on acc_shift, or if the permutation null shows that observed CIV rates are within the null envelope, then the difficulty-conditioned monotonicity invariant is measuring sampling variance rather than a structural evaluation-signal failure, and the central claim is wrong. A second falsifier: if every model with high aggregate accuracy also has monotone rho_conf and rho_agree, the claim that accuracy rank hides signal inversion collapses.

## References

1. PURE: A Diagnostic Benchmark for Multi-Granularity Multimodal Identity Matching

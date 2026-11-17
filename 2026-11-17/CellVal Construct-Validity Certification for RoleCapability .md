# CellVal: Construct-Validity Certification for Role–Capability Benchmark Cells

## Motivation

Benchmark atlases such as InclusionMed assign each benchmark to a role–capability cell using structured audits and independent reproduction, then report capability-level scores and within-benchmark task-tag breakdowns (E1, E6). Neither audit nor reproduction tests whether a benchmark filed under, say, 'clinical decision-making' actually discriminates decision-making from the correlated capability 'patient communication' — the very comparison InclusionMed's own tag analysis surfaces, since the top ten models all score higher on communication than decision-making (E1). The structural root cause is that cell membership is certified by provenance and reproducibility, which are necessary but silent on discriminant validity: a benchmark can be perfectly reproducible and still measure a correlated capability, so every downstream capability-level score inherits an unverified attribution.

## Key Insight

If a benchmark's task-level tags carry a capability signal at all, then a model-by-tag score matrix restricted to that benchmark must exhibit a reproducible, model-invariant tag-factor structure in which the target capability loads on a distinct factor from correlated capabilities — a property that is testable with held-out models and that provenance audits cannot manufacture.

## Method

**What it is.** CellVal is a construct-validity certification layer that takes as input a benchmark B, its assigned role–capability cell c*, its task-level tag vocabulary T (including a target tag t* and a set of correlated-capability tags C), and a model panel M split into a certification fold M_cert and a held-out fold M_hold. It outputs a per-benchmark certificate with status {certified, conditionally-certified, rejected} plus the estimated tag-factor loading matrix.

**Load-bearing assumption (stated explicitly).** CellVal assumes that a benchmark's task-level tags carry a capability signal separable from topic and difficulty, such that a model-by-tag score matrix restricted to that benchmark will exhibit a reproducible, model-invariant factor structure in which the target capability loads on a distinct factor from correlated capabilities. This assumption is not tested by provenance audits and is the object of the tag-validity gate below; if it fails, the certificate is rejected as a tag–topic confound rather than silently relocating the provenance-vs-discriminant gap into the tag layer.

**How it works.**

```pseudocode
INPUT: benchmark B, cell c*, tags T (target t*, correlated set C), model panel M
HYPER: n_folds=5, loading_threshold=0.40, cross_loading_threshold=0.30,
       alpha_disc=0.05, min_models_per_fold=8, bootstrap_B=1000,
       topic_partial_r2_max=0.50, topic_loading_max=0.30

PHASE 0 — Tag-validity gate (topic-confound control)
  # t* and C must be defined by an explicit capability-demand rubric, not surface topic.
  # Topic tags T_topic are a disjoint vocabulary (e.g., organ system, modality, image type).
  require: t* and each c in C have a written capability-demand rubric entry
           (what cognitive operation the item demands), reviewed by 2 domain experts
  for each tag t in {t*} ∪ C:
      R2_topic[t] = R^2 from OLS of S[:, t] on the topic-tag score matrix S[:, T_topic]
  if max_t R2_topic[t] > topic_partial_r2_max:
      return {status: "rejected: tag-topic confound", R2_topic}
  # Residualize capability tags on topic tags before factor analysis
  S_res[:, {t*} ∪ C] = residuals of S[:, {t*} ∪ C] on S[:, T_topic]

PHASE 1 — Score matrix construction
  for each model m in M (M_cert ∪ M_hold):
      S[m, t] = mean correctness of m on items of B tagged t   # per-tag accuracy
  drop tags with < 20 items or < min_models_per_fold models scored

PHASE 2 — Factor structure on M_cert
  Z = column-standardize S_res[M_cert, :]
  (F, L) = factor_analysis(Z, n_factors=k)   # k chosen by parallel analysis
  identify target factor f* = argmax_f |L[t*, f]|
  # discriminant-validity tests
  (a) loading test:      L[t*, f*] >= loading_threshold
  (b) cross-loading:     max_{c in C} |L[c, f*]| <= cross_loading_threshold
  (c) known-groups test: for each correlated tag c in C, paired bootstrap
      (B=1000) of S_res[:, t*] - S_res[:, c] over M_cert; require the sign of
      the mean difference to be stable across >= 95% of bootstrap resamples
      AND |effect| >= alpha_disc-equivalent minimum effect size (Cohen d >= 0.2)
  (d) topic-control check: refit factor model with T_topic included as nuisance
      columns; require L[t*, f*] >= loading_threshold and
      max_{topic in T_topic} |L[topic, f*]| <= topic_loading_max

PHASE 3 — Model-invariance check on M_hold
  refit L_hold on Z[M_hold, :]
  congruence = Tucker congruence between L[t*, :] and L_hold[t*, :]
  require congruence >= 0.85

PHASE 4 — Certificate
  if (a) and (b) and (c) and (d) and (congruence ok):  status = certified
  elif (a) and (b) and (d):                            status = conditionally-certified
  else:                                                status = rejected
  return {status, L, congruence, bootstrap_sign_stability, R2_topic}
```

**Why this design.** We chose confirmatory factor analysis over a learned discriminant head because the target question is whether tag scores *already contain* a separable capability signal, not whether a new head can be trained to assert one — a learned head would be exactly the anti-pattern of manufacturing independence the data does not contain, and a reviewer would cite that the tags are the audit artifact under test, not a training surface. We chose a known-groups bootstrap test against correlated tags over a simple correlation threshold because correlation between t* and a correlated tag c can be high for benign reasons (both hard), whereas a *stable signed gap* is the signature of discriminant validity and is robust to difficulty confounds, accepting the cost that we need paired per-model scores and therefore more models than a correlation test. We chose a held-out model fold for the congruence check over in-fold cross-validation because the certification claim is about the benchmark, not the model panel, and in-fold CV would reward benchmarks whose tag structure is overfit to the specific frontier models in M_cert, accepting the cost that M_hold must be disjoint and large enough for stable loadings. We chose a three-way certificate over a binary pass/fail because a benchmark can pass loading and cross-loading but fail known-groups when the panel is small, and collapsing that into 'rejected' would discard usable benchmarks; the conditional tier preserves the audit trail. We added the Phase 0 tag-validity gate and the Phase 2(d) topic-control check because the load-bearing assumption — that tags carry capability rather than topic — is exactly what the factor analysis would otherwise take for granted; the gate makes the assumption explicit and falsifiable inside the method rather than relocating the provenance-vs-discriminant gap into the tag layer. This is not a variant of InclusionMed's structured audit (E6): that audit inspects provenance and reproducibility, whereas CellVal runs a statistical discriminant test on the benchmark's own score matrix — a domain expert would describe the audit as 'is this benchmark real and reproducible' and CellVal as 'does this benchmark measure what its cell claims'.

**Why it measures what we claim.** The target-factor loading `L[t*, f*]` measures *construct salience of the target capability* because the assumption is that if task-level tags are a valid operationalization of capabilities, then items sharing a capability must co-vary in model correctness beyond item difficulty and topic; this assumption fails when tags are assigned by surface topic rather than by capability demand, in which case the loading reflects topic clustering rather than capability — which is why Phase 0 residualizes on topic tags and Phase 2(d) requires the target factor to survive with topic tags as nuisance columns. The cross-loading bound `max_{c in C} |L[c, f*]|` measures *discriminant separation from correlated capabilities* because the assumption is that a correlated capability, if it were the true latent, would load on the same factor as the target; this assumption fails when a correlated capability is genuinely nested inside the target (e.g., communication always requires some decision-making), in which case the bound penalizes a legitimate benchmark. The bootstrap sign-stability of `S_res[:, t*] - S_res[:, c]` measures *known-groups discrimination* because the assumption is that if t* and c were the same construct, the per-model gap would be centered at zero and its sign would flip across resamples; this assumption fails when the panel is homogeneous in a way that fixes the sign for difficulty reasons, in which case the test reflects panel composition rather than construct separation. The Tucker congruence between M_cert and M_hold loadings measures *model-invariance of the tag-factor structure* because the assumption is that a construct-valid benchmark's factor structure is a property of the benchmark, not of the model panel; this assumption fails when the benchmark is so hard that most models sit at floor, in which case congruence reflects noise rather than invariance. The topic-partial R² `R2_topic[t]` measures *how much of a capability tag's score variance is explained by topic tags* because the assumption is that capability tags are not surface-topic proxies; this assumption fails when the tag vocabulary is topic-derived, in which case the certificate is rejected as a tag–topic confound rather than certified on a topic factor.

## Contribution

['A construct-validity certification layer (CellVal) that augments provenance-and-reproduction audits with confirmatory factor analysis, cross-loading bounds, known-groups bootstrap tests, and held-out-model congruence checks, producing a three-tier per-benchmark certificate.', 'A design principle: cell-to-capability assignment should be certified by discriminant validity against correlated capabilities within the same benchmark, not only by reproducibility, because reproducible benchmarks can still fail to separate correlated capabilities.', "A reusable audit protocol and certificate schema that can be applied to any benchmark atlas with task-level tags, demonstrated on the InclusionMed Atlas's role–capability cells."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | InclusionMed Atlas + DxFiles tag matrix | Has role–capability cells and task-level tags |
| Dataset | Synthetic ground-truth generator (separable, nested, topic-confounded benchmarks) | Provides known latent capability structure to quantify sensitivity/specificity |
| Ground truth | Pre-registered expert-rating protocol: 2 blinded clinicians rate each item's capability demand on the rubric; report Cohen's kappa; disagreements adjudicated by a third rater | Operationalizes the independent construct audit as a concrete label |
| Primary metric | Certificate status agreement with expert-rating label (accuracy, sensitivity, specificity) | Directly tests discriminant-validity verdict |
| Secondary metric | Minimum detectable effect size given panel size (power analysis) | Reports where the certificate is reliable vs noise-dominated |
| Baseline | InclusionMed structured audit | Provenance/reproducibility audit, no statistical test |
| Baseline | Raw per-tag accuracy ranking | Assumes tag scores are self-validating |
| Baseline | Correlation-threshold discriminant check | Naive t* vs c correlation gating |
| Ablation-of-ours | CellVal minus held-out congruence phase | Isolates model-invariance contribution |
| Ablation-of-ours | CellVal minus Phase 0 tag-validity gate and Phase 2(d) topic control | Isolates topic-confound control contribution |
| Control condition | Difficulty-matched item subset (items matched on empirical difficulty across t* and C) | Tests the additive-difficulty assumption for the signed gap |

### Why this setup validates the claim

The central claim is that a benchmark's tag structure can be statistically certified as a separable capability signal rather than a topic artifact, and that this certification is stable across model panels. The InclusionMed Atlas plus DxFiles gives us benchmarks already partitioned into role–capability cells with task-level tags, so we can ask whether CellVal's certificate agrees with an independent construct-level audit of the same benchmarks — that agreement is the falsifiable target. The independent audit is operationalized as a pre-registered expert-rating protocol with two blinded clinicians rating each item's capability demand and reported inter-rater kappa, so the agreement metric is against a concrete label rather than an undefined audit. The synthetic ground-truth generator supplies benchmarks with known latent structure (separable, nested, topic-confounded), letting us quantify CellVal's sensitivity and specificity against a known label and distinguishing it from any generic factor-analysis application. The structured-audit baseline tests whether provenance/reproducibility checks alone predict construct validity (we predict they do not, because they never touch the score matrix). The raw-accuracy and correlation-threshold baselines test whether simpler statistical summaries already recover the same verdict; if they match CellVal everywhere, our factor machinery is unnecessary. The ablation removes the held-out congruence phase to show that model-invariance is doing real work rather than being decorative, and the second ablation removes the topic-confound control to show that the tag-validity gate is load-bearing. The difficulty-matched control condition tests the additive-difficulty assumption behind the signed gap. The primary metric — agreement between certificate status and the expert-rating label — is the right one because it directly measures whether the discriminant verdict tracks the construct, not whether our pipeline runs.

### Expected outcome and causal chain

**vs. InclusionMed structured audit** — On a benchmark whose items are reproducible and well-provenanced but whose tags are assigned by surface topic (e.g., a radiology QA set tagged by organ rather than by reasoning demand), the structured audit produces a clean pass because provenance and reproduction succeed, while the benchmark's tag scores actually cluster by topic. Our method instead flags this as conditionally-certified or rejected because the Phase 0 topic-partial R² gate, the cross-loading bound, and the known-groups gap fail on the score matrix. We expect a noticeable disagreement concentrated on benchmarks with high provenance scores but low tag–capability coherence, and near-parity on benchmarks where topics and capabilities coincide.

**vs. Raw per-tag accuracy ranking** — On a case where a correlated capability tag c is simply harder than t*, raw accuracy ranks models by t* and shows a large apparent gap that looks like discriminant validity, but the gap is a difficulty artifact that flips when the panel shifts to a differently-skilled model set. Raw ranking produces a confident but unstable verdict because it never separates difficulty from capability. Our method instead requires a stable signed gap across bootstrap resamples, a difficulty-matched control condition, and a held-out congruence check, so we expect raw ranking to over-certify and CellVal to reject exactly those difficulty-confounded benchmarks, with the divergence concentrated on benchmarks whose tags differ mainly in item difficulty.

**vs. Correlation-threshold discriminant check** — On a case where t* and c are both hard, their per-model scores correlate strongly for benign difficulty reasons, so the correlation-threshold check rejects a benchmark that is actually construct-valid. Our method instead passes it because the known-groups bootstrap tests a signed gap rather than a raw correlation, absorbing the shared difficulty. We expect the correlation baseline to show false rejections on high-difficulty benchmarks while CellVal preserves them, and the two to agree on benchmarks where difficulty is matched across tags.

**vs. CellVal minus held-out congruence phase** — On a benchmark whose tag structure is overfit to the specific frontier models in the certification fold, the ablated version certifies it because in-fold loadings look clean, but the structure does not transfer. Our method instead downgrades it because the Tucker congruence against the held-out fold drops. We expect the ablation to over-certify on benchmarks tuned to frontier-model behavior, with the gap appearing only on the held-out-fold subset.

**vs. CellVal minus Phase 0 tag-validity gate and Phase 2(d) topic control** — On a topic-confounded benchmark whose tags are surface-topic proxies, the ablated version certifies it because the factor analysis recovers a topic factor that the target tag loads on. Our method instead rejects it as a tag–topic confound because the Phase 0 topic-partial R² exceeds the threshold and the target factor fails to survive with topic tags as nuisance columns. We expect the ablation to over-certify on topic-tagged benchmarks, with the gap concentrated on benchmarks whose tags are assigned by organ or modality rather than capability demand.

### What would falsify this idea

If CellVal's certificate status agrees with the pre-registered expert-rating label no better than the raw-accuracy or correlation-threshold baselines, or if the synthetic ground-truth study shows sensitivity/specificity indistinguishable from a generic factor-analysis application, or if the disagreements are uniform across benchmarks rather than concentrated on the predicted failure modes (topic-tagged, difficulty-confounded, and frontier-overfit benchmarks), then the factor-based discriminant test is not recovering a construct signal and the central claim is wrong.

## References

1. InclusionMed: Benchmarking AI for Inclusive Healthcare

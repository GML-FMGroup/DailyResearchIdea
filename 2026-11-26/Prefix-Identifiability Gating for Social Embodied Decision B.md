# Prefix-Identifiability Gating for Social Embodied Decision Benchmarks

## Motivation

Across the social embodied intelligence literature, latent social intent and norms are treated as directly supervised targets rather than as latent variables whose identifiability from partial observation must be established: AIRIS scores intent recognition by matching expert-curated labels, SOLAMI's SynMSI synthesizes paired multimodal intent data, and CogMir prompts LLM agents with social-science scenarios, all assuming the correct intent is recoverable from the prefix. This shared structural failure means a model can be penalized for waiting when the prefix genuinely does not determine whether intervention is warranted, conflating recognition error with irreducible ambiguity. The field therefore lacks a benchmark-level guarantee that the observation prefix identifies the latent norm up to the equivalence class that justifies intervening versus waiting.

## Key Insight

The set of observation prefixes consistent with an expert intent label forms an equivalence class, and only prefixes whose class is a singleton with respect to the intervene-versus-wait partition can legitimately carry a decision score; prefixes with non-singleton classes must be scored on calibrated abstention, which is a property of the label-conditional prefix distribution rather than of any model's confidence.

## Method

**What it is.** PIG (Prefix-Identifiability Gating) is a benchmark-side calibration procedure for AIRIS that partitions every time-anchored query into a *decidable* set (the observation prefix identifies the latent social intent up to the intervene/wait partition) and an *ambiguous* set (the prefix is consistent with multiple intent classes crossing that partition), using the expert-curated intent labels only as a calibration set, never as supervision for the evaluated model.

**Load-bearing assumption (stated explicitly).** PIG assumes that the frozen encoder phi maps observation prefixes of the same latent social intent to a unimodal, approximately Gaussian cluster in phi-space, so that the label-conditional density d_k(q) reflects latent-intent identifiability rather than scene/object/action context. This assumption is load-bearing: if phi entangles intent with scene context (e.g., a reaching motion encoded identically whether reaching to help or to take), then d_k(q) measures scene similarity, K_hat(q) under- or over-counts ambiguity, and the decidable/ambiguous partition is a property of phi rather than of the data distribution. PIG therefore does not assert that identifiability is encoder-independent; it treats the partition as encoder-conditional and reports encoder-induced variance as an explicit uncertainty band (Phase 2b below).

**How it works.**

```pseudocode
INPUT: AIRIS queries Q = {(clip c, time t, prefix x_{c,<=t}, expert intent label y, gold action a* in {wait, intervene, respond})}
       Calibration split C subset Q (disjoint from test split T; |C| = 1,600 queries, stratified over the 11 context categories)
       Encoders Phi = {phi_1, ..., phi_M} with M=3 frozen encoders:
           phi_1 = CLIP-ViT-B/32 image encoder + temporal mean over prefix frames
           phi_2 = CLIP-ViT-B/32 image encoder + temporal max over prefix frames
           phi_3 = supervised intent probe: linear layer (dim 512) trained on C only with cross-entropy on y, features from phi_1
       Equivalence-class radius rho (default: 95th percentile of intra-label prefix distance on C, computed per encoder)
       Decision-spread threshold tau (default 0.2)

PHASE 1 -- Label-conditional prefix geometry (calibration only, C; repeated per encoder phi_m in Phi):
  for each intent class k in expert label set K:
      mu_k^m  <- mean_{q in C, y_q=k} phi_m(x_q)
      Sigma_k^m <- empirical covariance of {phi_m(x_q) : y_q=k}, regularized by adding 1e-3 * I
      R_k^m   <- {q in C : y_q = k and ||phi_m(x_q)-mu_k^m||_{(Sigma_k^m)^{-1}} <= rho_m}
      kappa_k^m <- fraction of R_k^m whose gold action a* = intervene   # class-conditional intervention propensity
  # Calibration-validation diagnostics (reported, not used for gating):
  #   silhouette score per intent class in phi_m-space
  #   condition number of Sigma_k^m per class
  #   bootstrap (B=200 resamples of C) standard deviation of kappa_k^m

PHASE 2 -- Identifiability test per test query q in T (per encoder phi_m):
  d_k^m(q) <- ||phi_m(x_q)-mu_k^m||_{(Sigma_k^m)^{-1}}   for all k in K
  K_hat^m(q) <- { k in K : d_k^m(q) <= rho_m }        # prefix-consistent intent classes under phi_m
  if K_hat^m(q) is empty: mark q as OUT-OF-SUPPORT under phi_m
  elif max_{k in K_hat^m(q)} kappa_k^m - min_{k in K_hat^m(q)} kappa_k^m <= tau:
      mark q as DECIDABLE under phi_m with target action a*(q)
  else:
      mark q as AMBIGUOUS under phi_m; target = calibrated waiting (model must output 'wait' or explicit abstain)

PHASE 2b -- Encoder-consensus partition (the reported partition):
  For each q in T, collect the M per-encoder labels {DECIDABLE, AMBIGUOUS, OUT-OF-SUPPORT}.
  q is DECIDABLE iff all M encoders agree on DECIDABLE.
  q is AMBIGUOUS iff all M encoders agree on AMBIGUOUS.
  q is OUT-OF-SUPPORT iff all M encoders agree on OUT-OF-SUPPORT.
  Otherwise q is marked UNSTABLE and excluded from scoring; the fraction of UNSTABLE queries is reported as the encoder-induced variance band.

PHASE 3 -- Decision-aware scoring on T (using the Phase-2b consensus partition):
  score_decidable  <- accuracy of model action vs a* on DECIDABLE queries (premature commitment penalized as before in AIRIS)
  score_ambiguous  <- rate at which model outputs wait/abstain on AMBIGUOUS queries minus false-wait rate on DECIDABLE queries
  PIG_score        <- score_decidable - lambda * false_wait_rate_on_decidable (lambda=1.0)
  Report PIG_score alongside: fraction UNSTABLE, per-encoder PIG_score (to expose encoder sensitivity), and bootstrap 95% CI of PIG_score over 200 resamples of T.
```

**Why this design.** We chose a *label-conditional generative model of prefixes* (per-class mean and covariance in a frozen encoder space) over a learned classifier or a log-prob gate because the identifiability question is about the data distribution, not about any model's confidence; a classifier would collapse onto Anti-Pattern 3 (log-prob as calibrated gate) and would be unfalsifiable, whereas the label-conditional density directly yields the equivalence class. We chose a single fixed radius rho calibrated on a held-out split over per-query adaptive thresholds because a global radius makes the decidable/ambiguous partition auditable and reproducible across the 45 evaluated models, accepting the cost that heterogeneous classes with unequal variance are only partially corrected by the per-class Sigma_k^{-1} whitening. We chose to *exclude* out-of-support queries rather than force a decision because forcing a decision on a prefix with no consistent intent class would reintroduce the very label-matching failure AIRIS exhibits (E4), accepting the cost of a smaller scored test set. We chose the kappa_k intervention-propensity spread (max-min) as the partition criterion over a raw entropy-over-classes criterion because the decision boundary is intervene-versus-wait, not intent-class identity, so two intent classes with identical kappa_k are decision-equivalent and should not count as ambiguity. We chose a three-encoder consensus (Phase 2b) over a single frozen encoder because the load-bearing assumption above may fail for any one encoder; requiring agreement across CLIP-mean, CLIP-max, and a supervised probe trained on C only makes the reported partition robust to encoder-specific entanglement, and reporting the UNSTABLE fraction makes the encoder-induced variance explicit rather than hidden. This is not a variant of AIRIS's intention-recognition dimension: AIRIS scores label matching, whereas PIG scores only queries whose label is identifiable from the prefix, and the closest named prior is standard selective-prediction / conformal abstention, which calibrates a *model's* abstention on labeled data; PIG instead calibrates the *benchmark's* decidable subset using the label-conditional prefix geometry, so no model is trained or tuned on the calibration split.

**Why it measures what we claim.** The quantity d_k(q) = ||phi(x_q)-mu_k||_{Sigma_k^{-1}} measures prefix-consistency with intent class k because we assume the frozen encoder phi maps prefixes of the same latent intent to a unimodal, approximately Gaussian cluster whose 95th-percentile radius rho captures within-class variation; this assumption fails when the encoder entangles intent with scene context (e.g., a person reaching is encoded similarly whether reaching to help or to take), in which case d_k reflects scene similarity rather than intent identifiability. The equivalence class K_hat(q) measures the set of intents identifiable from the prefix because we assume every intent class consistent with the prefix lies within rho of its class mean; this fails when the true intent is a mixture not represented by any single class mean, in which case K_hat(q) under-counts ambiguity. The spread max_kappa - min_kappa measures decision-relevant ambiguity because we assume kappa_k estimated on the calibration split is a stable class-conditional intervention propensity; this fails when the calibration split under-represents a context category (AIRIS has 11), in which case the spread reflects sampling noise rather than genuine decision ambiguity. The false-wait rate on DECIDABLE queries measures premature-commitment calibration because we assume the decidable set is exactly the set on which waiting is suboptimal by construction; this fails when a* itself was annotated under ambiguity, in which case false-wait reflects label noise rather than model miscalibration. The Phase-2b UNSTABLE fraction measures encoder-induced partition variance because we assume that agreement across CLIP-mean, CLIP-max, and a supervised probe is evidence that the partition reflects the data distribution rather than any single encoder's geometry; this fails when all three encoders share the same entanglement (e.g., all are CLIP-derived), in which case UNSTABLE under-reports encoder sensitivity and the partition remains encoder-conditional.

## Contribution

['A benchmark-level identifiability gating procedure (PIG) that partitions AIRIS queries into decidable and ambiguous sets using label-conditional prefix geometry, converting the intention-recognition dimension from label matching into an identifiability-guaranteed decision problem.', 'A decision-aware scoring rule that separates premature-commitment error on decidable queries from calibrated-abstention credit on ambiguous queries, replacing the single accuracy number with a two-sided score that penalizes both premature intervention and unnecessary waiting.', 'An audit of the AIRIS 45-model evaluation under PIG, reporting how much of the previously reported fragmented competence is attributable to irreducible prefix ambiguity rather than model error.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AIRIS test split (held-out from PIG calibration; |C|=1,600 calibration queries, stratified over 11 context categories) | Time-anchored social queries with expert intent and action labels |
| Primary metric | PIG_score (decidable accuracy minus false-wait rate) with bootstrap 95% CI over 200 resamples of T | Single number capturing both commitment and restraint |
| Baseline | AIRIS raw intention-recognition scoring | Tests whether label-matching alone rewards unidentifiable queries |
| Baseline | Log-prob confidence gate (selective prediction) calibrated on the same frozen encoder features | Tests whether model-internal confidence matches prefix identifiability |
| Baseline | Always-wait / always-intervene heuristic | Tests whether PIG partition is trivially separable |
| Ablation-of-ours | PIG with entropy-over-classes criterion (no kappa spread) | Tests whether decision-relevant partition beats intent-identity partition |
| Ablation-of-ours | PIG with single encoder phi_1 only (no Phase-2b consensus) | Tests whether multi-encoder consensus changes the partition |
| Diagnostic | Per-class silhouette score, covariance condition number, and bootstrap std of kappa_k on C | Tests the load-bearing encoder-disentanglement assumption |
| Diagnostic | Encoder-disentanglement probe: scene-context-matched but intent-different prefix pairs, measuring same-cluster rate | Tests whether d_k reflects intent rather than scene |

### Why this setup validates the claim

The central claim is that a benchmark-side, label-conditional prefix geometry can separate queries whose social intent is identifiable from those where intervention vs. waiting is genuinely underdetermined, and that scoring only on the identifiable set produces a more diagnostic ranking of the 45 AIRIS models than raw label matching. The raw AIRIS scoring baseline tests the sub-claim that label matching conflates identifiable and ambiguous prefixes (E4's fragmented, temporally miscalibrated competence); if PIG reorders models relative to this baseline specifically on the ambiguous subset, the partition is doing work. The log-prob confidence gate tests the competing hypothesis that a model's own confidence already tracks identifiability — if it did, PIG's calibration split would be redundant, so a gap between the two isolates the value of benchmark-side geometry over model-side confidence. The always-wait/always-intervene heuristic guards against the degenerate possibility that the decidable/ambiguous split is trivially predictable from action base rates. The entropy-criterion ablation isolates whether the decision-relevant kappa-spread criterion — rather than any partition at all — is what produces the gain. The single-encoder ablation and the encoder-disentanglement probe directly test the load-bearing assumption: if the single-encoder and consensus partitions coincide and the probe shows intent-different prefixes fall in the same cluster, the partition reflects scene similarity rather than intent identifiability and the central claim is falsified. PIG_score is the right primary metric because it jointly penalizes premature commitment on decidable queries and unjustified waiting on ambiguous ones, which is exactly the trade-off AIRIS's decision-aware metrics were designed to expose (E1, E4); a metric that only measured accuracy or only measured abstention would be satisfiable by a degenerate policy.

### Expected outcome and causal chain

**vs. AIRIS raw intention-recognition scoring** — On a case where a person's reaching motion is consistent with both helping and taking (entangled encoder features), the baseline produces a confident wrong intent label and scores it as a miss, because it treats label matching on every query as the target. Our method instead marks that query AMBIGUOUS via a wide kappa spread across the prefix-consistent classes and scores waiting/abstention as correct, because the decision boundary is intervene-versus-wait rather than intent identity. We expect a noticeable PIG_score gain concentrated on the ambiguous subset and near-parity on the decidable subset, with model rankings shifting most for embodied specialists that over-commit.

**vs. Log-prob confidence gate (selective prediction)** — On a case where the prefix is genuinely ambiguous but the model's log-prob is sharply peaked (hallucinated certainty, as in E2's prosocial-irrationality finding), the baseline produces a high-confidence intervention and its gate admits the query as decidable, because model confidence is calibrated to training bias rather than to prefix identifiability. Our method instead excludes that query from the decidable set using label-conditional density in frozen encoder space, because identifiability is a property of the data distribution, not of any model. We expect the confidence gate to show a much smaller decidable/ambiguous separation and to mis-rank models whose confidence is systematically overconfident, yielding a visible PIG_score gap that grows with model overconfidence.

**vs. Always-wait / always-intervene heuristic** — On a case where a decidable query clearly calls for intervention, the always-wait heuristic produces a false wait, and on an ambiguous query the always-intervene heuristic produces a premature commitment, because neither has any notion of identifiability. Our method instead assigns each query to the set where its action is well-defined, so the decidable set rewards intervention and the ambiguous set rewards restraint. We expect both heuristics to score near zero on PIG_score while PIG-scored models retain substantial positive scores, confirming the partition is non-degenerate and that the metric is not satisfiable by a constant policy.

**vs. PIG with entropy-over-classes criterion (ablation)** — On a case where two intent classes have identical kappa_k but different identities, the entropy criterion marks the query AMBIGUOUS while the kappa-spread criterion marks it DECIDABLE, because the decision boundary is intervene-versus-wait rather than intent identity. We expect the entropy ablation to produce a lower PIG_score and a partition that disagrees with the kappa-spread partition on a measurable fraction of queries, isolating the value of the decision-relevant criterion.

**vs. PIG with single encoder phi_1 only (ablation)** — On a case where CLIP-mean entangles intent with scene context, the single-encoder partition marks a scene-similar but intent-different query as DECIDABLE, while the Phase-2b consensus marks it UNSTABLE or AMBIGUOUS. We expect the single-encoder PIG_score to differ from the consensus PIG_score by an amount that bounds the encoder-induced variance, and we report the UNSTABLE fraction as the explicit uncertainty band.

### What would falsify this idea

If PIG's gain over raw AIRIS scoring is uniform across the decidable and ambiguous subsets rather than concentrated on the ambiguous subset where the predicted identifiability failure occurs, or if the log-prob confidence gate matches PIG's decidable/ambiguous separation, or if the encoder-disentanglement probe shows intent-different but scene-matched prefixes fall in the same cluster (so d_k reflects scene similarity), then benchmark-side prefix geometry is not measuring latent-intent identifiability and the central claim is wrong.

## References

1. AIRIS: Benchmarking Social Embodied Intelligence in Egocentric Streaming Videos

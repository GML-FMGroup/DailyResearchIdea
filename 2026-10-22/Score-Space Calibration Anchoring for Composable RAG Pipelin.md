# Score-Space Calibration Anchoring for Composable RAG Pipelines

## Motivation

Compositional RAG evaluation frameworks such as LASAGNA sweep hundreds of pipeline configurations and observe that retrieval-quality gains are non-additive, yet they treat the embedding model as a frozen black box whose relevance scores are assumed stable across recomposition. This assumption fails structurally: changing chunking granularity or inserting a critic alters the score distribution, so any downstream threshold or abstention rule tuned on one configuration silently miscalibrates on another. Prior work on retrieval granularity (Dense X Retrieval) and adaptive retrieval (When Not to Trust Language Models) likewise assumes scores are comparable across pipeline choices, and neither offers a label-free mechanism to restore or certify calibration after recomposition.

## Key Insight

Recomposition-induced score shifts are not arbitrary distortions but monotone bijections of the relevance score axis, so a single per-configuration scalar — estimable from unlabeled scores via a rank/quantile matching identity — restores the entire score distribution to a reference calibration, making thresholds and abstention decisions transferable across configurations without per-configuration labels.

## Method

```pseudocode
# SCAN: Score Calibration ANchoring for RAG pipelines
# Inputs: reference configuration C0 with labeled calibration set D0 = {(q, s, y)} of size 500;
#         target configuration Ct with unlabeled scores St = {s_i} on a probe query set Q_probe of size 300;
#         downstream threshold tau_0 (retrieval cutoff) and abstention threshold alpha_0 tuned on C0.
# Output: anchored scores for Ct, transferred thresholds (tau_t, alpha_t), and a certification flag.

1. Fit reference calibration map on C0:
   - Estimate empirical CDF F0 of C0's relevance scores on D0.
   - Fit isotonic regression g0: score -> P(correct | score) on D0 (monotone by construction).
   - Record anchor a0 = median(F0) and reference quantile grid Q0 = F0^{-1}([0.05,0.10,...,0.95]).

2. Estimate target anchor a_t from unlabeled probe scores St:
   - Compute empirical CDF Ft of St on Q_probe.
   - Solve for scalar a_t such that the quantile-matched identity holds:
       min_{a>0}  sum_{k} | Ft^{-1}(p_k) - a * F0^{-1}(p_k) |   over p_k in Q0
     (closed-form: a_t = median_k( Ft^{-1}(p_k) / F0^{-1}(p_k) )).
   - This is the rank/quantile matching identity: under a monotone bijection that is a pure scale change,
     the quantile ratios are constant; a_t is that constant. (Load-bearing assumption: recomposition
     transforms the score axis by a single global positive scalar; if the transform is affine or
     non-monotone, the quantile ratios vary with p_k and a_t is only a best scalar approximation.)

3. Anchor target scores: s_hat = s / a_t  for every s in Ct's score stream.

4. Transfer thresholds:
   - tau_t = a_t * tau_0 ;  alpha_t = a_t * alpha_0.
   - Recompute g_t(score) = g0(score / a_t) as the target calibration map.

5. Certification test (label-free):
   - Split Q_probe in half; estimate a_t on half A, test on half B.
   - Compute Kolmogorov-Smirnov statistic between anchored B scores and F0.
   - If KS < epsilon (epsilon = 0.05), set certified = True; else certified = False
     and fall back to a per-configuration isotonic refit on a small labeled set of 100 queries.

6. Return (s_hat, tau_t, alpha_t, g_t, certified).
```

**Why this design.** We chose a single scalar anchor over a full per-configuration recalibration because the structural claim is that recomposition induces a monotone bijection, not an arbitrary distortion; a full refit would require labels the target configuration does not have, whereas the scalar is estimable from unlabeled probe scores, accepting the cost that if the bijection assumption is violated (e.g., the critic introduces a non-monotone score transformation), the anchor will partially fail and certification will flag it. We chose quantile matching over moment matching (mean/variance) because relevance scores are heavy-tailed and moments are unstable under chunking granularity changes, accepting the cost that quantile estimation requires a probe set of at least a few hundred queries (we use 300). We chose isotonic regression over Platt scaling for the reference map because isotonic makes no parametric assumption about the score-correctness relationship, accepting the cost of needing a labeled reference set of moderate size (we use 500). A domain expert would not describe SCAN as a variant of LASAGNA: LASAGNA measures composition effects but never operates on the score distribution itself, whereas SCAN introduces a reparameterization operator and a label-free certification test that LASAGNA's sweep design does not contain.

**Why this measures what we claim.** The anchor a_t measures the scale of the monotone reparameterization induced by recomposition because the rank/quantile matching identity assumes that the target score distribution is a scalar multiple of the reference on the quantile axis; this assumption fails when the critic inserts a non-monotone score transform (e.g., a thresholded critic that clips low scores), in which case a_t reflects the best scalar approximation and the KS certification statistic will exceed epsilon, correctly flagging non-invariance. The certification KS statistic measures calibration invariance because it assumes the probe query set is drawn from the same query distribution as deployment; this assumption fails under distribution shift in queries, in which case the KS statistic reflects probe-set mismatch rather than score-space distortion. The transferred threshold tau_t measures the preservation of retrieval cutoff semantics because it assumes the reference threshold tau_0 was tuned on a calibrated score axis; this assumption fails if the reference configuration itself was miscalibrated, in which case tau_t inherits that miscalibration.

## Contribution

(1) A score-space calibration anchoring operator (SCAN) that defines a monotone bijection induced by chunking granularity and critic insertion and estimates a single per-configuration scalar anchor from unlabeled scores via a rank/quantile matching identity. (2) A label-free certification test based on a held-out KS statistic that flags when the monotone-bijection assumption is violated, enabling safe fallback to per-configuration refitting. (3) A threshold-transfer procedure that maps retrieval cutoffs and abstention thresholds across pipeline reconfigurations without per-configuration labels, demonstrated on LASAGNA's existing 450-configuration sweep.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LASAGNA 450-config sweep + Dense X granularity suite | Recomposition and granularity shifts are exactly the monotone-bijection triggers |
| Primary metric | KS certification pass rate vs. realized calibration error (ECE) | Directly tests label-free invariance claim, not just accuracy |
| Baseline | Per-configuration isotonic refit (labeled) | Upper bound requiring labels SCAN avoids |
| Baseline | Unanchored threshold transfer (tau_0 reused) | Tests whether scalar anchoring is doing the work |
| Baseline | LASAGNA additive-effect prediction | Tests whether score-space operator beats composition measurement |
| Ablation-of-ours | SCAN without certification (always accept) | Isolates the falsification-flagging role |

### Why this setup validates the claim

The central claim is that recomposition induces a monotone reparameterization of the relevance score axis, so a single scalar anchor estimable from unlabeled probe scores recovers threshold semantics and calibration without labels. The LASAGNA 450-configuration sweep is the right dataset because it varies the composition of RAG techniques while holding the underlying scorer family roughly fixed — precisely the regime where a monotone bijection should hold and where a scalar should suffice. The Dense X granularity suite stresses the assumption from the other direction: chunking changes alter the score distribution scale, which is what the anchor is designed to absorb. The primary metric, KS certification pass rate paired with realized calibration error, is chosen because accuracy alone cannot distinguish "the anchor worked" from "the threshold happened to be lucky"; the KS statistic is the same quantity the method uses internally, so agreement between predicted certification and realized calibration error is a direct test of whether the certification logic is sound. The labeled per-configuration isotonic refit is the upper bound: if SCAN approaches it on certified configurations, the label-free claim holds. The unanchored threshold transfer baseline tests whether the scalar is doing real work or whether thresholds transfer trivially. The LASAGNA additive baseline tests whether operating in score space beats measuring composition effects. The no-certification ablation tests whether the flag is informative or merely decorative.

### Expected outcome and causal chain

**vs. Per-configuration isotonic refit (labeled)** — On a case where the target configuration recomposes a reranker with a different score scale, the baseline produces a well-calibrated map because it uses target labels to fit a fresh monotone function. Our method instead estimates a single anchor from unlabeled probe quantiles and rescales the reference map because it exploits the monotone-bijection structure. We expect near-parity on certified configurations (KS below epsilon) and a visible gap only on the uncertified minority where the bijection is violated — the gap should be concentrated, not uniform.

**vs. Unanchored threshold transfer (tau_0 reused)** — On a case where recomposition rescales the critic's output by a factor of roughly two, the baseline produces a retrieval cutoff that admits far too many or too few chunks because it assumes the score axis is unchanged. Our method instead divides scores by the quantile-matched anchor and rescales tau_0 accordingly because the anchor recovers the multiplicative distortion. We expect a large gap in retrieval precision/recall at fixed budget on rescaled configurations and near-parity on configurations whose scale is unchanged.

**vs. LASAGNA additive-effect prediction** — On a case where two individually helpful techniques interact non-additively, the baseline produces a predicted combined gain well above the measured one because it models composition as additive on the performance axis rather than as a reparameterization of the score axis. Our method instead detects the score-space distortion via the anchor and flags non-invariance when the KS statistic exceeds epsilon because it operates on the distribution itself. We expect SCAN's certification to correlate with the sign of the additive-prediction error, concentrated on configurations LASAGNA mispredicts most.

### What would falsify this idea

If SCAN's calibration error gap versus the labeled refit is uniform across certified and uncertified configurations rather than concentrated on the uncertified ones, the certification test is not detecting the monotone-bijection violation it claims to detect. Likewise, if the anchor provides no advantage over unanchored threshold transfer on configurations where the score scale demonstrably changes, the scalar-reparameterization claim is wrong.

## References

1. Retrieval Augmented Generation Compositionality
2. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
3. Dense X Retrieval: What Retrieval Granularity Should We Use?
4. Few-shot Learning with Retrieval Augmented Language Models
5. Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models
6. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories

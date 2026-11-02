# Post-Processing-Invariant Reconstruction-Error Recalibration for Unified Forgery Detection

## Motivation

Venus-DefakerOne2 and similar reconstruction-based forensic foundation models assume that low reconstruction error is a sufficient witness of authenticity, yet this sufficiency silently breaks under benign post-processing such as JPEG compression and resampling, which raise reconstruction error on pristine content and can lower it on forged content. The root cause is structural: reconstruction error is measured in a pixel/feature space that is not invariant to the post-processing operator, so a decision boundary tuned on clean data (as in Entropy-based Test Adaptation) cannot restore sufficiency because the witness itself is corrupted, not merely shifted.

## Key Insight

Reconstruction error becomes a sufficient forgery witness only when it is re-expressed through a post-processing-invariant null distribution estimated from the test image itself, so that the same error magnitude means the same thing regardless of the benign operator applied.

## Method

**What it is.** PIRE (Post-processing-Invariant Reconstruction-Error recalibration) takes as input a test image $x$, the forensic model's reconstruction $\hat{x}=D(E(x))$, and a bank of benign post-processing operators $\mathcal{T}=\{\text{JPEG}_{q},\text{resample}_{s}\}$; it outputs a recalibrated witness score $w(x)\in[0,1]$ and a binary forgery decision, replacing the raw error $\|x-\hat{x}\|$ with a score whose null distribution is conditioned on the post-processing evidence in $x$.

**How it works.**

```pseudocode
Input: x, reconstruction x_hat, operator bank T, quantile alpha=0.95, bins B=8
1. e_raw  = ||x - x_hat||_2^2                      # raw reconstruction error
2. For each T_i in T:  e_i = ||T_i(x) - T_i(x_hat)||_2^2   # operator-commuted errors
3. Estimate post-processing fingerprint p(x):
     p_j = fraction of 8x8 blocks whose DCT high-frequency energy ratio
           falls in bin j  ->  p(x) in R^B     (JPEG/resample leave distinct
           high-frequency signatures; no learned head)
4. Build null calibration set by applying every T_i to a held-out clean
   reference set R (|R|=2000), recording (e_raw, p) pairs -> null table
   N[(p-bin, T_i)] = empirical CDF of e_raw under operator T_i and
   fingerprint bin p-bin.
5. At test time, locate p(x)'s bin, take the operator-averaged null CDF
   F_null(e_raw | p-bin) = mean_i N[(p-bin, T_i)](e_raw).
6. w(x) = 1 - F_null(e_raw | p-bin)                # recalibrated witness
7. Decision: forgery if w(x) >= alpha (alpha=0.95), else authentic.
Output: w(x), decision
```

**Load-bearing assumption (explicit).** The fingerprint bin $p(x)$ is assumed to be a sufficient statistic for the conditional distribution of $e_{\text{raw}}$ under benign post-processing: images sharing a fingerprint bin are assumed to share the same null error CDF. This assumption is load-bearing because the entire recalibration routes the test image to a null CDF estimated under the same operator; if the bin conflates content with operator, the null mixes operators with different error scales and $w(x)$ reflects bin misassignment rather than forgery evidence. The assumption fails when double compression or an unseen denoiser maps to a bin whose null was estimated from different operators.

**Why this design.** We chose an operator-commuted null table over a learned recalibration head because the sufficiency failure is a distributional shift caused by a known, enumerable operator family, so the null can be measured rather than learned, accepting the cost that operators outside $\mathcal{T}$ are not covered. We chose a DCT high-frequency fingerprint over a CNN post-processing classifier because the fingerprint is deterministic and requires no training data, accepting that it coarsely bins images (B=8) and may confuse unusual but benign capture pipelines. We chose to average null CDFs across operators rather than select the single most likely operator because operator identity is often ambiguous after double compression, accepting a small loss of power when one operator dominates. This is not a variant of Entropy-based Test Adaptation (Venus-DefakerOne2): that method re-weights prediction modalities by entropy at the decision boundary, whereas PIRE changes the witness itself by conditioning its null distribution on post-processing evidence — a domain expert would not describe a null-distribution recalibration as a boundary-tuning variant.

**Why it measures what we claim.** The computational quantity $w(x)=1-F_{\text{null}}(e_{\text{raw}}\mid p\text{-bin})$ measures *reconstruction-error sufficiency under post-processing* because it equals the probability that a clean image with the same post-processing fingerprint would produce an error at least as large, under the assumption that the operator-commuted null table faithfully represents the benign post-processing distribution for that fingerprint bin; this assumption fails when an unseen operator (e.g., a novel denoiser) produces a fingerprint falling into a bin whose null was estimated from different operators, in which case $w(x)$ reflects the mismatch between the test operator and the calibration bank rather than forgery evidence. The operator-commuted errors $e_i$ measure *post-processing invariance* because they quantify how much the reconstruction error moves when the benign operator is applied to both image and reconstruction, under the assumption that a genuine forgery witness should be stable under such commuting; this assumption fails when the operator is non-commutative with the reconstruction (e.g., strong resampling that destroys high-frequency forgery traces), in which case $e_i$ reflects operator-induced distortion rather than witness stability.

## Contribution

['A post-processing-invariant recalibration module (PIRE) that replaces raw reconstruction error with a null-distribution-conditioned witness score, making the sufficiency assumption of reconstruction-based forensic models explicit and testable.', "A design principle: forgery witnesses in reconstruction-based detectors should be calibrated against an operator-commuted null estimated from the test image's own post-processing fingerprint, not against a global clean-data threshold.", 'An operator bank and DCT-fingerprint binning scheme that requires no additional training data and can be attached to any reconstruction-based forensic foundation model.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Forgery benchmarks with benign post-processing variants | Tests null shift where operators actually occur |
| Primary metric | AUC on post-processed forgery subsets | Measures ranking under the predicted distribution shift |
| Baseline | Raw reconstruction error | Isolates whether recalibration, not error, drives gains |
| Baseline | Venus-DefakerOne2 | Tests against strong forensic foundation model |
| Baseline | Entropy-threshold adaptation | Tests whether entropy boundary-tuning suffices |
| Baseline | Conformal prediction on raw error (no fingerprint) | Tests whether fingerprint conditioning adds beyond standard calibration |
| Baseline | Learned post-processing classifier feeding same null table | Tests whether deterministic fingerprint suffices vs learned classifier |
| Ablation-of-ours | PIRE without operator commutation | Isolates contribution of operator-commuted null |
| Ablation-of-ours | PIRE with per-bin coverage test (calibration validity) | Directly tests fingerprint-bin sufficiency assumption |

### Why this setup validates the claim

The central claim is that reconstruction-error sufficiency fails specifically under benign post-processing, and that conditioning the null on post-processing evidence restores it. The dataset must therefore contain the same forgery types under both clean and post-processed conditions, so that the effect can be localized rather than confounded with forgery difficulty. Raw reconstruction error tests whether any recalibration is needed at all: if it already ranks post-processed forgeries correctly, PIRE's mechanism is unnecessary. Venus-DefakerOne2 tests the stronger alternative that a broad foundation model already absorbs post-processing shifts, which would make a hand-built null table redundant. Entropy-threshold adaptation tests the specific rival mechanism—boundary reweighting—that PIRE explicitly disclaims; if it matches PIRE, the claimed distinction collapses. The conformal prediction baseline calibrates the same reconstruction error on a held-out clean set without fingerprint conditioning, testing whether the fingerprint-conditioned null is merely a re-derivation of standard calibration. The learned post-processing classifier baseline feeds the same null table but replaces the deterministic DCT fingerprint with a trained classifier, testing whether the fingerprint's coarse binning loses critical operator information. The no-commutation ablation removes operator commutation while keeping the fingerprint-conditioned null, so any gain attributable to commuting errors is separated from the gain from fingerprint binning. The per-bin coverage test directly measures the empirical coverage of the null CDF on held-out clean images under each operator (target 95% at alpha=0.95) and reports bins where coverage deviates, testing the faithfulness assumption rather than assuming it. AUC on post-processed subsets is the right metric because the claim is about ranking clean versus forged images under a shifted null, not about a fixed threshold, and it directly exposes whether the null recalibration restores separability where it was lost.

### Expected outcome and causal chain

**vs. Raw reconstruction error** — On a case where a forged region is JPEG-recompressed at quality 70, raw error shrinks because compression smooths the splice boundary, so the baseline scores the forgery near the clean distribution and ranks it below many authentic images. Our method instead conditions the null on the high-frequency fingerprint bin that compression produces, so the same small error is compared against a null of equally compressed clean images, and the witness rises. We expect a noticeable AUC gap on post-processed subsets and near-parity on clean subsets, with the gap growing as compression strength increases.

**vs. Venus-DefakerOne2** — On a case where a forgery is followed by resampling that displaces high-frequency forensic traces, the foundation model's learned features drift toward the authentic manifold because its training distribution under-represents that operator composition, yielding confident authentic predictions. Our method instead measures the operator-commuted error directly, so resampling applied to both image and reconstruction cancels common distortion, and the fingerprint bin routes the case to a null estimated under the same resampling. We expect PIRE to close most of the gap on heavily post-processed subsets while the foundation model retains an edge on clean, in-distribution forgeries.

**vs. Entropy-threshold adaptation** — On a case where a forgery sits near the decision boundary after mild recompression, entropy reweighting adjusts the threshold but leaves the underlying error statistic shifted, so the case is reclassified by boundary movement rather than by a corrected null, producing unstable decisions across operator strengths. Our method instead replaces the statistic with a calibrated tail probability, so the decision threshold is fixed at alpha and the score itself moves with the post-processing evidence. We expect PIRE to show a flatter error-versus-operator-strength curve, with entropy adaptation degrading where the shift is largest.

**vs. Conformal prediction on raw error (no fingerprint)** — On a case where a forged region is JPEG-recompressed at quality 70, conformal prediction calibrates the raw error on a held-out clean set without conditioning on post-processing, so the null distribution is a mixture over operators and the small error of the compressed forgery is compared against a null that includes uncompressed clean images with larger errors, yielding a conservative witness that under-detects. Our method instead conditions the null on the fingerprint bin, so the same error is compared against a null of equally compressed clean images, and the witness rises. We expect PIRE to outperform conformal prediction on post-processed subsets, with the gap growing as compression strength increases, while both methods perform similarly on clean subsets.

**vs. Learned post-processing classifier feeding same null table** — On a case where a forgery is followed by resampling that displaces high-frequency forensic traces, the learned classifier may misclassify the operator due to content entanglement, routing the case to a null estimated under a different operator and producing a miscalibrated witness. Our method uses a deterministic DCT fingerprint that, while coarse, is not confounded by content in the same way, and the operator-averaged null provides robustness to misclassification. We expect PIRE to match or exceed the learned classifier baseline on post-processed subsets, with the learned classifier showing higher variance across operator strengths due to classification errors.

**vs. PIRE without operator commutation** — On a case where a forgery is followed by resampling, the no-commutation ablation uses only the raw error conditioned on the fingerprint bin, so the null is estimated from clean images under the same operator but without commuting the operator through the reconstruction. This means the null reflects the distribution of raw errors under the operator, not the operator-commuted errors, and may not capture the operator's effect on the reconstruction error. Our method commutes the operator through both image and reconstruction, so the null is estimated from operator-commuted errors, which better isolates the forgery signal. We expect full PIRE to outperform the no-commutation ablation on post-processed subsets, with the gap growing as operator strength increases.

**vs. PIRE with per-bin coverage test** — This is not a baseline but a calibration validity check. We expect the per-bin coverage of the null CDF on held-out clean images to be close to 95% at alpha=0.95 for bins where the fingerprint is informative, and to deviate for bins where content dominates or double compression occurs. We will report coverage per bin and operator, and identify bins with coverage below 90% as failure modes of the fingerprint-bin sufficiency assumption.

### What would falsify this idea

If PIRE's AUC gain over raw reconstruction error is uniform across clean and post-processed subsets rather than concentrated on the post-processed subsets where the predicted null shift occurs, the central claim is wrong. Likewise, if the no-commutation ablation matches full PIRE, the operator-commuted null is not doing the work we attribute to it. Additionally, if the per-bin coverage test shows that the null CDF does not achieve nominal coverage on held-out clean images for most bins, the fingerprint-bin sufficiency assumption is violated and the recalibration claim collapses.

## References

1. Venus-DefakerOne2: A Vision-Forensics Foundation Model with Broader Coverage

# Calibration-Latent Metric Depth for Binocular Vision-Language Models

## Motivation

Existing 3D vision-language models treat camera geometry as an input rather than an inference target: StereoVLM's calibration-conditioned metric depth head is only identifiable when intrinsics and extrinsics are supplied, and its abstract frames the whole pipeline around calibrated binocular observations. This is a structural bottleneck across the field — UniVLG builds on 2D-to-3D lifting and VG-LLM extracts geometry priors from video, but neither recovers calibration jointly with metric scale, so metric 3D reasoning collapses whenever calibration is unknown, degraded, or absent. The root cause is that calibration enters the architecture as a hard precondition rather than a latent variable that can be marginalized, making the metric-scale parameter unidentifiable exactly in the uncalibrated regime where embodied systems operate.

## Key Insight

Treating calibration as a latent variable with an explicit prior and marginalizing it through an uncertainty-aware disparity likelihood makes metric scale identifiable from binocular evidence alone, because the focal-length/scale ambiguity becomes a constrained posterior coupling rather than an unconstrained product.

## Method

**What it is.** CaliLaDepth (Calibration-Latent Depth) is a binocular vision-language model that takes a stereo pair (and optional partial calibration) and outputs metric depth, a posterior over intrinsics/extrinsics, and a metric-scale factor, all consumed by a language head for spatial reasoning. It replaces StereoVLM's calibration-conditioned depth head with a probabilistic joint inference module in which calibration is a latent variable. **Load-bearing assumption (stated explicitly):** the focal-length/baseline (scale) ambiguity in stereo is resolved by marginalizing a learned diagonal-Gaussian calibration posterior through a disparity likelihood, i.e., the stereo pair plus a broad prior constrains the focal-times-baseline product enough for the posterior to concentrate near the true calibration rather than trading off scale against disparity in a flat, unidentifiable ridge. This assumption is scale-breaking only because the prior is not scale-invariant: we use a learned absolute-metric prior over scene scale (camera height and object-size statistics) that enters the likelihood as a metric anchor, and we treat baseline as known/measured (the common embodied case) so that only focal length is latent. The identifiable quantity is the product f*b; we report posterior concentration along the f*b direction as the primary identifiability diagnostic rather than per-parameter sigma_c.

**How it works.**

```pseudocode
INPUT: left image I_L, right image I_R, optional partial calib c_partial (baseline b known/measured; focal f latent)

Phase 1 — Geometry encoder (shared with VLM backbone):
  F_L, F_R = StereoViT(I_L), StereoViT(I_R)          # ViT-B/16, patch 16, 12 layers, 768-d
  D_prior  = DisparityHead([F_L, F_R])               # dense disparity, no calib; 4-level conv decoder, channels 256->128->64->1

Phase 2 — Calibration-latent posterior (focal only; baseline fixed):
  mu_f, log_sigma_f = CalibEncoder([F_L, F_R, c_partial])   # 2-layer MLP, hidden=256, GeLU; diagonal Gaussian q(f|I)
  f_samples = mu_f + sigma_f * eps,  eps~N(0,I),  K=8 reparameterized draws
  # Prior p(f): log-normal with mean log(f0) and std 0.3, where f0 is the training-set median focal; this prior is scale-breaking because it is not invariant under f -> s*f.

Phase 3 — Uncertainty-aware disparity likelihood:
  for k in 1..K:
     d_k = DisparityFromGeometry(f_samples[k], b, F_L, F_R)     # uses f_k, fixed b
     log_w_k = log p(D_prior | d_k, sigma_d)                    # heteroscedastic sigma_d head: 3-layer conv, channels 64->64->1, softplus output, variance floor 1e-4
  D_metric = sum_k softmax(log_w_k) * (f_k * b / (d_k + eps))   # metric depth per sample
  scale_s  = MetricScaleHead(D_metric)                          # residual log-scale correction; 2-layer MLP, hidden=128

Phase 4 — Language head:
  tokens = [F_L pooled, D_metric tokens, mu_f, scale_s]
  answer = LLM(tokens)                                       # spatial QA / grounding; LLM is frozen 7B decoder, LoRA rank 16

LOSS = L_disp_uncertainty + beta*KL(q(f|I) || p(f)) + L_metric + L_lang
  L_disp_uncertainty = sum_k w_k * Huber(D_prior, d_k) / (2*sigma_d^2) + log sigma_d
  beta scheduled 0.01 -> 0.1 over training (linear over first 10k steps); p(f) = log-normal(mean=log f0, std=0.3)
```

**Why this design.** We chose a diagonal-Gaussian calibration posterior over a point-estimate calibration head because marginalizing K=8 draws through the disparity likelihood is what makes the focal-length/scale product constrained rather than free, accepting the cost of K-fold depth computation at train time. We chose disparity supervision as the primary geometric signal over direct metric-depth supervision because StereoVLM reports disparity supervision generalizes substantially better than direct binocular depth, accepting that metric scale must then be recovered by a separate residual head. We chose a heteroscedastic per-pixel sigma_d head over a fixed loss weight because the calibration posterior's spread must modulate the disparity likelihood for the marginalization to be meaningful, accepting extra parameters and a risk of sigma_d collapse that we counter with a variance floor of 1e-4. We chose a KL term with a scheduled beta over joint end-to-end training of calibration and depth because unconstrained joint training lets the scale head absorb calibration error, accepting slower early convergence. A domain expert would not describe this as 'StereoVLM plus a Bayesian wrapper': StereoVLM conditions on known calibration and has no posterior, no marginalization, and no scale-identifiability mechanism — the calibration-latent likelihood is a different inference object, not a bolt-on.

**Why this measures what we claim.** The calibration posterior variance sigma_f^2 measures calibration uncertainty because the reparameterized Gaussian is assumed to capture all ambiguity in focal length given the stereo pair and the scale-breaking prior; this assumption fails when calibration error is systematic (e.g., a mis-specified principal point), in which case sigma_f^2 reflects only aleatoric image noise and underestimates true error. The marginalized metric depth D_metric measures metric-scale identifiability because the weighted mixture over calibration samples is assumed to concentrate when the disparity likelihood is sharp under the true focal; this assumption fails under textureless or repetitive regions, in which case the mixture stays broad and D_metric reflects an average over implausible geometries rather than a resolved depth. The residual scale_s measures the residual focal/scale ambiguity because it is assumed to absorb only the component of scale not explained by the calibration posterior; this assumption fails when the posterior is confidently wrong, in which case scale_s silently corrects a calibration error and masks it from the reported uncertainty. The posterior concentration along the f*b direction (reported as the interquartile range of f_samples * b) measures identifiability because it is assumed to shrink as the disparity likelihood sharpens; this assumption fails when the likelihood is flat along the ridge, in which case the concentration reflects only the prior width.

## Contribution

(1) CaliLaDepth, a calibration-latent metric depth module for binocular VLMs that marginalizes a Gaussian calibration posterior through an uncertainty-aware disparity likelihood, replacing StereoVLM's calibration-conditioned head with joint probabilistic inference over calibration, disparity, and metric scale. (2) A design principle — calibration-uncertainty-aware disparity supervision with a scheduled KL — that keeps metric scale identifiable when calibration is absent or degraded, demonstrated by comparing calibrated, partially calibrated, and uncalibrated regimes. (3) A calibration-posterior calibration diagnostic that reports when reported uncertainty is trustworthy versus when the residual scale head is masking calibration error.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VSI-Bench + ScanQA/SQA 3D-QA split | Covers metric spatial reasoning and 3D grounding/QA |
| Primary metric | Metric depth error conditioned on calibration-error strata | Isolates identifiability, not aggregate accuracy |
| Secondary metric | Posterior coverage of true focal (f*b direction) | Tests whether uncertainty tracks calibration error |
| Baseline | StereoVLM (calibration-conditioned) | Tests point-estimate calibration vs latent posterior |
| Baseline | VG LLM (video-only 3D priors) | Tests whether explicit stereo calibration helps at all |
| Baseline | 2D-LVLM cdViews zero-shot (3D-QA) | Tests whether calibration latency beats view selection |
| Ablation-of-ours | CaliLaDepth w/o calibration posterior (point-estimate calib) | Isolates the marginalization contribution |
| Ablation-of-ours | CaliLaDepth w/o scale-breaking prior (scale-invariant prior) | Tests whether the prior breaks the f*b ridge |

### Why this setup validates the claim

The central claim is that treating calibration as a latent variable, marginalized through a heteroscedastic disparity likelihood, produces metric depth that is both more accurate and more honestly uncertain than point-estimate calibration conditioning. The setup is falsifiable because the dataset is stratified by calibration-error regime: near-nominal calibration, mild focal/baseline perturbation, and partial or missing calibration. StereoVLM tests the sub-claim that a point-estimate calibration head cannot express calibration ambiguity and therefore degrades specifically as calibration error grows; VG LLM tests whether explicit stereo geometry, even probabilistically treated, buys anything over video-derived priors; cdViews tests whether the calibration posterior is doing work that view selection cannot substitute for. The ablation—point-estimate calibration with the same disparity likelihood—is the discriminating control: if marginalization is the causal mechanism, the ablation should recover baseline-like behavior under calibration perturbation while matching ours under nominal calibration. The scale-invariant-prior ablation tests whether the scale-breaking prior is necessary to break the f*b ridge: if the prior is load-bearing, this ablation should show broad posterior concentration along f*b and degraded metric depth under perturbation. The primary metric is depth error conditioned on calibration-error strata rather than aggregate QA accuracy, because the predicted effect is a stratum-specific divergence; an aggregate metric would average away exactly the signal the claim predicts. The secondary metric—posterior coverage of the true focal along the f*b direction—directly tests whether sigma_f^2 measures calibration error: we compute the fraction of test frames where the true f*b lies within the 90% posterior interval, and expect coverage near 0.9 under nominal calibration and degradation under systematic error. A secondary QA/grounding readout confirms the geometric improvements propagate to language reasoning rather than remaining a depth-only artifact.

### Expected outcome and causal chain

**vs. StereoVLM** — On a case where the true focal length deviates from the nominal value used at inference, StereoVLM's point-estimate calibration head commits to a single focal/baseline pair and produces depth that is globally rescaled by the focal error, because it has no representation of calibration ambiguity and no mechanism to let the disparity likelihood down-weight a bad calibration. Our method instead samples K=8 calibrations from the learned posterior and reweights them by the heteroscedastic disparity likelihood, so implausible focal values are suppressed and the weighted metric depth concentrates near the correct scale. We expect a noticeable error gap on the perturbed-calibration stratum but near-parity on the nominal stratum.

**vs. VG LLM** — On a case requiring metric distance judgments in a textureless corridor, VG LLM's video-derived 3D geometry prior produces depth with plausible ordinal structure but poorly anchored absolute scale, because it learns geometry priors from video without explicit cross-view calibration constraints. Our method instead uses the calibrated stereo likelihood to anchor the focal-times-baseline product, so absolute scale is constrained by explicit geometry rather than inferred from appearance statistics. We expect our method to show a clear advantage on metric-scale questions while both methods behave similarly on relative/ordinal spatial questions.

**vs. cdViews (2D-LVLM zero-shot)** — On a case where the queried object is visible but the selected views under-sample the region, cdViews produces confident but inconsistent answers across view subsets, because view selection cannot recover metric geometry it never observed and has no uncertainty channel. Our method instead marginalizes over calibration and exposes posterior spread, so the same instance yields a metric depth estimate whose confidence tracks the actual geometric ambiguity. We expect our method to outperform specifically on metric-depth and calibration-sensitive QA items, with cdViews remaining competitive on purely semantic grounding.

**vs. CaliLaDepth w/o calibration posterior (ablation)** — On a case with systematic principal-point error, the ablation's point-estimate calibration head absorbs the error into the metric scale head, producing depth that looks confident but is silently biased, because a single calibration estimate cannot express the ambiguity that the scale head then hides. Our method instead spreads probability mass over calibrations consistent with the stereo pair, so the scale head corrects only the residual and the posterior variance flags the ambiguity. We expect the ablation to match ours under nominal calibration but to lose accuracy and calibration of its uncertainty under perturbation.

**vs. CaliLaDepth w/o scale-breaking prior (ablation)** — On a case with unknown focal length, the scale-invariant-prior ablation leaves the f*b product unidentifiable along the ridge, so the posterior remains broad or collapses to an arbitrary point and the scale head absorbs the remaining ambiguity, because a scale-invariant prior cannot select a unique scale. Our method instead uses a log-normal prior on focal that breaks the ridge, so the posterior concentrates along f*b and the scale head corrects only the residual. We expect the scale-invariant ablation to show broad posterior concentration along f*b and degraded metric depth under perturbation, while matching ours under nominal calibration.

### What would falsify this idea

If CaliLaDepth's gain over the point-estimate ablation is uniform across calibration-error strata rather than concentrated in the perturbed and partial-calibration strata, the marginalization is not the causal mechanism and the central claim fails. Likewise, if the calibration posterior variance fails to grow with injected calibration error while depth error grows, or if the scale-invariant-prior ablation matches ours under perturbation, the uncertainty is not measuring calibration ambiguity as claimed.

## References

1. StereoVLM: Binocular Vision-Language Models for Metric 3D Spatial Reasoning

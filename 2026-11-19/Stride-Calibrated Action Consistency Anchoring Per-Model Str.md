# Stride-Calibrated Action Consistency: Anchoring Per-Model Stride References to a Shared Physical Scale

## Motivation

StrideWorld measures action consistency by computing each model's stride from its own first steps and then judging later steps against that self-calibrated yardstick, so a model with distorted initial motion sets its own scale and cross-model commensurability is never established — a model that moves too little or too much on its first step is rewarded for staying consistent with its own error. This is a structural measurement problem, not a model-coverage problem: the benchmark already validates its stride measurements on rendered videos with known camera motion, so an absolute physical scale exists in the pipeline but is never used to tie different models' stride units together. The same relative-reference fragility recurs across the field's evaluation designs, where metrics are anchored to properties of the video under test rather than a shared external scale, making reported scores incomparable across model families.

## Key Insight

Because stride is a ratio of displacement to command magnitude that is linear in the unknown per-model scale factor, a single first-step measurement against a physically rendered video of known camera motion identifies that scale factor exactly, turning a model-relative consistency score into an absolute, cross-model-comparable quantity without altering the underlying consistency test.

## Method

**What it is.** CaliStride is a post-hoc calibration protocol for StrideWorld-style action-consistency scores. Its inputs are (i) a target video model's generated first-step clips for a set of calibration commands, (ii) physically rendered reference clips with known camera motion for the same commands, and (iii) the model's existing per-action stride measurements. Its output is a calibrated action-consistency score expressed in shared physical units (meters of camera translation per unit command stride), plus a scale factor per model.

**How it works.**

```pseudocode
# Phase 1: absolute reference scale
for each calibration command c in C_cal (|C_cal| = 32 shared commands, pre-registered, spanning translation magnitudes ||c|| in {0.1, 0.25, 0.5, 1.0, 2.0, 4.0} meters, 5-6 commands per magnitude):
    render reference clip R_c with known camera translation t_ref(c)  # ground-truth geometry
    s_ref(c) = t_ref(c) / ||c||          # absolute stride, meters per command unit

# Phase 2: per-model scale identification
for each model M:
    for each c in C_cal:
        generate first-step clip V_M(c)
        s_M(c) = measured_translation(V_M(c)) / ||c||   # StrideWorld first-step stride
    k_M = median_c( s_ref(c) / s_M(c) )   # scalar scale factor, robust to outlier commands
    # k_M maps model-relative stride units onto absolute physical units
    # proportionality diagnostic: compute ratio r_c = s_ref(c)/s_M(c) for all c in C_cal
    #   report coefficient of variation CV_M = std(r_c)/mean(r_c)
    #   report linearity fit R^2 of s_M(c) vs s_ref(c) across the 32 commands
    #   report residual non-proportionality: max_c |r_c - k_M| / k_M

# Phase 3: calibrated consistency score
for each evaluated action a of model M on start view v:
    stride_M(a) = StrideWorld first-step stride for a
    stride_abs(a) = k_M * stride_M(a)          # re-express in calibrated units
    err(a) = consistency_error(stride_abs(a), subsequent steps of a)  # same test as StrideWorld
score_cal(M) = mean_a err(a)
report score_cal(M) alongside k_M, raw score_raw(M), and the proportionality diagnostics (CV_M, R^2_M, max residual)
```

Hyperparameters: calibration set size |C_cal| = 32 commands drawn to span translation magnitudes (six magnitude bins, 5-6 commands each); scale estimator is the median (not mean) over calibration commands; no per-action tuning; k_M recomputed once per model. The proportionality diagnostics (CV_M, R^2_M, max residual) are computed once per model from the same 32 calibration commands and reported as first-class outputs.

**Why this design.** We chose a single scalar scale factor per model over a per-action or per-dimension calibration because stride is defined as a ratio linear in the unknown model scale, so one scalar is the minimal identifiable correction and a richer correction would fit measurement noise; the cost is that a model whose distortion is non-uniform across commands (e.g., correct on large translations, collapsed on small ones) is only corrected in the median sense, and its residual non-uniformity is reported rather than removed. We chose physically rendered reference clips with known camera motion over human-annotated stride targets because the benchmark already validates its measurements on rendered videos with known camera motion, so the absolute anchor is available at zero new annotation cost, accepting that the reference renderer's camera model must match the generator's command parameterization. We chose a median over calibration commands rather than a least-squares fit because calibration commands are few and outlier-prone, accepting slightly higher variance for robustness to a single badly generated calibration clip. We chose to report k_M alongside the calibrated score rather than folding it into a single number because k_M itself is diagnostic — a model with k_M far from 1 is one whose first-step motion is mis-scaled, which raw StrideWorld scores conceal. This is not a variant of standard metric normalization or score standardization (e.g., z-scoring benchmark outputs): those rescale by statistics of the evaluated population, whereas CaliStride rescales by an external physical quantity measured before the motion it judges, so the correction does not depend on which other models are in the comparison set.

**Why it measures what we claim.** The calibrated stride k_M · stride_M(a) measures absolute camera displacement per command unit because the assumption that the model's generated first-step translation is proportional to true translation with a single unknown scale factor makes the ratio s_ref(c)/s_M(c) constant across calibration commands, so the median identifies that constant; this assumption fails when the model's motion is nonlinearly distorted (e.g., a fixed minimum displacement that dominates small commands), in which case k_M reflects the median-command scale and the calibrated score inherits a command-dependent bias. The consistency error err(a) measures action consistency because the assumption that the first step establishes the intended pace for the command makes deviation of subsequent steps from that pace a consistency violation; this assumption fails when the first step itself is an outlier within the action, in which case err(a) reflects first-step noise rather than sustained inconsistency. The cross-model comparability of score_cal(M) measures commensurability because the assumption that the reference renderer and the generator share a command-to-translation parameterization makes the calibrated units physically identical across models; this assumption fails when a model interprets command units on a different internal scale, in which case k_M absorbs the interpretation difference and comparability holds only up to that interpretation. The proportionality diagnostics (CV_M, R^2_M, max residual) measure the constancy of the ratio s_ref(c)/s_M(c) because the assumption that the ratio is constant across calibration commands makes its coefficient of variation and linearity fit direct tests of that constancy; this assumption fails when the model's motion is nonlinearly distorted, in which case the diagnostics reveal the command-dependent bias and the calibrated score is reported as median-command rescaling rather than an absolute physical quantity.

## Contribution

(1) CaliStride, a cross-model calibration protocol that maps StrideWorld's per-model self-calibrated stride references onto a shared absolute physical scale using rendered clips with known camera motion, yielding a calibrated action-consistency score and a per-model scale diagnostic k_M. (2) The design principle that relative-reference benchmarks become cross-model comparable by identifying a single scalar scale factor from a small shared calibration command set, with the scale reported as a first-class diagnostic rather than discarded. (3) An analysis of how per-model scale factors distribute across open video models and how calibrated versus raw consistency rankings diverge.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | StrideWorld benchmark (140 start views, open models) | Provides physical references and per-action stride measurements |
| Primary metric | Calibrated action-consistency error score_cal(M) | Tests comparability in shared physical units |
| Baseline | Raw StrideWorld score_raw(M) | Uncalibrated model-relative stride units |
| Baseline | z-scored benchmark outputs | Population-statistics normalization, not physical anchoring |
| Baseline | Per-action least-squares scale fit | Fits measurement noise without external anchor |
| Ablation-of-ours | CaliStride with mean instead of median estimator | Tests robustness of scalar scale identification |
| Diagnostic | Proportionality diagnostics: CV_M, R^2_M, max residual of s_ref(c)/s_M(c) across the 32 calibration commands | Directly tests the constancy assumption underlying the median estimator |

### Why this setup validates the claim

The central claim is that CaliStride's external physical anchor produces cross-model comparable action-consistency scores where raw StrideWorld scores and population-based normalization cannot. Raw StrideWorld is the natural baseline because it shares the identical consistency test but lacks calibration; if calibration matters, the gap between score_cal and score_raw must track each model's k_M deviation from 1, not appear uniformly. z-scoring tests the sub-claim that physical anchoring differs from statistical normalization: z-scores are invariant to which models are in the comparison set, whereas CaliStride should produce stable rankings even when the model pool changes. The least-squares baseline tests whether the median's robustness claim is real: on calibration sets containing one badly generated clip, least-squares should absorb the outlier and shift k_M, while median stays stable. The mean-vs-median ablation isolates the estimator's contribution from the anchoring's contribution. The proportionality diagnostics (CV_M, R^2_M, max residual) test the load-bearing assumption that the ratio s_ref(c)/s_M(c) is constant across calibration commands; if the diagnostics show high CV or low R^2 for a model, the calibrated score for that model is reported as median-command rescaling rather than an absolute physical quantity, and cross-model comparability is claimed only on the command range where the residual is below a stated tolerance (e.g., max residual < 0.2). The primary metric score_cal(M) is the right detector because it is expressed in shared physical units, so any residual cross-model incomparability surfaces directly as error magnitude, not as an unexplained ranking flip.

### Expected outcome and causal chain

**vs. Raw StrideWorld score_raw(M)** — On a model whose first-step motion is systematically compressed (k_M far from 1, e.g. generating roughly half the commanded translation), raw StrideWorld measures subsequent steps against a compressed first-step pace, so its consistency error stays low even though the model's absolute motion is wrong by a constant factor. Our method multiplies stride by k_M before the consistency test, so the same action is judged against a physically anchored pace. We expect score_cal to rise sharply for that model while staying near score_raw for models with k_M close to 1, producing a gap that correlates with |log k_M|. The proportionality diagnostics for that model should show low CV and high R^2 if the compression is uniform; if the diagnostics show high CV, the gap is attributed to median-command rescaling rather than absolute calibration.

**vs. z-scored benchmark outputs** — On a comparison set where a single model dominates the population, z-scoring recenters every model's score around that model's statistics, so a physically mis-scaled model can appear average simply because it sits mid-pack. Our method anchors to rendered reference geometry independent of the model pool, so the same model's calibrated score is unchanged whether or not the dominant model is included. We expect z-scored rankings to shift when the model pool changes while CaliStride rankings remain stable, with the divergence concentrated on models whose k_M deviates most and whose proportionality diagnostics indicate a well-defined scale (low CV, high R^2).

**vs. Per-action least-squares scale fit** — On a calibration set containing one badly generated clip (e.g. a command where the model produces a near-static first step), least-squares fits the scale to that outlier and inflates k_M for the whole model, corrupting every subsequent action's calibrated error. Our median estimator ignores the single outlier because it requires only that most calibration commands be well-behaved. We expect the least-squares variant to show elevated score_cal on clean actions of otherwise well-behaved models, while our median version shows parity there and divergence only on the genuinely mis-scaled model. The proportionality diagnostics will flag the outlier command as a high-residual point, confirming that the median's robustness is due to ignoring that point rather than to a different scale estimate.

### What would falsify this idea

If the gap between score_cal and score_raw is uniform across all models regardless of their k_M deviation from 1, then the calibration is not correcting a real physical mis-scaling and the central claim fails. Likewise, if CaliStride rankings shift as much as z-scored rankings when the model pool changes, the physical-anchor advantage over population normalization is not real. Additionally, if the proportionality diagnostics show high CV and low R^2 for all models, the constant-scale premise is false and the calibrated score cannot be claimed as an absolute physical quantity.

## References

1. StrideWorld: Benchmarking Scene Rigidity and Action Consistency in Interactive Video World Model

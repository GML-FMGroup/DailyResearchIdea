# Noise-Robust Family-Balanced Replay for Spatial VLM Post-Training

## Motivation

ChronoReplay replays source-native spatial targets uniformly, treating every automatically generated 3D annotation as ground truth. This assumption is structurally fragile: SpatialVLM's 2-billion-example pipeline derives targets from automatic 3D reconstruction, whose depth and pose errors produce relation-label noise that is correlated with the answer family (e.g., metric distance vs. relative direction). Uniform family-balanced replay therefore amplifies, rather than attenuates, the noisiest families, and no prior spatial-VLM branch (SpatialVLM, SpatialBot, EmbSpatial-Bench) provides a noise-aware estimator for this supervision.

## Key Insight

Because annotation noise in automatic 3D pipelines is heteroscedastic across spatial answer families and observable through cross-family disagreement on the same scene, a family-conditional noise model estimated from agreement statistics yields a reweighting that is identifiable without clean labels, turning family-balanced replay into a robust estimator rather than a uniform one.

## Method

**What it is.** NR-ChronoReplay (Noise-Robust ChronoReplay) is a post-training procedure that augments ChronoReplay's family-balanced joint SFT with a family-conditional noise-transition estimator and a robust cross-entropy loss. Inputs: chronologically ordered episodes with source-native targets, grouped by answer family; outputs: a single checkpoint trained with per-example, per-family soft targets.

**Load-bearing assumption (stated explicitly).** The estimator assumes that cross-family disagreement on the same spatial fact is a valid, unbiased proxy for the true annotation-noise rate eta_f of each family: agreement is high when the source target is correct and low when it is corrupted, and the disagreement is attributable to label noise rather than to model error or genuine cross-family ambiguity. This assumption is closed by the premise that families have independent error channels. The known failure mode is shared depth/pose bias across families (e.g., metric distance and relative direction both inherit the same depth bias), in which case agreement stays high precisely when both are wrong and eta_f underestimates noise. The method therefore includes a shared-bias diagnostic (Phase 1b) and a synthetic-corruption calibration check (Phase 1c) that must pass before the transition matrix is trusted.

**How it works.**

```pseudocode
# Phase 1: family-conditional noise estimation from cross-family agreement
# Precondition audit (data-audit step):
#   For each scene s, match queries across families to the same spatial fact
#   using the source reconstruction's object-pair IDs and relation triple
#   (subject_id, predicate_type, object_id). Coverage rate = fraction of scenes
#   with >=2 families querying the same triple. Report coverage; scenes with
#   no multi-family overlap fall back to eta_f = 0 for the unmatched family.
for each scene s in a held-out 5% split of episodes:
    for each answer family f:
        collect all (question q, source-native target y) pairs in family f for scene s
        # Cross-family consistency signal: the same spatial fact is queried
        # by different families (e.g., metric distance, relative direction, ordinal)
        compute per-pair agreement a_{q} = 1 if the model's family-f prediction
            is consistent with the majority vote of other families' predictions
            on the same spatial fact, else 0
    estimate family noise rate eta_f = 1 - mean_q a_{q}   # scalar per family
    # Transition matrix T_f: P(observed y | true y*) with smoothing
    T_f = (1 - eta_f) * I + eta_f * Uniform(K_f)   # K_f = # labels in family f

# Phase 1b: shared-bias diagnostic (residual disagreement after removing a common depth-bias term)
#   Fit a single scalar depth-bias term b_d on the held-out split by regressing
#   metric-distance residuals on reconstruction depth; subtract b_d from all
#   families' targets; recompute eta_f^resid. Report delta_f = eta_f - eta_f^resid.
#   If delta_f is large for a family, that family's eta_f is dominated by shared
#   bias and the transition matrix under-smooths; flag it in the results table.

# Phase 1c: synthetic-corruption calibration check (must pass before Phase 2)
#   Take 2,000 clean human-verified examples spanning all families. Inject a
#   known depth-quantization channel shared across families at rate r in {0.05, 0.10, 0.20}.
#   Run Phase 1 on the corrupted set; report Pearson correlation between
#   estimated eta_f and true injected rate r, per family. If correlation < 0.7
#   for any family, that family's transition matrix is disabled (eta_f = 0).

# Phase 2: robust family-balanced joint SFT
for each training batch:
    sample episodes with family-balanced sampler (as in ChronoReplay)
    for each example (x, y, f):
        soft_target = T_f[y]                      # row of transition matrix
        loss += - sum_k soft_target[k] * log p_theta(k | x)
    update theta with AdamW, lr 1e-5, 2 epochs

# Phase 3: optional clean-validation reweighting (small, 500 examples)
# human-verified subset used only to fit a single scalar temperature tau
# that rescales eta_f; no per-example labels used.
```

**Why this design.** We chose a family-conditional transition matrix over per-example learned noise models because annotation errors in automatic 3D pipelines are generated by shared reconstruction artifacts (depth quantization, pose drift) that act at the family level, so per-example models would overfit to spurious features; the cost is that within-family heterogeneity is not captured. We chose cross-family agreement as the noise signal over model log-probability because log-prob calibration for correctness is unreliable, whereas agreement across independently-parameterized answer interfaces on the same spatial fact is a direct consequence of the reconstruction error that produced the label; the cost is that a family with no overlapping queries cannot be estimated and falls back to eta_f = 0. We chose label smoothing via a uniform transition matrix over a learned full matrix because the uniform component is the maximum-entropy assumption under unknown error direction, avoiding the need for clean per-family labels; the cost is that systematic directional bias (e.g., depth always underestimated) is only partially corrected. We chose to fit only a single scalar temperature on a small clean validation set rather than full reweighting, because full reweighting reintroduces the annotation-noise assumption through the validation labels themselves. The Phase 1b shared-bias diagnostic and Phase 1c synthetic-corruption check are included because the load-bearing assumption (agreement isolates label noise) is falsifiable when families share a depth/pose bias; these checks make the failure mode observable rather than silent.

**Why it measures what we claim.** The family noise rate eta_f measures the annotation-error magnitude for family f because cross-family agreement on the same spatial fact is assumed to be high when the source target is correct and low when it is corrupted by reconstruction error; this assumption fails when two families share the same systematic pipeline error (e.g., both metric distance and relative direction inherit the same depth bias), in which case eta_f underestimates true noise and the transition matrix under-smooths. The transition matrix T_f measures the label-noise channel because the uniform-mixture form assumes errors are directionally unbiased within a family; this assumption fails when errors are biased (e.g., all distances rounded down), in which case T_f reflects a smoothed but still biased target. The clean-validation temperature tau measures the residual calibration gap between estimated and true noise because a small human-verified set provides an unbiased estimate of the loss surface; this assumption fails when the clean set is not representative of the training distribution, in which case tau corrects only global scale, not family-specific miscalibration. The Phase 1b residual eta_f^resid measures the non-shared component of annotation noise because subtracting a common depth-bias term removes the correlated error channel; this assumption fails when the depth bias is not a single scalar (e.g., pose drift varies per scene), in which case delta_f is an incomplete correction. The Phase 1c synthetic-corruption correlation measures whether eta_f tracks true injected noise because the injected channel is known and shared; this assumption fails when the model's own predictions on the corrupted set are themselves biased by the same channel, in which case the correlation is inflated and the check is not diagnostic.

## Contribution

['NR-ChronoReplay: a noise-robust family-balanced replay estimator that couples a family-conditional label-noise transition matrix, estimated from cross-family agreement on shared spatial facts, with a robust cross-entropy loss, requiring no clean per-example labels.', 'A design principle for spatial VLM post-training: annotation noise from automatic 3D pipelines is heteroscedastic across answer families and identifiable from cross-family disagreement, so family-balanced sampling must be paired with family-conditional denoising rather than uniform replay.', 'A lightweight calibration protocol using a 500-example human-verified set to fit a single scalar temperature that rescales estimated family noise rates, avoiding full clean-label supervision.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ChronoReplay streaming episodes (OVO-S-Bench, OST-Bench, UCS-Bench) | Same data and benchmarks isolate the noise-handling contribution |
| Primary metric | Three-benchmark mean accuracy | Single aggregate sensitive to family-level label noise |
| Baseline | Ordinary joint SFT (ChronoReplay) | Tests whether robust loss beats standard family-balanced training |
| Baseline | Natural turn weighting | Tests whether family balancing, not noise modeling, drives gains |
| Baseline | Shifted-order control | Tests whether chronological ordering interacts with noise correction |
| Baseline | Generic consensus/co-training noise estimator (MentorNet-style forward correction with learned confusion matrix) | Tests whether family-conditioning, not generic noise modeling, drives gains |
| Ablation-of-ours | eta_f = 0 (transition matrix disabled) | Isolates the contribution of the noise-transition estimator |
| Ablation-of-ours | Phase 1b shared-bias correction removed (delta_f not subtracted) | Tests whether shared-bias diagnostic changes the estimate |
| Diagnostic | Phase 1c synthetic-corruption recovery: Pearson correlation between estimated eta_f and injected depth-quantization rate r in {0.05, 0.10, 0.20} | Tests whether eta_f tracks true noise, not model consensus |
| Diagnostic | Data-audit coverage rate: fraction of scenes with ≥2 families querying the same (subject_id, predicate_type, object_id) triple | Tests the estimator's precondition |
| Diagnostic | Cross-dataset transfer: fit eta_f on OVO-S-Bench, apply to OST-Bench, report accuracy change | Tests whether noise model is reusable diagnostic |

### Why this setup validates the claim

The central claim is that source-native spatial targets carry family-level annotation noise from shared 3D reconstruction artifacts, and that a family-conditional transition matrix recovers accuracy lost to that noise. The design makes this falsifiable by holding data, ordering, and serialization fixed while varying only how noise is handled. Ordinary joint SFT tests the raw claim that robustness helps at all; natural turn weighting tests whether the gain is merely a reweighting artifact rather than genuine noise correction; the shifted-order control tests whether the effect depends on chronological structure, which the noise estimator assumes when pooling cross-family agreement per scene. The generic consensus/co-training baseline tests whether the family-conditioning is what produces the gain, rather than any generic noise estimator. The eta_f = 0 ablation is the sharpest test: if removing the transition matrix leaves accuracy unchanged, the noise model is doing nothing and the claim collapses. The Phase 1c synthetic-corruption diagnostic directly tests the load-bearing assumption: if estimated eta_f does not track the known injected depth-quantization rate, the agreement statistic is measuring model consensus, not label noise, and the method's premise is falsified. The data-audit coverage rate quantifies how many scenes actually satisfy the estimator's precondition (multi-family queries on the same spatial fact), so the claimed applicability is bounded by measured coverage. Three-benchmark mean accuracy is the right metric because the predicted effect is a family-level shift in correct-label probability, which aggregates into overall accuracy rather than into a single-family diagnostic; a metric concentrated on one family would not distinguish noise correction from family-specific tuning.

### Expected outcome and causal chain

**vs. Ordinary joint SFT (ChronoReplay)** — On a case where a metric-distance target is corrupted by depth quantization so the source-native label is off by a bin, the baseline fits the wrong label at full confidence because its cross-entropy treats every target as clean. Our method instead softens that target toward the uniform transition row for the metric family, because eta_f estimated from cross-family disagreement flags the family as noisy, so we expect a noticeable gap on the noisy-family subset with parity on the cleanly-annotated family subset.

**vs. Natural turn weighting** — On a case where a relative-direction answer is systematically biased by pose drift, the baseline reweights turns but still pushes the biased label hard because weighting changes example frequency, not label trust. Our method instead redistributes probability mass within the direction family according to T_f, because the family noise rate captures the shared reconstruction error, so we expect gains concentrated on direction-family items while turn-weighted and our method tie on items from families with near-zero estimated noise.

**vs. Shifted-order control** — On a case where the same spatial fact is queried by distance and ordinal families within one scene, the shifted-order baseline breaks the chronological adjacency that lets cross-family agreement be computed per scene, so its targets stay sharp and biased. Our method instead pools agreement across families within each scene regardless of turn order, because the estimator is scene-conditioned, so we expect the gain over shifted-order to be largest precisely on scenes with multi-family overlap and to shrink toward parity on single-family scenes.

**vs. Generic consensus/co-training noise estimator (MentorNet-style forward correction with learned confusion matrix)** — On a case where a metric-distance target is corrupted by depth quantization, the generic estimator learns a confusion matrix from model predictions without family conditioning, so it conflates families with different noise rates and applies a single average correction. Our method instead estimates a separate eta_f per family, so we expect the generic estimator to under-correct the noisiest family and over-correct the cleanest, yielding a smaller overall gain and a flatter per-family improvement profile than our family-conditional transition matrix.

**vs. eta_f = 0 ablation** — On a case where depth bias shifts an entire family's targets, the ablation behaves like ordinary SFT and overfits the biased labels because no smoothing is applied. Our method instead applies the uniform-mixture transition for that family, because eta_f is nonzero and the maximum-entropy component absorbs unknown error direction, so we expect the ablation to recover most of the baseline's loss and our full method to show a clear gap over it on the noisy families.

**vs. Phase 1b shared-bias correction removed** — On a case where metric distance and relative direction share the same depth bias, the uncorrected estimator sees high agreement and sets eta_f near zero, so the transition matrix under-smooths. Our method instead subtracts the fitted scalar depth-bias term b_d and recomputes eta_f^resid, so we expect the corrected version to show a larger gain on the shared-bias families and a measurable delta_f in the diagnostic table.

**Phase 1c synthetic-corruption recovery** — On a controlled set of 2,000 clean human-verified examples with injected depth-quantization rate r in {0.05, 0.10, 0.20}, we expect estimated eta_f to correlate with r at Pearson ≥ 0.7 per family; if correlation falls below 0.7 for a family, that family's transition matrix is disabled and the method reverts to eta_f = 0 for that family, which would falsify the estimator's validity for that family.

**Data-audit coverage** — We expect the coverage rate (fraction of scenes with ≥2 families on the same spatial fact) to be reported per benchmark; if coverage is below 30% on a benchmark, the estimator's precondition is not met there and gains on that benchmark are not attributable to the noise model.

**Cross-dataset transfer** — We expect eta_f fitted on OVO-S-Bench to produce a nonzero accuracy gain when applied unchanged to OST-Bench, with the gain concentrated on families whose estimated eta_f is high in both datasets; if transfer gain is zero, the noise model is a one-off tuning artifact rather than a reusable diagnostic.

### What would falsify this idea

If the gain over ordinary joint SFT is uniform across all answer families rather than concentrated on families with high estimated eta_f, the family-conditional noise model is not the cause and the central claim is wrong. Likewise, if the eta_f = 0 ablation matches the full method on the noisy-family subset, the transition matrix is inert and the mechanism is falsified. Additionally, if the Phase 1c synthetic-corruption check yields Pearson correlation below 0.7 between estimated eta_f and injected noise rate for the noisy families, the agreement statistic is measuring model consensus rather than label noise, and the load-bearing assumption fails.

## References

1. ChronoReplay: One Answer Interface Is Not Enough for Streaming Spatial Intelligence
2. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
3. VideoLLM-online: Online Video Large Language Model for Streaming Video
4. SpatialBot: Precise Spatial Understanding with Vision Language Models
5. RoboVQA: Multimodal Long-Horizon Reasoning for Robotics
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Interactive Language: Talking to Robots in Real Time
8. VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners

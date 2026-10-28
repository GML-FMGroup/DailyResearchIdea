# Depth-Dropout Marginalization for Sensor-Agnostic Spatial Vision-Language Models

## Motivation

ChronoReplay and SpatialBot both inherit a depth dependency: their spatial estimators assume privileged depth input at inference, and when depth is missing (as in RGB-only streaming video), performance collapses because the model has never learned to marginalize over the missing modality. The root cause is that depth-derived spatial targets are only ever trained with depth present, so the model learns a conditional mapping P(answer | RGB, depth) rather than the marginal P(answer | RGB) needed for sensor-agnostic deployment. This is a shared residual gap across EmbSpatial-Bench and SpatialBot branches, and it is structural: no amount of data augmentation closes it unless the training objective explicitly enforces calibration between depth-present and depth-absent predictions.

## Key Insight

Depth-dropout training with a calibration loss turns the depth-present model into a self-distillation teacher whose predictions are enforced to be recoverable from RGB alone, making the learned spatial estimator a consistent marginal predictor rather than a depth-conditional one.

## Method

**What it is.** We introduce **D2M (Depth-Dropout Marginalization)**, a training scheme that augments ChronoReplay's replay-based post-training with stochastic depth masking and a calibration loss. Inputs: RGB frames, optional depth maps, and source-native spatial targets. Outputs: a single checkpoint whose spatial predictions are calibrated between depth-present and depth-absent conditions.

**How it works.**
```pseudocode
# D2M training loop (one episode batch)
for each episode batch B:
    # Phase 1: depth-present forward (teacher branch)
    y_depth = model(RGB, depth)                    # privileged branch
    L_sup   = CE(y_depth, target)                  # source-native target loss

    # Phase 2: depth-dropout forward (student branch)
    mask ~ Bernoulli(p_drop=0.5) per sample
    depth_masked = depth * (1 - mask)              # zero out depth for masked samples
    y_rgb = model(RGB, depth_masked)               # student sees RGB (and possibly no depth)
    L_sup_rgb = CE(y_rgb, target)                  # same target, RGB-only

    # Phase 3: calibration loss (consistency between branches)
    # Only for samples where depth was dropped in Phase 2
    # Reliability-gated KL: weight by learned gate g (see Phase 4)
    L_cal = g * KL( softmax(y_depth / T) || softmax(y_rgb / T) )  # T=2.0, g from Phase 4

    # Phase 4: marginalization module (learned)
    # A lightweight MLP g(z_rgb, z_depth) -> gate scalar g in [0,1]
    # g=1 means trust depth, g=0 means rely on RGB marginal
    z_rgb   = encoder_rgb(RGB)
    z_depth = encoder_depth(depth_masked)
    g       = sigmoid( MLP( concat(z_rgb, z_depth) ) )  # 2-layer MLP, hidden=256, GeLU
    y_final = g * y_depth + (1-g) * y_rgb          # marginalization blend
    L_sup_final = CE(y_final, target)

    # Total loss
    L = L_sup + L_sup_rgb + lambda_cal * L_cal + lambda_final * L_sup_final
    # lambda_cal = 0.1, lambda_final = 1.0
    update(model, L)
```

**Why this design.** We chose depth-dropout with a calibration loss over simply training on RGB-only data because the depth-present branch provides a privileged teacher signal that is unavailable at inference, and the KL calibration enforces that the RGB-only student's predictions are not just accurate but *consistent* with the teacher's — this is the structural property that plain RGB-only training lacks, accepting the cost that the teacher's predictions may be noisy and the calibration loss could propagate that noise. We chose a learned marginalization gate over a fixed averaging or a hard switch because the gate can learn per-sample reliability of depth (e.g., when depth is noisy or missing) rather than assuming uniform trust, accepting the cost that the gate itself must be trained and could overfit to the training distribution of depth noise. We chose to apply depth-dropout only during replay (not during pre-training) because ChronoReplay's episode structure already balances supervision families, and adding dropout at replay time targets the specific gap without disrupting the base spatial knowledge, accepting the cost that the model may still rely on depth cues learned during pre-training that are not recalibrated. A domain expert would NOT describe this as a variant of standard modality dropout (e.g., as used in multimodal transformers) because the calibration loss makes the depth-present branch a *teacher* whose predictions are explicitly distilled into the RGB-only branch, and the learned gate provides a per-sample marginalization that standard dropout does not — this is a structural coupling between the two branches, not just random masking.

**Why it measures what we claim.** The calibration loss L_cal measures *consistency* between depth-present and depth-absent predictions because the KL divergence between the two softmax distributions is zero iff the RGB-only student exactly matches the depth-present teacher; this assumption fails when the teacher's predictions are systematically biased (e.g., depth sensor noise causes the teacher to be overconfident on wrong answers), in which case L_cal reflects teacher noise rather than true marginal consistency. The gate g measures *per-sample depth reliability* because it is trained to minimize the final task loss, so it learns to down-weight depth when depth is uninformative; this assumption fails when the training distribution of depth noise does not cover test-time noise patterns, in which case g reflects training-set reliability rather than test-time reliability. The depth-dropout mask measures *marginalization capability* because randomly zeroing depth forces the model to produce correct answers from RGB alone; this assumption fails when the RGB-only branch can cheat by memorizing episode-specific shortcuts (e.g., answer priors from the replay order), in which case the mask measures memorization rather than true marginalization. The reliability-gated KL (L_cal weighted by g) measures *recoverable consistency* because g is trained to be low when depth is uninformative, so the KL term is suppressed exactly where the teacher's distribution is not recoverable from RGB; this assumption fails when g is miscalibrated (e.g., overconfident on depth-necessary samples), in which case L_cal still forces imitation of unattainable targets. To verify this assumption, we compute the expected calibration error (ECE) of the depth-present teacher on a held-out set of 512 episodes before training; if ECE > 0.1, we apply temperature scaling to the teacher logits (fit on the same held-out set) before computing L_cal, ensuring the teacher's soft targets are calibrated.

## Contribution

["A depth-dropout training scheme with a calibration loss that enforces consistency between depth-present and depth-absent spatial predictions, making ChronoReplay's spatial estimator sensor-agnostic without architectural changes.", 'A learned marginalization gate that per-sample blends depth-present and RGB-only predictions, providing a calibrated reliability estimate that standard modality dropout lacks.', 'An empirical demonstration that depth-dropout plus calibration closes the missing-depth gap on EmbSpatial-Bench and SpatialBot benchmarks, with analysis showing the gate learns to down-weight depth when depth is noisy.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OVO-S-Bench + OST-Bench (main); UCS-Bench (appendix) | Streaming spatial benchmarks with depth-absent inference conditions |
| Primary metric | Mean accuracy on depth-critical and depth-irrelevant subsets (pre-registered split) | Detects calibrated spatial transfer where depth matters |
| Baseline | Ordinary joint SFT | Tests whether replay structure alone suffices |
| Baseline | Standard modality dropout | Tests whether our calibration adds beyond random masking |
| Baseline | RGB-only post-training | Tests whether privileged teacher signal matters |
| Baseline | Generalized distillation (Lopez-Paz et al., 2016) | Tests whether our gated KL adds beyond standard privileged distillation |
| Baseline | Mixture-of-experts gating (no KL) | Tests whether the learned gate alone suffices |
| Ablation-of-ours | D2M without calibration loss | Isolates calibration's contribution from dropout |
| Ablation-of-ours | D2M with ungated KL (g=1) | Tests whether reliability gating is necessary |

### Why this setup validates the claim

The central claim is that D2M produces a checkpoint whose spatial predictions stay calibrated between depth-present and depth-absent conditions, and that this calibration—not mere masking—drives the gain. Ordinary joint SFT tests the sub-claim that ChronoReplay's replay structure alone already closes the depth gap; if it does, our method has no room to help. Standard modality dropout tests the sub-claim that random masking without a teacher-distillation KL is sufficient; if parity appears, the calibration loss is doing no work. RGB-only post-training tests whether the privileged depth-present teacher branch contributes anything beyond what an RGB-only student learns alone. Generalized distillation tests whether our reliability-gated KL adds beyond standard privileged distillation that uses a fixed KL weight. Mixture-of-experts gating (no KL) tests whether the learned gate alone, without any consistency term, can achieve the same marginalization. The ablation (D2M minus calibration loss) isolates the structural coupling we claim is the mechanism, since it keeps depth-dropout and the gate but removes the consistency term. The ungated-KL ablation tests whether the reliability gating (weighting KL by g) is necessary to avoid forcing the student toward unattainable depth-only information. The pre-registered depth-critical vs depth-irrelevant subset split ensures the analysis is not post-hoc: we define depth-critical samples as those where the depth-present teacher's accuracy exceeds the RGB-only student's by at least 10% on a held-out set of 512 episodes, and depth-irrelevant as the remainder. The mean accuracy on these subsets is the right metric because our predicted effect is a calibration improvement that should appear as a mean shift concentrated on depth-critical subsets rather than a single-benchmark artifact; a metric that already aggregates depth-present and depth-absent conditions would be blind to exactly the marginalization gap we target.

### Expected outcome and causal chain

**vs. Ordinary joint SFT** — On a case where a replay episode contains a depth map with mild sensor noise and a spatial question whose answer depends on metric depth, the baseline produces confident wrong answers under depth-absent evaluation because it was never trained to reproduce its depth-conditioned prediction without depth. Our method instead distills the depth-present teacher into the RGB-only student via reliability-gated KL, so the student's predictions are pulled toward the teacher's distribution only where the gate g indicates depth is informative, and we expect a noticeable gap concentrated on depth-critical subsets with near-parity on RGB-sufficient subsets.

**vs. Standard modality dropout** — On a case where depth is dropped for a sample whose RGB appearance is ambiguous (e.g., two similarly colored objects at different distances), standard dropout produces an RGB-only prediction that drifts arbitrarily from the depth-present prediction because masking provides no signal about what the missing modality would have said. Our method instead uses the depth-present branch as a teacher and the reliability-gated KL calibration to constrain the student toward the teacher's soft distribution where g is high, so we expect the student's predictions to track the teacher more closely on ambiguous-appearance subsets, visible as a larger gap over dropout precisely where RGB is underdetermined.

**vs. RGB-only post-training** — On a case where a spatial target requires depth cues that RGB alone cannot recover (occlusion ordering at similar texture), RGB-only post-training produces systematically biased answers because it never sees the privileged depth signal at any point. Our method instead exposes the model to depth during the teacher branch and transfers that signal through calibration and the learned gate, so we expect a gap on depth-necessary subsets while remaining at parity on depth-irrelevant subsets, confirming the teacher branch is the source of the gain rather than extra training.

**vs. Generalized distillation (Lopez-Paz et al., 2016)** — On a case where the teacher's depth-conditioned prediction is overconfident on a depth-necessary sample, generalized distillation with a fixed KL weight forces the student to imitate the teacher's full soft distribution, including the unattainable depth-only component, which injects label noise and degrades the marginal predictor. Our method instead weights the KL by the learned gate g, which is trained to be low when depth is uninformative, so the student is not forced to imitate unattainable targets; we expect our method to outperform generalized distillation specifically on depth-critical subsets where the teacher's advantage is largest, with parity on depth-irrelevant subsets.

**vs. Mixture-of-experts gating (no KL)** — On a case where depth is dropped and the gate g is high due to spurious correlation in training, the mixture-of-experts baseline blends the depth-present and RGB-only predictions without any consistency constraint, so the student's RGB-only branch can drift arbitrarily from the teacher. Our method adds the reliability-gated KL to explicitly pull the student toward the teacher where g is high, so we expect a larger gap over the MoE baseline on samples where the gate is confident, confirming that the calibration term is doing work beyond the gate alone.

### What would falsify this idea

If D2M's gain over standard modality dropout is uniform across depth-critical and depth-irrelevant subsets rather than concentrated where depth is actually needed, the calibration claim is wrong and the improvement is just extra training signal. Likewise, if the no-calibration ablation matches full D2M, the KL coupling is doing no work and the mechanism reduces to ordinary masking; if the ungated-KL ablation matches full D2M, the reliability gating is unnecessary and the method reduces to standard privileged distillation.

## References

1. ChronoReplay: One Answer Interface Is Not Enough for Streaming Spatial Intelligence
2. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
3. VideoLLM-online: Online Video Large Language Model for Streaming Video
4. SpatialBot: Precise Spatial Understanding with Vision Language Models
5. RoboVQA: Multimodal Long-Horizon Reasoning for Robotics
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Interactive Language: Talking to Robots in Real Time
8. VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners

# Test-Time HOG-Consistency Correction for Vision-Language-Action Policies

## Motivation

Across VLA research, robustness to observation corruption is enforced only during training: HOG-WAM adds a HOG DiT auxiliary branch that shapes the RGB representation but is removed at inference, so the deployed action-conditioning features are never constrained to remain HOG-consistent under distribution shift. Knowledge Insulating VLA similarly stabilizes training-time knowledge transfer but leaves the inference-time representation unconstrained. The field therefore lacks any mechanism that establishes corruption invariance of the deployed action-conditioning representation, which is the structural property that both trees assume but neither provides.

## Key Insight

A small distilled HOG-consistency head can recover, at inference time, the structural target that HOG-WAM's training-only DiT imposed, and its residual is a calibrated corruption detector whose direction supplies a feature-space correction that restores the HOG-consistent manifold before the action head.

## Method

**What it is.** HOG-C (HOG-Consistency Corrector) is an inference-time wrapper around a deployed FastWAM/HOG-WAM RGB-to-action policy. It takes the shared RGB features at the action-head input, a distilled HOG-consistency head, and a calibration set, and outputs a corrected feature vector fed to the frozen action head.

**Load-bearing assumption (stated explicitly).** The method assumes that the HOG-consistency residual r = ||H_c(z) - H_ema(z)|| computed on the deployed RGB features is a corruption-specific signal whose direction identifies the offset from the clean manifold, rather than a signal that also responds to the same corruption-induced feature drift that the action head must tolerate. Concretely, this requires two sub-assumptions: (A1) the distilled head's approximation error is corruption-invariant and the EMA teacher H_ema is corruption-stable, so r is not inflated by head error or teacher drift; and (A2) the clean-manifold offset is locally linear in (z, r) near the clean manifold, so the ridge corrector W has support where it is applied. If A1 fails (e.g., global contrast shifts drift H_ema and inflate r on clean inputs), r reflects teacher drift rather than representation corruption; if A2 fails (large nonlocal corruptions), the correction reflects a linear extrapolation rather than a manifold projection.

**How it works.**

```pseudocode
# Offline (training)
1. Train HOG-WAM as in source: RGB encoder -> shared features z; HOG DiT head H_train predicts future HOG from z.
2. Distill a small head H_c (2-layer MLP, width 256, GeLU activation) so H_c(z) ~= HOG_target, using clean demonstration pairs; freeze RGB encoder. H_ema is the frozen EMA teacher (decay 0.999) of H_train's HOG predictor, also frozen.
3. Calibration: on clean held-out episodes (N_cal = 512 episodes), compute residual r = ||H_c(z) - H_ema(z)|| on the clean manifold; fit threshold tau = 95th percentile of r, and fit a linear corrector W (ridge regression, lambda=1e-2) mapping (z, r) -> z_clean - z on clean data. Additionally, generate paired clean/corrupted features using the same corruption suite used at test time (four corruption types, severity levels 1-5) and refit W on the union of clean and corrupted pairs, so the correction direction is estimated where it is actually applied.

# Inference
4. Given corrupted observation, compute z = RGB_encoder(o).
5. Compute r = ||H_c(z) - H_ema(z)||.
6. If r > tau: z_corr = z + W @ [z; r]; else z_corr = z.
7. Action head (frozen) maps z_corr -> action chunk.
```

**Why this design.** We chose to distill a small HOG-consistency head from the training-only HOG DiT rather than retrain the full DiT at inference because the DiT's cost (diffusion sampling over HOG tokens) is prohibitive for real-time control, accepting the cost that the distilled head is only an approximation of the DiT's structural signal. We chose a residual-threshold gate with a calibrated 95th-percentile threshold over a learned gate because calibrated log-probabilities are known to be miscalibrated for correctness, accepting the cost that the threshold must be re-fit per deployment domain. We chose a linear ridge corrector over a nonlinear corrector because the HOG-consistency residual is expected to be locally linear in feature space near the clean manifold, accepting the cost that large corruptions may require nonlinear correction. We chose to freeze the action head and correct only its input rather than fine-tuning the whole policy because this preserves the pretrained VLA semantics, accepting the cost that the correction cannot change the action head's inductive bias. A domain expert would not describe this as a variant of Knowledge Insulating VLA or a training-time regularizer: the mechanism operates on the deployed representation at test time and uses a residual whose direction is a correction signal, not a training gradient.

**Why it measures what we claim.** The residual r = ||H_c(z) - H_ema(z)|| measures corruption-induced deviation from the HOG-consistent manifold because the assumption is that the EMA teacher's HOG prediction is a stable structural target under corruption; this assumption fails when the corruption also shifts the teacher's prediction (e.g., global contrast changes), in which case r reflects teacher drift rather than representation corruption. The linear corrector W measures the local direction from corrupted to clean features because the assumption is that the clean-manifold offset is locally linear in (z, r); this assumption fails for large nonlocal corruptions, in which case the correction reflects a linear extrapolation rather than a manifold projection. The threshold tau measures the boundary of the clean manifold because the assumption is that clean residuals are concentrated below the 95th percentile; this assumption fails under domain shift, in which case tau reflects the calibration domain rather than the deployment domain.

## Contribution

['A test-time HOG-consistency corrector (HOG-C) that distills the training-only HOG DiT into a lightweight residual head and uses its calibrated residual to gate and correct the action-conditioning representation at inference.', 'A design principle showing that structural supervision removed at inference can be recovered as an inference-time invariance check, converting a training-time regularizer into a deployment-time correction mechanism.', 'A calibration procedure that fits the residual threshold and linear corrector on clean held-out episodes, enabling per-deployment adaptation without retraining the policy.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoboTwin-Plus corruption suite | Four controlled corruptions isolate representation-level failure. |
| Primary metric | Success rate under corruption | Directly measures control robustness, the claimed target. |
| Baseline | FastWAM (uncorrected) | Tests whether correction beats the unguarded deployed policy. |
| Baseline | HOG-WAM (training-time HOG supervision) | Tests whether test-time correction rivals structural training. |
| Baseline | Naive residual gate (no calibration) | Tests whether calibration, not gating alone, drives gains. |
| Baseline | TENT/BN-adaptation | Tests whether test-time feature adaptation alone suffices. |
| Baseline | Activation steering (linear probe direction) | Tests whether a learned steering direction rivals manifold projection. |
| Ablation-of-ours | HOG-C with linear corrector removed | Isolates the correction direction from the detection gate. |
| Ablation-of-ours | HOG-C with W fit on clean data only | Tests whether fitting W on paired clean/corrupted features matters. |
| Diagnostic | Teacher drift ||H_ema(z_corrupt) - H_ema(z_clean)|| per corruption type | Validates assumption A1 (teacher stability). |
| Diagnostic | Held-out regression residuals of (z, r) -> z_clean - z | Validates assumption A2 (local linearity). |
| Control | Synthetic head-approximation error injected on clean inputs | Shows r stays below tau under head error alone. |

### Why this setup validates the claim

The central claim is that a test-time, HOG-consistency-gated linear correction of the shared RGB features recovers control performance lost to visual corruption without retraining and without disturbing clean-scene behavior. FastWAM is the right baseline for the primary sub-claim: it is the deployed policy without any structural signal, so any gap between FastWAM and HOG-C on corrupted subsets is attributable to the corrector alone. HOG-WAM tests a stronger sub-claim — that a cheap distilled, inference-time surrogate can approach the benefit of training-time HOG supervision — so HOG-C closing part of the FastWAM–HOG-WAM gap is the predicted signature, while matching HOG-WAM would be a surprise. The uncalibrated-gate baseline tests whether the 95th-percentile calibration, rather than mere gating, is doing the work. TENT/BN-adaptation and activation steering test whether the gain is specific to HOG-consistency manifold projection or is generic test-time feature adaptation. The ablation (gate retained, corrector removed) separates detection from correction: if gains persist without the corrector, the mechanism is misattributed. The ablation with W fit on clean data only tests whether the paired clean/corrupted calibration is necessary. The two diagnostic plots directly test the load-bearing assumptions A1 and A2 before any success-rate claims: teacher drift per corruption type validates that H_ema is a stable target, and held-out regression residuals validate that the clean-manifold offset is locally linear. The control experiment injecting synthetic head-approximation error on clean inputs shows that r stays below tau under head error alone, confirming that r measures representation corruption rather than distilled-head approximation error. Success rate under corruption is the right metric because the claim is about behavior, not feature statistics; a feature-space metric could improve while actions stay wrong. Clean-scene success is the control condition that must stay at parity, making the design falsifiable in both directions.

### Expected outcome and causal chain

**vs. FastWAM (uncorrected)** — On a case where sensor noise perturbs the RGB input so the encoder's shared features drift off the clean manifold, FastWAM feeds those drifted features straight into the frozen action head and emits an action chunk aimed at a mislocalized target, because it has no signal telling it the representation is corrupted. Our method computes the HOG-consistency residual, sees it exceed the calibrated threshold, and applies the ridge correction that nudges the feature back toward the clean manifold before the same frozen head acts. We expect a clear success-rate gap concentrated on the noisy subsets, with near-parity on clean scenes.

**vs. HOG-WAM (training-time HOG supervision)** — On a case where mixed noise corrupts the observation, HOG-WAM's training-time HOG DiT has shaped the encoder to be robust, so it usually succeeds; our distilled head is only an approximation and may under-correct on the hardest corruptions. Where HOG-WAM still fails — e.g. corruption types its training distribution underweighted — our test-time gate can fire per-instance and correct, because it operates on the deployed representation rather than a fixed training prior. We expect HOG-C to land between FastWAM and HOG-WAM overall, with possible parity or slight edge on corruption types absent from HOG-WAM's training mix.

**vs. Naive residual gate (no calibration)** — On a case where an in-distribution clean observation produces a moderately large residual because the distilled head is imperfect, the uncalibrated gate fires spuriously and applies a correction to an already-clean feature, degrading an otherwise correct action. Our calibrated 95th-percentile threshold is fit on clean held-out episodes precisely so these benign residuals fall below it, so the corrector stays dormant on clean inputs. We expect the naive gate to show a clean-scene success drop and a smaller corrupted-subset gain, while ours shows clean-scene parity plus the corrupted-subset gain.

**vs. TENT/BN-adaptation** — On a case where corruption shifts feature statistics, TENT adapts normalization parameters to the corrupted batch, which may recover some performance but can drift the representation away from the clean manifold because it optimizes entropy, not HOG-consistency. Our method uses an explicit HOG-consistency residual and a calibrated correction direction, so it targets the structural property that TENT ignores. We expect TENT to show smaller gains on corruption types where feature statistics shift without entropy change, and our method to outperform it on those subsets.

**vs. Activation steering (linear probe direction)** — On a case where corruption shifts features along a direction correlated with the probe, activation steering applies a fixed correction that may over- or under-correct because it lacks a per-instance residual gate. Our method gates on the calibrated residual and applies a correction whose direction is estimated from paired clean/corrupted features, so it adapts per instance. We expect activation steering to show clean-scene degradation and smaller corrupted-subset gains, while ours maintains clean-scene parity.

### What would falsify this idea

If HOG-C's gain over FastWAM is uniform across clean and corrupted subsets rather than concentrated on the corruption types where the residual gate should fire, the correction is not acting through the claimed corruption-detection mechanism. Likewise, if the ablation without the linear corrector recovers the same gains, the residual gate alone — not the manifold correction — is responsible, and the central claim is wrong. Additionally, if the diagnostic plots show teacher drift ||H_ema(z_corrupt) - H_ema(z_clean)|| is large for a corruption type where HOG-C claims a gain, assumption A1 is violated and the residual reflects teacher drift rather than representation corruption.

## References

1. HOG-WAM: Improving Visual Robustness through Future Gradient Prediction

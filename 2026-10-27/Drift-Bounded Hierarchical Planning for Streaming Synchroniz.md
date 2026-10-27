# Drift-Bounded Hierarchical Planning for Streaming Synchronized Spatial Audio

## Motivation

Existing streaming spatial audio generators such as SwanSphere rely on a single-pass causal autoregressive diffusion transformer with SVAC alignment, which has no explicit mechanism to bound synchronization drift over long horizons; the paper itself notes that the causal AR diffusion transformer lacks hierarchical planning or memory, so small per-step timing errors accumulate and the audio-visual sync anchor is never explicitly represented or corrected. This is a structural failure: without a coarse event-timestamp plan and a trajectory-level invariant, the low-level decoder can only react locally and cannot recover from drift once it exceeds the receptive field.

## Key Insight

Synchronization drift over long horizons is bounded not by adding memory to the decoder but by making the coarse event-timestamp plan an explicit latent variable whose drift is regularized by a learned Lipschitz-style bound on the audio-video latent trajectory, so the planner and decoder are coupled through a shared invariant rather than a controller.

## Method

**What it is:** DriftBounded-Swan (DBS) is a two-level generative model for streaming synchronized spatial audio. A high-level planner emits coarse audio-visual synchronization anchors (onset/event timestamps with associated visual-event embeddings) at 5 Hz; a low-level causal autoregressive diffusion transformer (the SwanSphere backbone) consumes these anchors as conditioning tokens and generates spatial audio at 50 Hz in a streaming manner. A contrastive sync-consistency loss enforces a bounded Lipschitz invariant between audio and video latent trajectories across the context window. **Load-bearing assumption (stated explicitly):** the Euclidean distance d(t)=||a_traj(t)-v_traj(t)||_2 between the frozen SVAC-aligned video encoder's latents and the audio decoder's latents is monotone in perceptual audio-visual sync error, so that penalizing its temporal derivative bounds real synchronization drift. This assumption is not taken for granted: we calibrate and verify it on held-out data (see 'Calibration and verification' below), and if the correlation between d(t) and measured onset error is weak (Spearman ρ < 0.5), we replace raw latent Euclidean distance with a learned calibrated sync-distance head (a 2-layer MLP, hidden=256, GeLU, trained on annotated onset timestamps with an L1 loss to predict |onset error| in ms) and regularize the derivative of that calibrated scalar instead of raw latent distance.

**How it works (pseudocode):**
```python
# Inputs: panoramic video V (streaming), text prompt T
# Output: spatial audio A (streaming, 48kHz stereo)

# --- High-level planner (5 Hz) ---
# Encoder E_v: video frame -> visual latent z_v (frozen SVAC-aligned encoder)
# Planner P: 4-layer Transformer (d_model=512, 8 heads, FFN=2048) over sequence of z_v + text tokens -> anchor sequence {(t_i, e_i)} where
#   t_i = onset timestamp (regressed as continuous value in seconds, L1 loss), e_i = event embedding (256-d, visual-semantic)

# --- Low-level causal AR diffusion transformer (50 Hz) ---
# At each streaming step s:
#   context = [past audio latents, current anchor tokens, text tokens]
#   x_s ~ DiffusionTransformer(context, causal_mask)  # one denoising step per frame

# --- Training losses ---
# L_diff: standard diffusion denoising loss on audio latents
# L_plan: timestamp regression loss on (t_i) (L1) and contrastive loss on (e_i) (InfoNCE, τ=0.07)
# L_sync: contrastive sync-consistency loss with drift-bound regularizer

# Drift-bound regularizer:
# For each context window W of length L = 2 seconds (100 frames at 50 Hz):
#   a_traj = audio latent trajectory (from decoder)
#   v_traj = video latent trajectory (from frozen encoder)
#   d(t) = || a_traj(t) - v_traj(t) ||_2  # per-frame distance (or calibrated head output if ρ < 0.5)
#   L_sync = sum_{t in W} [ max(0, d(t) - d(t-1) - beta) + gamma * d(t) ]  # beta = 0.05 (learned tolerance), gamma = 0.1 (absolute-drift penalty)
#   + contrastive term: pull d(t) toward 0 for positive pairs (same event), push d(t) > margin (margin=0.2) for negative pairs (shuffled events)

# Total loss: L = L_diff + lambda_1 * L_plan + lambda_2 * L_sync
# lambda_1 = 0.1, lambda_2 = 0.5 (tuned on validation)
```

**Calibration and verification (for the load-bearing assumption):** On a held-out calibration set of 512 annotated onset events from SpatialAudio-Sync, we compute d(t) for each frame and correlate it against the absolute onset error (in ms) between generated audio events and ground-truth visual onsets. We report Spearman ρ. If ρ ≥ 0.5, we use raw d(t) as the sync proxy. If ρ < 0.5, we train a calibrated sync-distance head: a 2-layer MLP (input: concatenation of a_traj(t) and v_traj(t), hidden=256, GeLU, output: scalar) with L1 loss to predict |onset error| in ms, trained on the same 512 calibration events, and use its output as d(t) in L_sync. This head is frozen during decoder training.

**Why this design:** We chose a high-level planner that emits explicit timestamp anchors over a learned memory bank in the decoder because timestamps are a discrete, interpretable sufficient statistic for synchronization, accepting the cost that the planner must be trained with timestamp supervision (which we obtain from the automated annotation pipeline). We chose a Lipschitz-style drift-bound regularizer over a simple MSE alignment loss because the regularizer penalizes *changes* in the audio-video distance rather than absolute distance, which prevents the decoder from ignoring the planner's anchors when the video stream is noisy, accepting the cost that the regularizer requires a context window of at least 2 seconds to estimate the derivative. We chose a contrastive sync-consistency loss over a discriminator because contrastive pairs are easier to construct from the annotation pipeline (positive: same event, negative: shuffled event) and avoid adversarial training instability, accepting the cost that the margin must be tuned per dataset. Unlike the two-module controller of prior work (e.g., Toolformer-style routing), DBS has no controller: the planner's anchors are *conditioning tokens* for the decoder, and the drift-bound regularizer couples their latent trajectories directly, so the planner and decoder are trained end-to-end with a shared invariant.

**Why this measures what we claim:** The computational quantity `d(t) = ||a_traj(t) - v_traj(t)||_2` measures *audio-visual synchronization* because we assume that the SVAC-aligned video encoder and the audio decoder share a latent space where Euclidean distance is monotone in perceptual sync error; this assumption fails when the video encoder's latent space is not isometric to the audio latent space (e.g., due to domain shift in panoramic video), in which case `d(t)` reflects encoder misalignment rather than sync error. We explicitly test this assumption via the calibration procedure above (Spearman ρ on 512 held-out events) and fall back to a learned calibrated sync-distance head if ρ < 0.5. The drift-bound term `max(0, d(t) - d(t-1) - beta) + gamma * d(t)` measures *bounded synchronization drift* because we assume that the per-frame distance derivative is a sufficient statistic for drift accumulation over the context window; this assumption fails when the drift is non-monotonic (e.g., oscillatory sync errors), in which case the regularizer may allow large absolute drift if it is locally flat. The added absolute-drift penalty `gamma * d(t)` (gamma=0.1) closes this failure mode by penalizing absolute distance, not just its derivative. The contrastive term measures *event-level sync consistency* because we assume that shuffling event timestamps is a valid intervention that breaks sync without changing audio content; this assumption fails when the audio content itself is event-agnostic (e.g., ambient noise), in which case the contrastive loss provides no gradient.

## Contribution

['A hierarchical latent planner that emits coarse audio-visual synchronization anchors as explicit conditioning tokens for a causal autoregressive diffusion transformer, enabling streaming spatial audio generation with a coarse-to-fine sync plan.', 'A contrastive sync-consistency loss with a learned drift-bound regularizer that enforces a bounded Lipschitz-style invariant between audio and video latent trajectories, directly operationalizing bounded synchronization drift over long horizons.', 'An end-to-end training procedure that jointly optimizes the planner, decoder, and sync invariant without a separate controller, demonstrating that explicit anchor conditioning and trajectory-level regularization are jointly necessary for long-horizon sync stability.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | SpatialAudio-Sync (panoramic video + annotated onsets) | Provides timestamp anchors and event labels for supervised sync. |
| Primary metric | Event-level sync F1 (onset alignment within 50ms) | Directly measures claimed improvement in synchronization. |
| Secondary metric | Drift slope: linear regression of sync error (ms) vs. stream length (0–60s) | Quantifies bounded drift, not just thresholded F1. |
| Baseline | Non-streaming spatial audio diffusion (SwanSphere offline) | Tests whether streaming causal design sacrifices sync. |
| Baseline | Streaming audio-only AR model (no visual anchors) | Tests whether visual anchors are necessary for sync. |
| Baseline | Two-module controller (Toolformer-style routing) | Tests whether end-to-end anchor conditioning beats routing. |
| Baseline | Learned-memory decoder (recurrent/attention memory bank on SwanSphere backbone) | Tests whether explicit anchors beat memory augmentation. |
| Ablation-of-ours | DBS without drift-bound regularizer | Isolates contribution of Lipschitz sync-consistency loss. |
| Ablation-of-ours | DBS without absolute-drift penalty (gamma=0) | Isolates contribution of absolute-drift term. |
| Ablation-of-ours | DBS with raw latent distance vs. calibrated head | Tests the load-bearing assumption directly. |

### Why this setup validates the claim

The central claim is that DBS achieves bounded synchronization drift in streaming spatial audio by coupling a high-level planner's timestamp anchors with a low-level diffusion transformer via a drift-bound regularizer. The dataset SpatialAudio-Sync provides ground-truth onset timestamps and event labels, enabling a direct, falsifiable test of sync accuracy. The primary metric, event-level sync F1 at 50ms tolerance, directly measures alignment between generated audio events and visual onsets—the exact quantity the drift-bound regularizer is designed to preserve. The secondary metric, drift slope over 0–60s, quantifies whether drift is bounded (slope near zero) rather than merely low on average, speaking to the structural failure named in the motivation. The non-streaming baseline tests whether the streaming constraint inherently harms sync; if DBS matches or exceeds it, the claim that streaming need not sacrifice sync is supported. The streaming audio-only baseline tests whether visual anchors are necessary; if DBS outperforms it, the planner's anchors are validated as a sufficient statistic for sync. The two-module controller baseline tests whether end-to-end anchor conditioning is superior to routing; a win here supports the design choice of shared invariants. The learned-memory decoder baseline tests whether explicit anchor latents plus the drift invariant beat memory augmentation, isolating what is genuinely new. The ablation without the drift-bound regularizer isolates its contribution: if removing it causes a disproportionate drop in sync F1 on noisy or long-event subsets, the regularizer's role in preventing drift is confirmed. The ablation without the absolute-drift penalty (gamma=0) tests whether the added term closes the non-monotonic drift failure mode. The ablation comparing raw latent distance vs. calibrated head directly tests the load-bearing assumption: if the calibrated head yields significantly higher sync F1, the raw Euclidean distance is not monotone in perceptual sync error. Together, these comparisons form a falsifiable test: the predicted pattern is that DBS's advantage is concentrated on subsets where sync drift is likely (e.g., long continuous events, noisy video), not uniform across all data.

### Expected outcome and causal chain

**vs. Non-streaming spatial audio diffusion (SwanSphere offline)** — On a case with a long continuous event (e.g., a moving sound source over 10 seconds), the offline baseline can globally optimize alignment and thus maintain sync, but it cannot operate in streaming mode. Our method, while streaming, uses the planner's anchors as conditioning tokens and the drift-bound regularizer to prevent error accumulation, so we expect DBS to achieve sync F1 close to the offline baseline (within a small margin, e.g., ≤5 F1 points) on such events, with the gap widening only for very long contexts (≥30s) where the offline model has full future access. The observable signal is near-parity on short events and a modest gap on the longest events, demonstrating that streaming does not catastrophically degrade sync. Additionally, the drift slope for DBS should be near zero (≤1 ms/s), while the offline baseline has no drift by construction.

**vs. Streaming audio-only AR model (no visual anchors)** — On a case where a visual event (e.g., a door slam) occurs without a strong audio cue in the prompt, the audio-only baseline has no mechanism to align its generated audio to the visual onset, so it produces audio with random phase relative to the event, yielding low sync F1. Our method instead receives the planner's anchor token at that timestamp and the drift-bound regularizer pulls the audio latent trajectory toward the video latent trajectory, so we expect a large sync F1 gain (e.g., ≥15 F1 points) on visually salient events, with parity on events that are also strongly indicated in the text prompt. The signal is a concentrated improvement on the visual-only event subset.

**vs. Two-module controller (Toolformer-style routing)** — On a case with a noisy video segment (e.g., rapid camera motion), the controller baseline may misroute or ignore the visual stream due to routing errors, causing sync drift. Our method has no controller; the planner's anchors are always conditioning tokens, and the drift-bound regularizer penalizes changes in audio-video distance even when the video is noisy, so we expect DBS to maintain lower drift (higher sync F1, e.g., ≥10 F1 points) on noisy segments, while both methods perform similarly on clean segments. The observable signal is a gap that emerges specifically under video noise.

**vs. Learned-memory decoder (recurrent/attention memory bank on SwanSphere backbone)** — On long-horizon segments (≥30s), the memory-augmented decoder can recall past context but has no explicit timestamp plan or drift invariant, so its sync error may still accumulate. Our method uses explicit anchor latents and a drift-bound regularizer, so we expect DBS to show a lower drift slope (e.g., ≤1 ms/s vs. ≥3 ms/s for memory) and higher sync F1 on the longest segments, while both methods perform similarly on short segments. The observable signal is a divergence in drift slope as stream length increases.

### What would falsify this idea

If the ablation (DBS without drift-bound regularizer) shows no significant drop in sync F1 on noisy or long-event subsets, then the regularizer is not responsible for bounded drift and the central claim fails. Similarly, if DBS's gain over the audio-only baseline is uniform across all events rather than concentrated on visually salient ones, the planner's anchors are not the causal mechanism. If the calibrated sync-distance head does not outperform raw latent distance (i.e., Spearman ρ ≥ 0.5 and similar F1), the load-bearing assumption is validated; if ρ < 0.5 and the calibrated head yields no improvement, the latent-distance proxy is invalid and the method's sync mechanism is unsupported.

## References

1. Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer
2. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
3. Boosting Latent Diffusion with Flow Matching
4. Analyzing and Improving the Training Dynamics of Diffusion Models
5. Flow Matching for Generative Modeling
6. Scalable Diffusion Models with Transformers

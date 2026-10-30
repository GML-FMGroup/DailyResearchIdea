# Reconstruction-Probed Omni-Encoder Post-Training for Modality-Preserving Shared-Weight Fusion

## Motivation

Unified omni-modal parsers such as Youtu-Parsing-Omni encode all modalities with a single shared-weight encoder and optimize only token-level parsing targets via OSAD, so nothing in the training signal forces the shared token space to retain modality-specific structure; cross-modal fusion invariance is assumed rather than established. The structural root cause is that token-level supervision is a sufficient statistic for the parsing output but not for the input modality, so the encoder can collapse distinct modality features that the parser never needs to distinguish. This is a meta-level failure recurring across shared-weight multimodal systems (the Qwen3-Omni and Qwen-Audio lines likewise rely on multi-task tags or scale to implicitly avoid interference, without a probe that detects representational collapse).

## Key Insight

A modality-reconstruction probe makes modality-specific information a trained invariant of the shared encoder: because the decoder is lightweight and cannot itself regenerate spectrograms or patches, the reconstruction loss is minimized only if the encoder's token representations remain an information-sufficient, invertible encoding of each modality, converting an unverified fusion assumption into a measurable and penalized property.

## Method

**What it is.** We introduce RPO (Reconstruction-Probed Omni post-training), a regularizer added to OSAD post-training of a shared-weight omni-encoder. Inputs: a batch of (modality m, raw input x_m, token-level target y_m) triples from the ten parsing tasks. Outputs: the updated encoder weights θ and, per modality, a small reconstruction decoder D_m. The training objective is L = L_OSAD + λ Σ_m L_rec^m, where L_rec^m = ||D_m(z_m) − f_m(x_m)||² and z_m are the encoder's token representations for modality m.

**Load-bearing assumption (stated explicitly).** The probe is valid only if a 2-layer, ~1M-parameter MLP decoder trained with MSE on frozen raw targets (log-mel, patches) can be minimized only when the shared encoder's token representations retain the modality-specific information the parser needs. This is assumption A: that linearly-decodable structure in f_m(x_m) is a superset of the parser-relevant detail. It fails (failure mode F) when parser-relevant cues (e.g., fine phonetic transitions, glyph-level strokes) are nonlinearly encoded or lie in a subspace that f_m discards, in which case L_rec^m reflects only the retained subset and under-penalizes collapse in the discarded subspace. We therefore add a calibration step (step 4 below) that measures whether L_rec^m tracks parser-relevant retention, and a nonlinear-decoder control (step 5) that bounds F.

**How it works.**
```pseudocode
for each OSAD round r:
    θ_ref ← copy of current encoder weights        # reference-conditioned copy (OSAD)
    for each minibatch B = {(m, x_m, y_m)}:
        # 1. shared-weight encoding (unchanged from Youtu-Parsing-Omni)
        z_m = Encoder_θ(x_m)                       # tokens in unified space-time coords
        # 2. OSAD token-level objective (unchanged)
        L_osad = CrossEntropy(Parser_θ(z_m), y_m, ref=θ_ref)
        # 3. modality-reconstruction probe (NEW)
        for each modality m present in B:
            target_m = f_m(x_m)                    # frozen modality-specific features:
                                                   #   log-mel spectrogram (audio, 80 mel bins × T frames),
                                                   #   patch pixels (image/video frame, 16×16 patches),
                                                   #   byte-normalized text embedding (text, 768-d)
            recon_m = D_m(z_m)                     # D_m = 2-layer MLP, hidden 256, GeLU, ~1M params
            L_rec_m = MSE(recon_m, target_m)
        L = L_osad + λ * mean_m(L_rec_m)           # λ = 0.1, linearly warmed from 0 to 0.1 over 2k steps
        θ ← θ − η ∇_θ L                            # η = 1e-5; D_m updated jointly
    θ_ref ← θ                                       # OSAD refresh

# 4. calibration (run every 500 steps, on a held-out calibration set of 512 examples)
#    - train a linear probe P_m from z_m to parser-relevant labels (e.g., token-level
#      glyph/phoneme class) on the calibration set; record linear-probe accuracy acc_lin
#    - record L_rec_m on the same set
#    - report Pearson correlation r(L_rec_m, acc_lin) across checkpoints;
#      if |r| < 0.3, flag that L_rec_m does not track parser-relevant retention

# 5. nonlinear-decoder control (run once at end of post-training)
#    - replace D_m with a 4-layer MLP, hidden 512, GeLU (~5M params) and re-fit on frozen z_m
#    - report ΔMSE = MSE_linear − MSE_nonlinear; large ΔMSE bounds failure mode F
#      (parser-relevant detail that the linear probe cannot see)
```
Reconstruction targets f_m are frozen, non-learned transforms so the probe cannot cheat by co-adapting a target representation. Decoders are discarded after post-training; only the encoder and parser are deployed.

**Why this design.** We chose a small MLP decoder over a generative decoder because a high-capacity decoder could reconstruct targets from collapsed representations, defeating the probe; the cost is that the MLP can only invert linearly-decodable structure, which is exactly the structure we want preserved. We chose frozen modality-specific targets (log-mel, raw patches) over learned targets because learned targets admit co-adaptation between encoder and probe, which would let the encoder satisfy the loss without retaining information; the cost is that raw targets are high-dimensional and noisier, so λ must be kept small (0.1) to avoid dominating the parsing loss. We chose to add the probe to OSAD rather than a separate pretraining stage because OSAD's on-policy rollouts shift the encoder's representation distribution every round, so a probe trained only once would go stale; the cost is per-step compute roughly 1.15×, since the probe is ~1M params versus the 5B encoder. Unlike Qwen-Audio's hierarchical-tag multi-task conditioning, which avoids interference by partitioning the decoder input rather than by verifying encoder-side retention, RPO directly penalizes information loss in the shared representation, so a domain expert would not describe it as a variant of tag-conditioned multi-task training.

**Why it measures what we claim.** The reconstruction MSE L_rec^m measures modality-specific information retention in the shared encoder because the assumption is that a lightweight decoder can invert z_m to f_m(x_m) only if z_m is a sufficient statistic for x_m's modality features; this assumption fails when f_m discards information the parser needs (e.g., log-mel loses fine phonetic cues), in which case L_rec reflects only the retained subset and under-penalizes collapse in the discarded subspace. The frozen target f_m measures modality identity because the assumption is that f_m is not co-adapted with the encoder; this fails when f_m and the encoder share pretrained weights, in which case the probe measures shared-pretraining alignment rather than post-training retention. The decoder capacity (2-layer MLP) measures invertibility because the assumption is that linearly-decodable structure is the structure the parser relies on; this fails for structure requiring nonlinear decoding, in which case low MSE overstates retention. The calibration step (step 4) directly tests the first assumption by correlating L_rec^m with linear-probe accuracy on parser-relevant labels; the nonlinear-decoder control (step 5) bounds the third assumption by quantifying the MSE gap between linear and nonlinear decoders.

## Contribution

(1) RPO, a modality-reconstruction probe regularizer for OSAD post-training of shared-weight omni-encoders, with frozen modality-specific targets and lightweight per-modality decoders. (2) The design principle that a frozen-target, capacity-limited probe converts the unverified fusion-invariance assumption of shared-weight omni-encoders into a trained, measurable property, with a warm-up schedule that keeps it from dominating token-level parsing. (3) An analysis of when reconstruction MSE is and is not a valid proxy for modality-specific retention, identifying the target-frozenness and decoder-capacity conditions under which it fails.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Ten-task omni parsing suite (Youtu-Parsing-Omni) | Matches the ten parsing tasks the probe regularizes |
| Primary metric | Per-task parsing accuracy on rare/ambiguous subsets | Detects retention loss where collapse is most likely |
| Secondary metric | Pearson r between L_rec^m and linear-probe accuracy on parser-relevant labels (calibration set of 512 examples) | Tests whether reconstruction MSE tracks parser-relevant retention |
| Baseline | OSAD post-training without RPO | Isolates the probe's contribution to retention |
| Baseline | Qwen3-Omni | Strong omni model without encoder-side retention penalty |
| Baseline | Qwen-Audio tag-conditioned multi-task | Tests whether partitioning, not verification, suffices |
| Ablation-of-ours | RPO with learned (co-adapting) targets | Tests the frozen-target claim |
| Ablation-of-ours | RPO with 4-layer, hidden-512 nonlinear decoder (control) | Bounds failure mode F (nonlinearly encoded parser-relevant detail) |

### Why this setup validates the claim

The central claim is that a frozen-target reconstruction probe added to OSAD prevents modality-specific information collapse in the shared encoder, and that this collapse would otherwise appear as a degradation concentrated on the parsing tasks that depend on fine-grained modality detail. The no-RPO OSAD baseline tests the core sub-claim directly: if the probe is doing what we say, removing it should reproduce the collapse pattern, and adding it should suppress it. Qwen3-Omni tests whether the effect is an artifact of our particular post-training recipe or reflects a general gap in omni encoders that lack an encoder-side retention signal. Qwen-Audio's tag-conditioned multi-task approach tests the specific mechanism claim that partitioning decoder inputs avoids interference without verifying encoder-side retention, so it should behave like the no-probe baseline on the retention-sensitive subsets. The learned-target ablation tests the design choice that frozen targets are necessary: if co-adapting targets perform equally well, the probe is not measuring retention. The nonlinear-decoder control tests the load-bearing assumption A: if a 4-layer decoder achieves substantially lower MSE than the 2-layer MLP on the same frozen z_m, then linearly-decodable structure is not a superset of parser-relevant detail, and the probe's MSE understates retention. The secondary metric—correlation between L_rec^m and linear-probe accuracy on parser-relevant labels—directly operationalizes the claim that reconstruction MSE tracks the information the parser uses; a high correlation supports the probe as a valid proxy, while a low correlation falsifies it. Per-task accuracy on rare and ambiguous subsets is the right primary metric because collapse is predicted to be silent on head distributions and visible only where the parser must rely on modality-specific detail that the shared encoder could otherwise discard.

### Expected outcome and causal chain

**vs. OSAD without RPO** — On a case where a document page contains a faint handwritten annotation over printed text, the no-probe baseline produces a transcription that drops or misreads the annotation because OSAD's token-level objective, driven by head-distribution parsing targets, lets the shared encoder discard low-frequency modality detail that contributes little to average loss. Our method instead forces the encoder's token representations to remain linearly invertible to raw patch features because the frozen-target MSE penalizes any information the decoder cannot recover, so we expect a noticeable accuracy gap on the rare/ambiguous subset with near-parity on the clean head subset. The calibration curve should show L_rec^m decreasing as linear-probe accuracy on glyph labels increases, with Pearson r > 0.5 across checkpoints.

**vs. Qwen3-Omni** — On a case where an audio clip carries a quiet disfluency that changes the parsed transcript's meaning, Qwen3-Omni produces a fluent but semantically shifted parse because its omni training optimizes downstream token likelihood without an explicit encoder-side retention constraint, so fine acoustic detail can be smoothed away. Our method instead keeps the log-mel-recoverable structure in the shared representation because the probe penalizes its loss, so we expect our advantage to concentrate on the audio parsing tasks with disfluent or noisy inputs rather than on clean speech. The nonlinear-decoder control should show a small ΔMSE (≤10% of linear MSE), indicating that the linear probe captures most of the parser-relevant acoustic detail.

**vs. Qwen-Audio tag-conditioned multi-task** — On a case where an image region's layout and its caption must both be parsed, the tag-conditioned baseline produces a layout-caption mismatch because it avoids interference by partitioning decoder inputs rather than by ensuring the encoder retains both modalities' detail, so one modality's features can be under-represented in the shared space. Our method instead penalizes retention loss per modality in the shared encoder because reconstruction probes are applied to each modality's tokens, so we expect a gap on mixed-modality documents that shrinks to parity on single-modality inputs.

**vs. RPO with learned targets (ablation)** — On a case where the parser needs a fine phonetic or glyph-level cue, the learned-target variant produces retention that looks good on its own probe loss but fails to transfer because the encoder and target co-adapt, letting the loss be satisfied without preserving the cue. Our method instead uses frozen targets so the probe cannot be gamed, so we expect the learned-target variant to show probe MSE comparable to ours but parsing accuracy closer to the no-probe baseline, a dissociation that confirms the frozen-target design.

**vs. RPO with nonlinear decoder (control)** — On a case where the parser relies on high-frequency glyph strokes, the 2-layer MLP probe may under-penalize their loss because they are nonlinearly encoded. The 4-layer decoder control quantifies this: if ΔMSE is large (>30% of linear MSE), the linear probe is missing parser-relevant detail, and we expect the nonlinear-decoder variant to yield higher accuracy on the rare/ambiguous subset, bounding failure mode F.

### What would falsify this idea

If the gain over the no-probe baseline were uniform across all parsing subsets rather than concentrated on the rare, ambiguous, and mixed-modality subsets where collapse is predicted, the claim that the probe prevents modality-specific information loss would be wrong. Likewise, if the learned-target ablation matched our frozen-target variant on those subsets, the co-adaptation argument for frozen targets would be falsified. Additionally, if the correlation between L_rec^m and linear-probe accuracy on parser-relevant labels were low (|r| < 0.3) or if the nonlinear-decoder control showed a large ΔMSE without a corresponding accuracy gain, the load-bearing assumption A would be falsified.

## References

1. Youtu-Parsing-Omni: One Encoder, One Schema, All Modalities
2. Qwen3-Omni Technical Report
3. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
4. Qwen2-Audio Technical Report
5. Qwen Technical Report
6. Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models
7. Finite Scalar Quantization: VQ-VAE Made Simple
8. Speak While You Think: Streaming Speech Synthesis During Text Generation

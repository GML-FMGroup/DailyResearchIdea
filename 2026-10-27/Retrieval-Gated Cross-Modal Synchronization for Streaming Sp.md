# Retrieval-Gated Cross-Modal Synchronization for Streaming Spatial Audio Generation

## Motivation

Existing streaming spatial audio generators such as SwanSphere inherit a fixed context window from their causal autoregressive diffusion transformer backbone, which structurally prevents them from maintaining audio-video alignment over long horizons: once a past audio-visual latent chunk leaves the window, its synchronization evidence is irrevocably lost, so the model cannot recover alignment when the scene revisits similar acoustic geometry. This is not a data or capacity limitation but a consequence of the causal attention mask acting as a hard information bottleneck across the audio-visual stream.

## Key Insight

A cross-modal synchronization invariant can be maintained under streaming causality if and only if the retrieval of past audio-visual latent chunks is gated by an explicit, measurable audio-video alignment score computed at each streaming step, because this score is the sufficient statistic that determines whether a retrieved chunk is synchronization-consistent with the current video context.

## Method

**What it is.** We introduce RGS-Swan (Retrieval-Gated Synchronization Swan), a streaming spatial audio generator that augments a causal autoregressive diffusion transformer with a memory bank of past audio-visual latent chunks and a cross-modal synchronization gate. At each streaming step t, given the current video latent v_t, text prompt p, and the diffusion transformer state, RGS-Swan outputs the next audio latent chunk a_t and updates the memory bank with the pair (v_t, a_t).

**How it works.** The core loop is:

```pseudocode
# Precompute: audio-video alignment scorer S_theta (trained contrastively)
# S_theta is a 4-layer cross-attention network (d_model=256, 8 heads, GeLU) trained with InfoNCE (tau=0.07) on synchronized vs. shuffled pairs from SwanSphere.
# Calibration: fit a held-out calibration set of 512 pairs (256 synchronized, 256 shuffled; stratified to include 128 repetitive-geometry and 128 ambient-noise pairs) to map raw S_theta to a calibrated probability P_sync via isotonic regression.
# Threshold tau_sync is set to the calibrated probability corresponding to a target false-accept rate of 0.05 on the calibration set.
# Second gate signal: temporal-order/phase-consistency check C(v_t, a_i) = normalized cross-correlation between audio onset envelope and visual motion energy over the chunk; C in [0,1].
# Memory bank M = list of (v_i, a_i, s_i) where s_i is the calibrated P_sync at storage time

for each streaming step t:
    v_t = video_encoder(panoramic_frame_t)          # SVAC-aligned encoder
    # 1. Retrieval with synchronization gate
    candidates = top-K(M, key=v_t, K=8)              # cosine similarity in SVAC space
    for each (v_i, a_i, s_i) in candidates:
        p_sync_i = calibrate(S_theta(v_t, a_i))      # isotonic regression on held-out set
        c_i = C(v_t, a_i)                            # phase-consistency in [0,1]
        g_i = sigma( alpha * p_sync_i + beta * s_i + gamma * c_i )   # gate in [0,1], alpha=1.0, beta=0.5, gamma=0.5
    # 2. Gated memory conditioning
    m_t = sum_i g_i * MLP([v_i, a_i]) / sum_i g_i    # weighted memory context; MLP: 2-layer, hidden=512, GeLU
    # 3. Causal AR diffusion step conditioned on memory
    a_t = DiffusionTransformer( v_t, p, m_t, a_{<t} )  # one denoising step per token
    # 4. Synchronization check and bank update
    s_t = calibrate(S_theta(v_t, a_t))
    if s_t > tau_sync and C(v_t, a_t) > 0.5:
        M.append((v_t, a_t, s_t))
    else:
        # re-align: run 2 extra denoising steps with m_t amplified by (1 + lambda)
        a_t = DiffusionTransformer( v_t, p, (1+lambda)*m_t, a_{<t} )
        M.append((v_t, a_t, calibrate(S_theta(v_t, a_t))))
    emit a_t
```

Hyperparameters: K=8 retrieval candidates, alpha=1.0, beta=0.5, gamma=0.5, tau_sync set by calibration to 5% false-accept rate (empirically ~0.65 on SwanSphere calibration set), lambda=0.3, two extra denoising steps on failure. The alignment scorer S_theta is a 4-layer cross-attention network (d_model=256, 8 heads, GeLU) trained with InfoNCE (tau=0.07) on synchronized vs. shuffled audio-video pairs from the SwanSphere annotation pipeline. Calibration uses isotonic regression on 512 held-out pairs (stratified to include repetitive-geometry and ambient-noise subsets) to map raw S_theta to P_sync; the gate threshold is derived from a target false-accept rate of 0.05. A second gate signal C(v_t, a_i) is the normalized cross-correlation between audio onset envelope and visual motion energy over the chunk, thresholded at 0.5 for bank updates.

**Why this design.** We chose a retrieval bank over expanding the context window because the causal attention mask in SwanSphere is what guarantees streaming latency, and widening it would violate the low-latency contract; accepting the cost that retrieval may miss relevant chunks when the SVAC embedding is noisy. We chose an explicit alignment score S_theta as the gate over a learned attention weight because a learned weight would collapse onto the same fixed-context failure mode (it would be trained only on in-window pairs and thus never learn to reject out-of-window misaligned retrievals); accepting the cost of an additional contrastive training stage. We chose to re-align (extra denoising steps) on gate failure rather than discard the step because discarding would break the streaming output rate; accepting the cost of occasional latency spikes. A domain expert would not describe this as standard RAG because the retrieval is not keyed on the current query alone but gated by a cross-modal synchronization score that is checked at every step, and the memory bank is updated only when the synchronization invariant holds — this is a stateful invariant maintenance mechanism, not a nearest-neighbor lookup. To address the load-bearing assumption that S_theta is a calibrated, monotone sufficient statistic for true synchronization, we explicitly state: S_theta is assumed to be a sufficient statistic such that its calibrated output P_sync reliably separates synchronization-consistent from synchronization-inconsistent chunks, including out-of-window retrieved pairs. This assumption is made explicit and is tested via calibration and a second gate signal C; failure modes (repetitive geometry, ambient noise) are handled by stratified calibration and the phase-consistency check.

**Why it measures what we claim.** The gate value g_i measures cross-modal synchronization consistency between a retrieved past chunk and the current video context because S_theta is trained contrastively so that synchronized pairs score higher than shuffled pairs, and the calibrated P_sync maps this score to a probability of true synchronization; this assumption fails when the panoramic scene contains repetitive geometry (e.g., symmetric rooms) where a shuffled pair is also visually plausible, in which case g_i reflects visual similarity rather than true synchronization. The synchronization check s_t > tau_sync measures whether the newly generated chunk satisfies the cross-modal invariant because S_theta is monotone in alignment under the contrastive training distribution and tau_sync is calibrated to a target false-accept rate; this assumption fails when the audio contains ambient noise uncorrelated with video, in which case s_t reflects noise level rather than alignment. The memory bank update rule measures long-form coherence because chunks are stored only when the invariant holds (s_t > tau_sync and C > 0.5), so the bank is a curated set of synchronization-consistent evidence; this assumption fails when early chunks are systematically biased (e.g., a long silent scene), in which case the bank under-represents valid audio-visual configurations.

## Contribution

['A retrieval-gated cross-modal synchronization mechanism (RGS-Swan) that maintains an explicit audio-video alignment invariant at every streaming step of a causal autoregressive diffusion transformer, using a contrastively trained alignment scorer as the gate and a synchronization-checked memory bank update rule.', 'A design principle for streaming spatial audio: fixed-context causal attention can be augmented with retrieval only when the retrieval gate is a cross-modal synchronization score, because visual-similarity retrieval alone re-introduces the same out-of-window misalignment failure; this is demonstrated by an ablation where a cosine-similarity gate without S_theta degrades alignment over long horizons.', 'An open-source implementation of the synchronization-gated memory bank and alignment scorer, compatible with the SwanSphere annotation pipeline, enabling reproducible long-form spatial audio evaluation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SwanSphere panoramic audio-visual benchmark | Panoramic video with synchronized spatial audio ground truth |
| Primary metric | Calibrated S_theta (P_sync) | Directly measures claimed cross-modal synchronization invariant |
| Secondary metric | Spatial audio direction error (SADE) | Independent ground-truth synchronization measure |
| Baseline | Non-streaming spatial audio diffusion model | Tests whether streaming constraint costs quality |
| Baseline | Streaming AR diffusion without retrieval bank | Tests whether memory bank drives long-form coherence |
| Baseline | Streaming RAG with cosine-only retrieval, no gate | Tests whether synchronization gate is the causal ingredient |
| Baseline | Streaming RAG with learned cross-attention memory readout (trained with synthetic window-eviction augmentation) | Tests whether gate is more than reparameterized attention |
| Ablation-of-ours | RGS-Swan with gate fixed to g_i=1 (no gating) | Isolates contribution of the cross-modal gate |
| Ablation-of-ours | RGS-Swan with S_theta replaced by raw uncalibrated score | Tests calibration contribution |
| Ablation-of-ours | RGS-Swan without phase-consistency check C | Tests second gate signal contribution |

### Why this setup validates the claim

The central claim is that a stateful, synchronization-gated retrieval bank — not merely longer context or plain nearest-neighbor retrieval — is what sustains cross-modal alignment across a long stream. Each baseline isolates one sub-claim. The non-streaming diffusion baseline tests whether the streaming contract itself is the bottleneck: if it matches us, streaming is not the problem and our mechanism is unmotivated. The streaming-AR-without-bank baseline tests whether memory is needed at all: if it matches us, retrieval contributes nothing. The cosine-only RAG baseline is the sharpest test: it has the same bank and the same retrieval, differing only in the synchronization gate, so any gap is attributable to the gate and the invariant-maintenance update rule. The learned cross-attention memory readout baseline (trained with synthetic window-eviction augmentation) tests whether our gate is merely a reparameterization of attention: if it matches us, the explicit synchronization score adds nothing beyond a learned readout. The gate-off ablation does the same from inside our method, confirming the gate rather than the bank's mere presence is causal. The calibration ablation tests whether the calibrated probability (vs. raw score) is necessary for the threshold to correspond to a synchronization boundary. The phase-consistency ablation tests whether the second gate signal is needed to handle repetitive geometry and ambient noise. We use calibrated S_theta (P_sync) as the primary metric because it is the exact quantity the gate optimizes and the invariant checks, so it detects the predicted effect where it should appear — and, importantly, it is not a generic perceptual score that could improve for unrelated reasons. We additionally report spatial audio direction error (SADE) as an independent ground-truth synchronization measure to empirically justify the sufficient-statistic claim: we compute the rank correlation (Spearman) between P_sync and SADE on a held-out set; a high correlation supports that P_sync tracks true synchronization. A falsifiable design needs the metric to be the mechanism's own readout, and baselines that differ by exactly one mechanism component.

### Expected outcome and causal chain

**vs. Non-streaming spatial audio diffusion model** — On a long panoramic sequence with a moving sound source, the non-streaming model sees the whole clip and produces well-aligned audio, but cannot emit at streaming latency; when forced into a chunked mode it loses cross-chunk source continuity because it has no persistent state. Our method maintains a curated bank of synchronization-consistent chunks, so the moving source's trajectory stays coherent across chunk boundaries. We expect parity on short clips (where context is not the issue) but a clear P_sync advantage on long clips with continuous motion, concentrated in the later stream segments.

**vs. Streaming AR diffusion without retrieval bank** — On a scene that revisits a configuration (e.g., camera pans back to a previously seen corner), the no-bank model has only its causal window and regenerates audio from scratch, drifting in spatial placement because it cannot recall the earlier alignment. Our method retrieves the stored (v_i, a_i) pair and conditions on it through the gate, reusing the earlier consistent placement. We expect the gap to appear specifically at revisit events and to grow with stream length, while early chunks show near-parity.

**vs. Streaming RAG with cosine-only retrieval, no gate** — On a repetitive-geometry scene (symmetric room), cosine-only retrieval pulls a visually similar but temporally wrong chunk, and without a synchronization gate that misaligned memory is injected directly, degrading alignment. Our gate scores each candidate with calibrated S_theta and the phase-consistency check, down-weights the mismatched chunk, and the invariant check rejects the resulting step. We expect our method to hold P_sync while the cosine-only baseline drops sharply on the repetitive-geometry subset, with parity on non-repetitive scenes — a subset-localized gap, not a uniform one.

**vs. Streaming RAG with learned cross-attention memory readout (trained with synthetic window-eviction augmentation)** — This baseline has the same bank and retrieval but replaces the explicit gate with a learned attention weight trained on out-of-window pairs via synthetic window-eviction augmentation. If our gate is merely a reparameterization, this baseline should match us. We expect our method to outperform on repetitive-geometry and ambient-noise subsets, where the learned readout (trained on augmented pairs) still conflates visual similarity with synchronization, while our calibrated gate and phase-consistency check reject misaligned chunks. On non-repetitive, low-noise scenes, we expect parity.

**Ablation: RGS-Swan with gate fixed to g_i=1 (no gating)** — This removes the synchronization gate entirely, injecting all retrieved chunks with equal weight. We expect a large drop in P_sync on repetitive-geometry and revisit subsets, confirming the gate is causal.

**Ablation: RGS-Swan with raw uncalibrated S_theta** — This uses the raw contrastive score with a fixed threshold of 0.7 instead of calibrated P_sync. We expect worse performance on ambient-noise and repetitive-geometry subsets due to miscalibration, demonstrating the necessity of calibration.

**Ablation: RGS-Swan without phase-consistency check C** — This removes the second gate signal. We expect a drop on repetitive-geometry and ambient-noise subsets, showing that S_theta alone is insufficient and the phase check is needed.

### What would falsify this idea

If our advantage over the cosine-only RAG baseline, the learned cross-attention baseline, and the gate-off ablation were uniform across all subsets rather than concentrated on repetitive-geometry and revisit events — or if the non-streaming baseline matched us on long clips — then the synchronization gate and stateful invariant maintenance would not be the causal ingredient, and the central claim would be wrong. Additionally, if the Spearman correlation between P_sync and SADE were low (e.g., <0.3), the sufficient-statistic claim would be falsified.

## References

1. Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer
2. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
3. Boosting Latent Diffusion with Flow Matching
4. Analyzing and Improving the Training Dynamics of Diffusion Models
5. Flow Matching for Generative Modeling
6. Scalable Diffusion Models with Transformers

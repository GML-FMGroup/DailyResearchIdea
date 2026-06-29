# Modality-Specific Temporal Encoding and Dynamic Fusion for Streaming Multimodal Foundation Models

## Motivation

Existing streaming multimodal foundation models like Wan-Streamer assume a shared temporal representation across modalities, using uniform tokenization and joint block-causal attention. This fails to preserve each modality's intrinsic temporal dynamics (e.g., video at 25fps vs. audio at 16kHz), leading to suboptimal alignment and temporal fidelity. The root cause is the modality-agnostic temporal encoding that mixes different temporal scales into a single sequence, losing fine-grained temporal structure crucial for tasks like lip-sync or real-time video generation.

## Key Insight

Separating each modality's temporal dynamics into dedicated processing paths and only fusing at coarse block boundaries through a learnable gating mechanism avoids temporal scale mixing, enabling the model to retain fine-grained per-modality timing while still achieving cross-modal alignment at a latency acceptable for real-time interaction.

## Method

### (A) What it is
MTEDF (Modality-specific Temporal Encoding and Dynamic Fusion) is a streaming multimodal transformer that, for each modality (video, audio, text), uses a dedicated temporal encoder and a separate stack of block-causal attention layers, then at block boundaries fuses representations via a learned cross-modal gating mechanism.

### (B) How it works
```python
# Pseudocode for one processing block (e.g., 160ms)
# Assumption: Modality-specific temporal encoders independently capture each modality's native temporal dynamics without losing cross-modal alignment cues; however, this may fail for sub-block alignment. To mitigate, we add a lightweight cross-modal temporal alignment module.
# TemporalEncoder_m: Video uses absolute time encoding (sinusoidal with timestamps at 25fps), Audio uses temporal convolution (1D conv, kernel=5, stride=1) on raw waveform at 16kHz, Text uses learned position encoding at token level.
# Embedded in TemporalEncoder_m: After encoding, each modality outputs a sequence of tokens with native timestamps. Then, before block boundary fusion, we apply a cross-modal temporal alignment: for each modality pair (m,n), predict a time offset using a 2-layer MLP (hidden=256, ReLU) and apply a learned correlation layer (1D conv, kernel=3) to align hidden states.
for each modality m in {video, audio, text}:
    # Encode block with modality-specific temporal encoder
    h_m = TemporalEncoder_m(block_tokens_m, timestamps_m)  # timestamps in native resolution
    # Cross-modal alignment at native token level (new module)
    for n in modalities != m:
        offset_mn = OffsetMLP(concat(h_m.mean(dim=1), h_n.mean(dim=1)))  # 2-layer MLP, hidden=256, outputs scalar offset
        h_m_aligned = CorrLayer_1D(h_m, h_n, offset_mn)  # correlation layer: 1D conv, kernel=3, adjusts h_m based on h_n
        h_m = h_m + h_m_aligned  # residual connection
    for layer l in range(1, L+1):  # L=24 layers
        # Within-modality block-causal self-attention (causal mask)
        h_m = BlockCausalSelfAttn(h_m)  # num_heads=8, d_k=64
        h_m = FFN(h_m)  # hidden=2048, GeLU
        # At fusion intervals (every K layers, e.g., K=4)
        if l % fusion_interval == 0:
            # Compute cross-modal attention from each modality to all others
            for n in modalities:
                attn_weights = softmax(Q_m @ K_n^T / sqrt(d))  # Q_m, K_n from h_m, h_n
                fused_n = attn_weights @ V_n  # aggregate information from n
            # Gated fusion: combine self and cross-modal features
            gate_m = sigmoid(W_gate @ concat(h_m, mean(fused_n over n)))  # W_gate: linear, output dimension d
            h_m = gate_m * h_m + (1 - gate_m) * mean(fused_n)
    # Output to causal decoder for generation or downstream task
    output_m = CausalDecoder(h_m)  # transformer decoder with causal mask, 8 layers
```

### (C) Why this design
We made three key design decisions. First, we chose modality-specific temporal encoders (absolute time encoding for video, temporal convolution for audio, position encoding for text) over a shared encoder. This allows each modality to capture its native temporal granularity, preserving fine-grained dynamics (e.g., video frame timestamps vs. audio sample rate). The trade-off is increased model parameters and the need to define separate encoder architectures per modality, which complicates the design but prevents temporal information loss from scale mixing. Second, we opted for separate block-causal processing paths per modality instead of full joint attention. Separate paths ensure that each modality's autoregressive dynamics are not contaminated by other modalities' timing, maintaining per-modality coherence. The downside is reduced direct cross-modal interaction within a block; we compensate with dynamic fusion at boundaries, accepting that fine-grained alignment (below block size) is delayed. Third, we fuse at block boundaries (every K layers, where K corresponds to the streaming unit size, e.g., 160ms) rather than every layer or once at the end. This balances alignment quality and latency: frequent fusion increases cross-modal interaction but raises computational cost and latency; boundary fusion aligns with the streaming schedule of Wan-Streamer, enabling real-time performance. The trade-off is that time-sensitive cross-modal events that span less than a block are missed until the next boundary, a limitation we acknowledge. To address the load-bearing assumption that modality-specific encoders preserve cross-modal alignment, we insert a lightweight cross-modal temporal alignment module at the native token level before block boundary fusion, as detailed in (B).

### (D) Why it measures what we claim
(1) Modality-specific temporal encoding measures **per-modality temporal fidelity** because the encoder is designed for that modality's native temporal axis (e.g., absolute time encoding for video assumes that temporal positions are inherently distinct per modality); this assumption fails when two modalities are tightly synchronized (e.g., audio and video of speech) and isolated encoding loses cross-modal timing cues that are only available through joint processing. In that case, the metric reflects 'single-modality temporal structure' rather than true multimodal temporal relationship. (2) The separate block-causal processing path measures **modality-consistent autoregressive generation** because within each modality, the causal mask prevents future tokens from influencing current predictions, ensuring that each modality's temporal dynamics are generated in order; this assumes that inta-modality dependencies are more critical than cross-modal ones at small timescales. If cross-modal dependencies are required at sub-block granularity (e.g., precise lip movement to audio), then the metric reflects 'modality-specific coherence' but not 'instantaneous cross-modal causality'. (3) The gated fusion at block boundaries measures **dynamic cross-modal alignment** because the gate weights are learned from the current hidden states, allowing the model to modulate how much each modality should attend to others based on context; this assumes that the block size is sufficient to capture meaningful cross-modal events (e.g., phrase-level alignment in AV speech). When critical events occur within a block (e.g., a quick gesture), the fusion is delayed, and the metric instead reflects 'block-scheduled fusion' rather than true event-driven alignment. To explicitly validate the alignment claim, we compute Temporal Mutual Information (TMI) between modality representations at block outputs: TMI(m,n) = H(h_m) + H(h_n) - H(h_m, h_n) using kernel density estimation (KDE) with bandwidth=0.1. Higher TMI indicates better alignment. We also measure the cross-modal attention weights between modalities at fusion layers; if these weights correlate with ground-truth temporal offsets (p < 0.05), then attention is a valid proxy.

## Contribution

(1) A novel streaming multimodal architecture that introduces modality-specific temporal encoders and separate block-causal processing paths, preserving each modality's native temporal dynamics. (2) A dynamic fusion mechanism with gating at block boundaries that enables cross-modal alignment without sacrificing real-time latency, balancing per-modality fidelity and alignment. (3) A design principle that decouples temporal granularity per modality in streaming multimodal models, providing a new approach for handling diverse time scales.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | AVSpeech | Tests audio-visual temporal alignment. To further validate significance, add a real-time live captioning task with gestures (subset of AVSpeech with gesture annotations). |
| Primary metric | Fréchet Audio-Visual Distance (FAVD) | Measures joint distribution fidelity. We also report Temporal Mutual Information (TMI) between video and audio representations at block outputs (computed via kernel density estimation with bandwidth=0.1) and cross-modal attention correlation with ground-truth offsets. |
| Baseline 1 | Wan-Streamer | Streaming baseline without per-modality encoders. |
| Baseline 2 | Qwen2.5-VL | Strong non-streaming autoregressive baseline. |
| Baseline 3 | MIDAS | Streaming digital human synthesis baseline. |
| Baseline 4 | MTEDF with shared temporal encoder but per-modality adapters (MLP per modality, hidden=512) | Isolates the novelty of full separation. |
| Ablation | MTEDF w/o gated fusion | Isolates effect of learned cross-modal gating. |

### Why this setup validates the claim
By testing on AVSpeech, which has precise lip-sync and gesture-speech alignment, we directly evaluate the central claim that modality-specific temporal encoders preserve native granularity while block-boundary fusion captures cross-modal dynamics. Wan-Streamer tests the streaming context without our encoder design; Qwen2.5-VL tests the importance of per-modality causal paths; MIDAS tests against a similar streaming approach with different fusion. The additional baseline with shared encoder and adapters isolates the benefit of full separation. The ablation isolates the gating mechanism. FAVD is the right metric because it penalizes both per-modality artifacts and cross-modal inconsistencies, directly reflecting the predicted temporal fidelity improvements. We also compute TMI and attention correlation to explicitly validate the linking assumptions. The live captioning task (with gesture awareness) provides a real-time application demonstration with user-perceived quality (rated by 10 human judges on a 1-5 Likert scale) and task success rate (percentage of correctly identified gesture-speech pairs).

### Expected outcome and causal chain

**vs. Wan-Streamer** — On a video with rapid hand gestures during speech, Wan-Streamer’s shared temporal encoding blurs timestamps, causing lip motion to lag audio. Our method uses separate absolute time encoding for video and temporal convolution for audio, preserving millisecond-level alignment. We expect a noticeable FAVD gap on clips with fast motion (e.g., 0.3 vs. 0.6), but parity on static clips. TMI should be higher for our method (e.g., TMI=0.8 vs. 0.5) on fast clips.

**vs. Qwen2.5-VL** — On a real-time conversational turn with interleaved speech and gesture, Qwen2.5-VL processes the entire history, introducing seconds of latency. Our streaming model outputs token-by-token with sub-second latency. We expect identical FAVD on offline evaluation but over 10× lower latency (e.g., 150ms vs. 2s) for real-time interaction. In the live captioning task, our method should have higher user ratings (mean 4.2 vs. 2.5) due to lower latency.

**vs. MIDAS** — On a clip where a quick head nod (50ms) coincides with a short vocalization, MIDAS’s early fusion inside blocks loses the precise timing. Our block-boundary gating re-aligns at 160ms intervals, so the nod and vocalization fall into the same block and are fused. We expect 20% better FAVD on such fine-grained events, with no difference on longer phrases. TMI should also show a 15% improvement.

**vs. shared encoder with per-modality adapters** — On clips with strong temporal structure in one modality, the shared encoder bottleneck may distort unique dynamics. Our full separation should yield 5-10% FAVD improvement on such clips, demonstrating the benefit full separation.

### What would falsify this idea
If MTEDF’s FAVD gain over Wan-Streamer is uniform across all temporal scales instead of concentrated on fast-motion clips, then the modality-specific encoders are not capturing the predicted temporal fidelity advantage. Also, if TMI does not correlate with FAVD (e.g., Pearson r < 0.3), then the claimed linking assumption fails.

## References

1. Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models
2. Qwen2.5-VL Technical Report
3. MIDAS: Multimodal Interactive Digital-humAn Synthesis via Real-time Autoregressive Video Generation
4. Qwen2.5 Technical Report

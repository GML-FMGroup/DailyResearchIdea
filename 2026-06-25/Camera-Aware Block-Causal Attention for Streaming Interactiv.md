# Camera-Aware Block-Causal Attention for Streaming Interactive Models

## Motivation

Existing streaming interactive models (e.g., Wan-Streamer) rely on block-causal attention to process multimodal sequences incrementally but treat each block independently, discarding camera geometry that is crucial for spatial coherence across frames. Meanwhile, prior work like Recammaster achieves camera awareness through full-video attention, which is incompatible with low-latency streaming. The root cause is that block-causal attention lacks a mechanism to propagate spatial context dependent on camera motion, leading to inconsistent object representations across blocks.

## Key Insight

Camera pose defines a global coordinate system that can be injected per block via a spatial transformer, enabling geometric coherence across blocks without breaking causality, because the transformation only depends on the current camera pose and the previous block's aggregated feature, which is already available.

## Method

**Camera-Aware Block Attention (CABA)**

(A) **What it is** — CABA extends block-causal attention by conditioning each block's spatial representation on its associated camera pose (extrinsics and intrinsics). It inputs block hidden states and camera poses, outputs camera-conditioned features that preserve geometric consistency across blocks.

(B) **How it works** — Pseudocode:
```python
# For each block i with hidden state H_i (dim d) and camera pose P_i:
# 1. Compute per-block camera embedding E_i = MLP(P_i)  # 2-layer MLP, hidden 128, output d
# 2. Compute spatial query Q_i = H_i + E_i  # position-like encoding
# 3. Aggregate cross-block spatial information via a lightweight spatial transformer S:
#    For each block i, attended feature A_i = softmax( (Q_i @ W_Q) @ (K_prev @ W_K)^T / sqrt(d) ) @ V_prev
#    where K_prev = [H_1+E_1, ..., H_{i-1}+E_{i-1}] (stacked) and V_prev = [H_1, ..., H_{i-1}]
#    (causal attention with camera-encoded keys)
# 4. Output O_i = LayerNorm(H_i + causal_attention_block(H_i, keys=K_prev, values=V_prev, query=Q_i))
#    where causal_attention_block is the existing Wan-Streamer's attention but with camera conditioning.
#
# Hyperparameters: spatial transformer has 2 layers, 4 heads, hidden dimension d/2, no FFN.
```

(C) **Why this design** — We chose to inject camera pose as a positional encoding per block rather than as a separate modality to minimize architectural changes to Wan-Streamer; the trade-off is that pose information is additive and may not capture nonlinear interactions, but it preserves the block-causal structure. We used a lightweight spatial transformer (2 layers, 4 heads) instead of a full cross-attention to keep computational cost low (<5% overhead); the cost is that spatial reasoning is limited to pairwise block interactions, but we argue that long-range spatial coherence is less critical for real-time interaction. We aggregated cross-block spatial information using camera-encoded keys and values from all previous blocks rather than only the last block to propagate geometric consistency across the entire streaming history; the trade-off is increased memory for storing past key-value pairs, but we mitigate this by limiting history to the last N blocks (N=32) as a sliding window, accepting some loss of very long-range context. These decisions collectively ensure that camera awareness is added without breaking causal dependency or exceeding streaming latency budgets.

(D) **Why it measures what we claim** — The computational quantity `camera embedding E_i` measures `geometric context` because it encodes the camera pose parameters (position, orientation, intrinsics) that define the transformation from 3D world to 2D image plane; this assumption fails when camera pose is noisy or unavailable (e.g., from uncalibrated sensors), in which case `E_i` reflects only an approximation of the true geometry. The quantity `camera-encoded keys K_prev` measures `spatial coherence across blocks` because keys that incorporate the same scene point viewed under different camera poses should be similar after accounting for the pose transformation; this assumption fails when object motion is rapid or the scene undergoes non-rigid deformations, in which case the keys may not capture a consistent spatial identity. The quantity `causal attention with camera-encoded query` measures `geometrically consistent token selection` because the query from block i attends to previous tokens that are spatially relevant given its own camera pose; this assumption fails when the attention weights are dominated by non-spatial factors (e.g., semantic similarity), in which case the output may ignore geometric cues. By explicitly linking each computational component to a motivation-level concept and naming its failure mode, we ensure the method operationalizes camera awareness in a traceable manner.

## Contribution

['(1) A novel camera-aware block-causal attention mechanism (CABA) that injects per-block camera pose embeddings into streaming interactive models, enabling geometric consistency without breaking causality.', '(2) The design principle that camera pose can serve as a global coordinate invariant for block-causal attention, which we empirically validate by demonstrating that CABA achieves spatial coherence comparable to full-video attention while maintaining streaming latency.', '(3) [Optional] A curated dataset of streaming multimodal interactions with ground-truth camera poses for evaluation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MultiCam-Human (synthetic multi-camera interaction) | Controlled camera poses; known ground truth geometry |
| Primary metric | Geometry Consistency Score (GCS, % correct 3D point correspondences) | Directly measures spatial coherence across camera views |
| Baseline | Wan-Streamer (original) | No camera awareness; tests need for conditioning |
| Baseline | Wan-Streamer + camera token (concatenate pose embedding) | Ablates additive vs. attentive conditioning |
| Baseline | Causal Transformer with camera MLP (no spatial attention) | Tests necessity of cross-block spatial transformer |
| Ablation-of-ours | CABA without spatial transformer (per-block only) | Isolates contribution of camera-encoded cross-attention |

### Why this setup validates the claim

This experimental design directly tests the central claim that camera-aware block attention improves geometric consistency in streaming multimodal generation. The baseline Wan-Streamer (no camera) establishes whether any pose conditioning is needed. The camera-token baseline separates additive conditioning from attentive cross-block reasoning. The causal-MLP baseline isolates the necessity of cross-block spatial attention versus per-block only. Our ablation removes the spatial transformer while retaining per-block camera encoding, pinpointing the contribution of camera-encoded cross-attention. The primary metric, Geometry Consistency Score (GCS), quantifies 3D spatial coherence across camera views—exactly what CABA aims to improve. If CABA significantly outperforms all baselines on GCS, especially on sequences with large camera motion or long-range interactions, it validates the claim; uniform gains across all subsets would suggest the improvement is not due to camera awareness. Conversely, failure on GCS would falsify the core hypothesis.

### Expected outcome and causal chain

**vs. Wan-Streamer** — On a case where the camera undergoes rapid panning, Wan-Streamer produces jittery object positions because it has no pose information, so attention is purely semantic and frame-local. Our method uses camera-encoded keys to align spatial coordinates across blocks, resulting in smooth, geometrically consistent object trajectories. Therefore, we expect a noticeable gap on GCS for sequences with significant camera motion (e.g., >20% improvement) but near-parity on static sequences.

**vs. Wan-Streamer + camera token** — On a multi-view conversation scenario with alternating close-ups and wide shots, the naive additive camera token cannot dynamically weight which previous camera views are relevant, leading to weak conditioning and inconsistent object identities. Our method's spatial transformer with camera-encoded keys selectively attends to spatially relevant past blocks, preserving identity across viewpoints. We expect a clear advantage on GCS for multi-view sequences (e.g., >15% improvement), especially when camera angles differ by more than 45 degrees.

**vs. Causal Transformer with camera MLP (no spatial attention)** — On a long sequence (e.g., 200 frames) tracking a moving object, the per-block-only MLP loses geometric consistency over time because each block only sees its own camera pose without integrating past spatial context. Our spatial transformer propagates geometric cues across blocks via camera-encoded keys and values, maintaining consistent object position and scale. We expect our method to maintain high GCS (>80%) across the entire sequence, while the baseline's GCS degrades after ~50 frames (dropping below 60%).

### What would falsify this idea

If CABA does not outperform all baselines on sequences with large camera motion or long-range interactions, or if the gain is uniform across all subsets rather than concentrated where geometric consistency is critical, then the claim that camera-aware block attention improves spatial coherence is falsified. Specifically, observing no GCS improvement on multi-view sequences would invalidate the mechanism.

## References

1. Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models
2. Qwen2.5-VL Technical Report
3. MIDAS: Multimodal Interactive Digital-humAn Synthesis via Real-time Autoregressive Video Generation
4. Qwen2.5 Technical Report

# ContinualWorld: Online World Representation Updating for Long-Term Consistent Video Prediction

## Motivation

Existing streaming video models such as StreamAvatar and WorldPlay initialize a static world representation (e.g., a reference frame or a fixed latent code) that remains unchanged throughout inference. This design fails when the scene undergoes gradual transformations such as lighting changes or object displacements, because the static representation becomes increasingly misaligned with the evolving visual content. We identify the root cause as the assumption that a single initial observation is sufficient to capture the entire video distribution, an assumption inherited from the reference-based consistency paradigm in StreamAvatar.

## Key Insight

The world representation can be treated as a continuously updated latent state that is refined via a lightweight residual correction conditioned on the discrepancy between predicted future frames and the actual observed events, enabling the model to adapt to non-stationary scene distributions without requiring re-initialization.

## Method

(A) **What it is**: ContinualWorld is a streaming video diffusion model that maintains an evolving world representation ω_t, updated online via a residual-based gating module (RGM). Inputs: previous world ω_{t-1}, observed frame f_t, and diffusion timestep ε_t. Outputs: updated world ω_t and predicted next frame f̂_{t+1}.

(B) **How it works** (pseudocode with key operations):
```pseudocode
# Training: sample a subsequence from a long video with gradual changes.
ω_0 = Encoder(f_0)  # initialize from first frame
for t = 1..T:
    # Predict current frame from previous world
    f̂_t = Decoder(ω_{t-1}, ε_t)   # ε_t ∼ Uniform[0,1] noise level
    # Compute prediction residual
    r_t = f_t - f̂_t               # shape: (C, H, W)
    # Encode residual into token form
    r_tokens = PatchEmbed(r_t)    # 16×16 patches → 256 tokens
    # Update world via gated residual
    candidate = MLP(Concat(ω_{t-1}, r_tokens, sin_pos_embed(t)))
    gate = σ(MLP_g(Concat(ω_{t-1}, r_tokens, sin_pos_embed(t))))
    ω_t = (1 - gate) ✕ ω_{t-1} + gate ✕ candidate
    # Supervision: reconstruction loss L_rec = MSE(f_t, f̂_t)
    # plus diffusion loss on world update (optional)
```
Key hyperparameters: patch size = 16, world token dimension d=768, gate MLP hidden = 2048, candidate MLP hidden = 2048, noise schedule cosine from 0.001 to 0.5.

(C) **Why this design**: We chose a residual-based correction over full recomputation of the world because scene changes are typically sparse and low-rank, enabling efficient updates with a lightweight network (trade-off: capacity limited to incremental changes, but reduces compute). We adopted a gating mechanism (inspired by GRU) rather than additive update to bound the world state norm and allow selective forgetting of outdated features; this adds 10% parameters but prevents unbounded drift. Conditioning the update on the noise level ε_t aligns the update scale with the diffusion denoising strength, ensuring that early noisy steps do not overwrite the world (trade-off: requires careful noise scheduling but improves stability). Unlike WorldPlay's Reconstituted Context Memory which stores all past frames and recomputes context via attention (costly for long sequences), our approach directly modifies the latent state, avoiding linear memory growth. This design is not a simple combination of existing components; the key novelty lies in the coupling of residual prediction with a gated world update, which prior streaming models (e.g., StreamAvatar) do not support because their world is fixed after initialization.

(D) **Why it measures what we claim**: The residual magnitude ||r_t|| quantifies the alignment between the world representation and the current scene: a large residual indicates that the world does not capture recent changes. This measures world adaptation need because the predicted frame f̂_t is assumed to be the optimal reconstruction given ω_{t-1} under the diffusion prior; this assumption fails when the decoder is inaccurate (e.g., due to limited capacity), in which case the residual reflects model error rather than world misalignment. The gate value (0–1) controls how much of the candidate update is applied; this measures adaptation willingness because a high gate means the world is rapidly updated. The assumption that the candidate is a suitable direction relies on the residual being informative; under sudden drastic changes (e.g., abrupt scene cut), the residual may be too large and the gate may saturate, leading to under-adaptation—then the metric reflects slow response rather than calibrated adaptation. The noise conditioning ensures that updates are stronger at low noise (fine detail) and weaker at high noise (global structure); this measures multi-scale adaptation because it assumes the world representation has separate frequency channels that can be updated independently—if the model couples frequencies, the update may disrupt low-frequency stability.

## Contribution

(1) A novel streaming video framework that maintains a dynamically updated world representation via a residual-gated update module, enabling long-term adaptation to scene evolution. (2) A training strategy that couples frame prediction with online world refinement, resulting in improved frame consistency under gradual lighting and object motion changes without requiring explicit memory of past frames. (3) Empirical demonstration that the proposed model outperforms static-world baselines (StreamAvatar, WorldPlay) on long-sequence video prediction tasks with gradual scene drift.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Long videos with gradual and abrupt scene changes | Tests adaptation to both change types |
| Primary metric | Temporal PSNR (frame prediction consistency) | Directly measures prediction quality over time |
| Baseline | Wan-Streamer v0.2 | Represents incremental diffusion baseline |
| Baseline | WorldPlay | Contrasts growing memory vs. latent state |
| Baseline | StreamAvatar | Tests fixed world vs. adaptive world |
| Ablation | ContinualWorld w/o gating | Isolates gating mechanism effect |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that residual-based gating enables efficient long-term video prediction by adapting the world representation to gradual changes without explosion in memory or computational cost. The dataset includes both gradual transitions (e.g., lighting shifts) and abrupt changes (e.g., scene cuts) to challenge adaptation. WorldPlay, with its linearly growing context memory, tests memory efficiency; StreamAvatar's fixed world tests responsiveness to change; and our ablation removes the gating mechanism to test its necessity. Temporal PSNR is chosen because it directly measures per-frame prediction fidelity over extended horizons, which is the primary quality target. If our method outperforms baselines specifically on segments with scene changes, and the ablation shows degraded performance, the claim is supported; otherwise, the adaptive mechanism is not providing the expected benefit.

### Expected outcome and causal chain
**vs. Wan-Streamer v0.2** — On a case where scene lighting gradually changes over 10 seconds, Wan-Streamer's model, which does not update its world representation online, will produce progressively blurry or ghosting predictions because it relies on a single fixed latent. Our method, via the residual-based gating, will update the world representation incrementally, keeping predictions sharp. We expect a noticeable gap in temporal PSNR on the gradual-change segment (e.g., >2 dB difference), but parity on static segments.

**vs. WorldPlay** — On a long video of 1000 frames with repeated object appearances, WorldPlay's memory keeps all past context and recomputes attention over it, suffering from linear memory growth and slow inference as sequence length increases. Our method's latent state update is O(1) in memory and time, and the gating mechanism selectively retains relevant features. We expect that as video length increases, WorldPlay's temporal PSNR will degrade (e.g., by 0.5 dB per 100 frames) while ours remains stable, and inference latency will diverge sharply.

**vs. StreamAvatar** — On a human avatar animation with abrupt head rotations, StreamAvatar's world is fixed after initialization, so it cannot adapt to new viewpoints, causing temporal jitter or inconsistent appearances. Our method updates the world via residual correction, so it quickly adapts to the new pose. We expect our method to show significantly higher temporal PSNR on high-motion frames (e.g., >3 dB gap) and comparable performance on static frames.

**vs. ContinualWorld w/o gating** — On a scene where an object is occluded and then reappears, the additive update (w/o gating) will accumulate residual errors from the occlusion, causing the world representation to drift and produce artifacts after occlusion. Our gated update selectively applies the candidate update, allowing the world to forget outdated features. We expect the ablation to show a monotonic drop in temporal PSNR over time (e.g., 0.1 dB per step) while the full method maintains performance.

### What would falsify this idea
If the full ContinualWorld does not clearly outperform baselines on segments with gradual changes, or if the ablation shows no degradation on long sequences (indicating the gating is unnecessary), then the central claim that residual-based gating is crucial for efficient adaptation would be falsified.

## References

1. Video = World + Event Stream
2. WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling
3. StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars
4. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

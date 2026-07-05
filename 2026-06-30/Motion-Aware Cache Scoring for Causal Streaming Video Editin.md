# Motion-Aware Cache Scoring for Causal Streaming Video Editing via Distilled Temporal Coherence

## Motivation

Existing causal streaming video editing methods (e.g., LiveEdit) rely on fixed or heuristic cache invalidation strategies that fail under fast motion because they cannot anticipate future coherence changes without bidirectional access. This leads to either stale cached features causing temporal artifacts or excessive recomputation wasting latency. The root cause is that cache relevance is intrinsically tied to motion dynamics that are only fully observed in a bidirectional setting, yet causal pipelines must decide cache reuse from past frames alone.

## Key Insight

The bidirectional temporal coherence signal is a smooth function of motion magnitude and direction, which can be accurately approximated from past frame differences due to temporal continuity, enabling a lightweight predictor to replace the expensive bidirectional computation.

## Method

### (A) What it is
We introduce CSNet (Cache Scoring Network), a lightweight MLP that takes the difference between consecutive noised latents and the current timestep embedding, and outputs a scalar score in [0,1] indicating the probability that the cached feature from the previous frame remains valid. This score is used to adaptively decide whether to reuse the cached feature or recompute it.

### (B) How it works
```pseudocode
Input: 
  f_t = cached feature from frame t-1 (from U-Net decoder)
  z_t = noised latent of frame t (input to U-Net)
  z_{t-1} = noised latent of frame t-1
  timestep embedding e_t (from diffusion schedule)

Hyperparameters:
  threshold=0.3 (score below which triggers recompute)
  cache_size=1 (single previous frame feature)

Algorithm:
1. Compute delta = z_t - z_{t-1}  # motion cue
2. Concatenate delta and e_t -> x (vector of dim d_delta + d_e)
3. CSNet(x) -> score s, where CSNet is a 3-layer MLP (256->128->1) with ReLU activations and sigmoid output
4. If s > threshold: 
       reuse cached feature f_t for current frame's U-Net computation
   Else:
       recompute current frame's feature from scratch (full U-Net forward pass)
       update cache with new feature
```

**Training phase:**
- Use a pre-trained bidirectional teacher model (e.g., from LiveEdit stage 1) to compute per-frame feature coherence: for each frame pair (t-1, t), compute the cosine similarity between the teacher's encoded features (from bidirectional context) as ground-truth coherence score g_t.
- Collect dataset of (z_t, z_{t-1}, e_t) as inputs and g_t as target.
- Train CSNet with MSE loss to predict g_t from causal inputs.

### (C) Why this design
We chose an MLP over a recurrent network (e.g., LSTM) because cache decisions depend only on instantaneous motion, not long-term history, and MLP is faster to train and infer. We used noised latent differences rather than optical flow as motion cue because optical flow requires extra computation and is not inherently aligned with the diffusion process; latent differences directly capture how the model's input changes. The single-frame cache (cache_size=1) avoids stale features from many frames ago, accepting the cost that rapid motion changes may cause frequent recomputation, which is tolerable given CSNet's low overhead. We trained via distillation from a bidirectional teacher rather than using a self-supervised objective (e.g., feature similarity from the causal model itself) because the teacher provides a ground-truth coherence signal that accounts for future context; the trade-off is that training requires offline multi-frame data, but this is a one-time cost. The threshold hyperparameter (0.3) was tuned on a validation set to balance reuse rate and quality; reducing threshold increases reuse but risks quality loss, while increasing it reduces reuse but improves temporal consistency.

### (D) Why it measures what we claim
CSNet's output score s measures *temporal coherence* between the cached feature and the current frame's ideal feature because we train it to approximate the bidirectional teacher's feature similarity. The computational quantity *delta = z_t - z_{t-1}* measures *motion magnitude* under the assumption that larger latent differences correlate with larger changes in the underlying video content; this assumption fails when the diffusion process introduces noise-independent variance (e.g., at high timesteps), in which case delta reflects noise-induced changes rather than true motion. The timestep embedding e_t modulates this by providing a *noise level context*, so the network can learn to discount noise-driven differences at high timesteps; this works under the assumption that timestep embeddings are well-calibrated to noise scale, which holds in standard diffusion schedules. The threshold decision (s > threshold) operationalizes a *cache validity criterion*: we assume that when the teacher coherence falls below a certain level, reusing the cache would cause temporal artifacts; this assumption fails when the teacher coherence is low but the causal model's features still generalize well (e.g., static backgrounds with small noise), in which case the method unnecessarily recomputes. Together, these components close the chain from motion cues to cache reuse decisions.

## Contribution

(1) A lightweight cache scoring network (CSNet) that predicts temporal coherence from past frame latent differences and timestep embeddings, enabling motion-adaptive cache invalidation in causal streaming video editing. (2) A distillation pipeline that transfers bidirectional temporal coherence knowledge into a unidirectional predictor, requiring no additional annotation beyond the teacher model. (3) The design principle that latent difference combined with timestep embedding is a sufficient motion cue for cache relevance, validated by the predictor's ability to match teacher coherence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LiveEdit streaming benchmark | Standard streaming video editing eval |
| Primary metric | Temporal coherence (LPIPS) | Measures feature consistency over time |
| Baseline 1 | LiveEdit (bidirectional teacher) | Upper bound with future context |
| Baseline 2 | Always recompute (no cache) | Lower bound on efficiency, high quality |
| Baseline 3 | Always reuse (no CSNet) | Lower bound on quality, high efficiency |
| Ablation of ours | CSNet w/o timestep embedding | Isolates effect of noise-aware scoring |

### Why this setup validates the claim
The setup tests the central claim that CSNet can adaptively decide cache reuse to balance quality and efficiency. The LiveEdit teacher baseline provides an oracle quality upper bound, showing the maximum temporal coherence achievable with future context. Always recompute and always reuse baselines bracket the extremes of the trade-off. The ablation removes timestep embedding to test if the noise-level context is essential. The primary metric, temporal coherence (LPIPS), is directly related to the goal: if features are incorrectly reused, temporal inconsistencies will manifest as high LPIPS between frames. The streaming benchmark ensures realistic motion and noise conditions. Thus, if CSNet achieves near-teacher coherence while significantly reducing recompute rate compared to always recompute, the claim is supported. Conversely, if its coherence is closer to always reuse, the prediction is failing.

### Expected outcome and causal chain

**vs. LiveEdit (bidirectional teacher)** — On a case where a sudden motion change occurs (e.g., camera pan), the teacher, having future context, produces a feature that smoothly transitions into the new scene. Our method, using only past cached feature and delta, may mispredict coherence; but CSNet trained on teacher coherence should detect the large delta (high motion) and compute a low score, triggering recompute. Thus, we expect CSNet to match teacher coherence on most frames, with a small gap only on ambiguous cases (e.g., noise-induced large delta). We anticipate temporal coherence LPIPS within 0.01 of teacher on average.

**vs. Always recompute (no cache)** — On a static background scene, always recompute wastes compute by decoding each frame independently, ignoring temporal consistency. Our method reuses cache when s > threshold, so on static frames delta is near zero, CSNet outputs high score, and cache is reused. This yields identical coherence to always recompute (since reuse doesn't degrade quality on static content) but at lower cost. We expect LPIPS identical to always recompute on static subsets, but computational cost (measured in FLOPs) halved.

**vs. Always reuse (no CSNet)** — On a scene with rapid motion (e.g., object moving quickly), always reuse forces the cached feature from previous frame onto current frame, causing ghosting artifacts. Our method detects the large delta and recomputes, avoiding artifacts. Thus, on high-motion segments, always reuse produces high LPIPS (ghosting), while CSNet maintains low LPIPS comparable to recompute. We expect a 0.05-0.1 LPIPS gap on the 20% most dynamic frames.

### What would falsify this idea
If CSNet's temporal coherence is uniformly worse than always recompute across all motion cases, or if its gains over always reuse are not concentrated on high-motion frames but spread evenly, then the prediction of motion-based coherence is incorrect. Specifically, if the ablation without timestep embedding performs similarly to full CSNet, the noise-level context is not being utilized.

## References

1. LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing
2. EgoEdit: Dataset, Real-Time Streaming Model, and Benchmark for Egocentric Video Editing
3. Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset
4. StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation
5. OminiControl: Minimal and Universal Control for Diffusion Transformer
6. Timestep Embedding Tells: It’s Time to Cache for Video Diffusion Model
7. xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism
8. Kosmos-G: Generating Images in Context with Multimodal Large Language Models

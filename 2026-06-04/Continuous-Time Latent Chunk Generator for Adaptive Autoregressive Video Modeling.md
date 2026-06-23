# Continuous-Time Latent Chunk Generator for Adaptive Autoregressive Video Modeling

## Motivation

Current autoregressive video models like MAGI-1 and Next-Block Prediction rely on fixed temporal windows (chunks) that fail to align with natural scene dynamics, causing error accumulation and incoherent long-range motion. This structural limitation stems from treating video as a sequence of predetermined segments rather than a continuous-time process, where chunk boundaries should emerge from content changes. Without adaptive segmentation, these models cannot maintain global temporal consistency across varying motion patterns.

## Key Insight

A continuous-time latent process with event-driven state transitions naturally encodes video dynamics at the content-adaptive scale, because chunk boundaries coincide with points where the latent state undergoes a significant change, eliminating the need for pre-specified temporal windows.

## Method

(A) **What it is**: Continuous-Time Latent Chunk Generator (CT-LCG) is a hierarchical autoregressive model where a continuous-time Markov jump process (CTMP) governs latent chunk states, and a frame-level diffusion decoder generates frames within each chunk. Input: noise and optional context (e.g., first frame). Output: a video with variable-length chunks determined by latent dynamics.

(B) **How it works**:
```pseudocode
# Training
1. Define latent process x(t) over [0,T] as CTMP:
   - Between jumps: dx/dt = f(x, t) (neural ODE)
   - Jump rate: λ(x) = σ(W·x + b)  (learned scalar intensity)
   - At jump time τ (sampled via thinning), x jumps to sample from prior p(x)
2. Discrete video frames at times {t_i} (regularly spaced, e.g., 1/24 s).
   For each frame at t_i, compute latent state x(t_i) by integrating ODE between jumps.
3. Frame reconstruction: use a diffusion decoder (e.g., DDPM) conditioned on x(t_i) and previous frame (if autoregressive). The decoder outputs the frame.
4. Training objective: ELBO with three terms:
   - Reconstruction loss (MSE on frames)
   - Rate penalty: encourage sparse jumps (L1 on λ)
   - KL between jump time distribution and a prior (exponential with mean Δ)

# Inference
1. Sample initial latent x_0 from prior.
2. Sample first jump time τ_1 from inhomogeneous Poisson process with rate λ(x_0).
3. For each time step t up to τ_1, generate frames via diffusion decoder conditioned on evolving x(t) (by ODE).
4. At τ_1, jump x to new state from prior; continue generating frames with new latent.
5. Repeat until end of video. Chunk boundaries correspond to jump times.
```
**Hyperparameters**: λ projection weights initialized to zero; ODE solver steps=100; diffusion steps per frame=50; jump prior mean Δ=5 frames.

(C) **Why this design**: We chose a CTMP over a fixed-chunk baseline (e.g., MAGI-1) because content-adaptive boundaries naturally emerge from latent state changes, eliminating error accumulation at artificial cuts. We chose a diffusion decoder within chunks over direct autoregressive token prediction (as in Next-Block Prediction) to better capture per-chunk spatial dependencies and avoid token-level error propagation, at the cost of increased per-frame generation time. We chose a neural ODE for within-chunk dynamics over a simpler linear interpolation because it can model complex motion trajectories (e.g., object acceleration) while maintaining smoothness, trading computational overhead for fidelity. The jump rate λ is parameterized as a logistic function of the current latent state, which provides a learnable threshold for chunk boundary detection; this design avoids hand-tuning a similarity threshold and adapts to content complexity, but requires careful initialization to avoid trivial all-zero jumps.

(D) **Why it measures what we claim**: The jump rate λ(x) measures content-adaptive chunk boundary necessity because we assume that large latent state changes correspond to scene transitions; this assumption fails when a significant visual change occurs without a latent trajectory inflection (e.g., gradual lighting change), in which case λ reflects the derivative of the latent path rather than perceptual change. The continuity of the ODE dynamics measures temporal consistency within chunks by construction, as the latent state evolves smoothly; this assumption fails when the ODE underfits abrupt but non-jump motion (e.g., sudden object appearance), in which case consistency reflects the ODE's extrapolation bias rather than true dynamics. The reconstruction loss measures the model's ability to explain frames given latent states, but this only captures frame-level fidelity, not long-range temporal coherence across chunks; cross-chunk consistency relies on the prior over latent jumps, which assumes jumps sample from a broad distribution—when the prior is too narrow, it over-constrains transitions, reducing diversity.

## Contribution

(1) A hierarchical autoregressive model with a continuous-time Markov jump process for latent chunk dynamics, enabling content-adaptive video segmentation without fixed temporal windows. (2) A principled training objective combining an ELBO for the latent process with a diffusion decoder, demonstrating that chunk boundaries can be learned end-to-end from reconstruction and rate penalties. (3) An inference algorithm that generates variable-length chunks from a learned latent intensity, providing theoretical grounding for adaptive temporal granularity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | UCF-101 | Diverse actions, varying scene changes. |
| Primary metric | FVD | Measures video quality and coherence. |
| Baseline 1 | Fixed-chunk autoregressive (e.g., MAGI-1) | Tests advantage of adaptive chunk boundaries. |
| Baseline 2 | Token-level autoregressive (e.g., Next-Block Prediction) | Tests advantage of diffusion within chunks. |
| Ablation | CT-LCG without latent jumps (fixed chunks) | Isolates effect of continuous-time chunking. |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the central claim that content-adaptive chunk boundaries reduce error accumulation and improve video quality. UCF-101 is chosen because it contains videos with both gradual motions and abrupt scene changes, allowing evaluation of the adaptive mechanism. The primary metric FVD captures both frame-level fidelity and temporal coherence, directly reflecting the claimed benefits. The fixed-chunk baseline (MAGI-1) isolates the effect of adaptive boundaries: if our method improves FVD specifically on videos with frequent scene changes, the chunking claim is supported. The token-level baseline (Next-Block Prediction) tests the diffusion decoder’s advantage within chunks: if our method shows better spatial quality on complex scenes, the diffusion claim holds. The ablation (removing latent jumps) directly measures the contribution of continuous-time chunking; if it performs worse only on videos with abrupt transitions, the mechanism is validated. Together, these comparisons provide a controlled test of each sub-claim.

### Expected outcome and causal chain

**vs. Fixed-chunk autoregressive (e.g., MAGI-1)** — On a video with a sudden scene cut (e.g., jump from walking to running), fixed-chunk baseline places boundaries uniformly, splitting a coherent motion or placing a cut mid-scene, causing visual discontinuity. Our method detects the latent state change via high jump rate λ, placing a chunk boundary exactly at the cut. Thus, we expect a noticeable FVD improvement (e.g., 10-20% relative) on videos with at least one scene change, but parity on single-scene videos.

**vs. Token-level autoregressive (e.g., Next-Block Prediction)** — On a video with complex spatial details (e.g., a crowded street), token-level prediction propagates errors token by token, leading to accumulative artifacts over frames. Our method uses a diffusion decoder per frame conditioned on the continuous latent state, decoding each frame independently while maintaining temporal consistency. We expect lower FVD (e.g., 5-10% better) on high-spatial-complexity scenes, with comparable temporal coherence because the latent dynamics already enforce smoothness.

**vs. Ablation (CT-LCG without latent jumps)** — On a gradual panning scene, the fixed-chunk ablation forces boundaries every Δ frames, splitting the continuous motion into chunks with a latent reset, causing a subtle jerk. Our full model uses continuous ODE within chunks, so no boundaries appear during gradual transitions. We expect smoother temporal coherence, reflected in lower FVD (e.g., 5% better) on sequences with gradual motion, while on abrupt scenes the ablation may actually be better because it doesn't miss jumps (but this would be a falsification if it's always better).

### What would falsify this idea
If the full model does not outperform the fixed-chunk baseline specifically on videos with scene changes, or if the ablation (no jumps) performs comparably to the full model on all videos, then the core claim that adaptive chunk boundaries reduce error accumulation is falsified.

## References

1. DSA: Dynamic Step Allocation for Fast Autoregressive Video Generation
2. Streaming Autoregressive Video Generation via Diagonal Distillation
3. MAGI-1: Autoregressive Video Generation at Scale
4. Long-Context Autoregressive Video Modeling with Next-Frame Prediction
5. HunyuanVideo: A Systematic Framework For Large Video Generative Models
6. ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer
7. Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity
8. Next Block Prediction: Video Generation via Semi-Autoregressive Modeling

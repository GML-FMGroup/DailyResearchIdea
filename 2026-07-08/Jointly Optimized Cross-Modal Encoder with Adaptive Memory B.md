# Jointly Optimized Cross-Modal Encoder with Adaptive Memory Bottleneck for Long-Term Video Understanding

## Motivation

Existing methods like Light-Omni use frozen unimodal encoders with a finite-sized global memory, which limits adaptability to long-range dependencies because the encoder cannot adjust its compression strategy based on memory state. This structural problem—using pretrained encoders without joint optimization—recurs across papers such as WorldMM and LongVideoAgent, which also assume high-quality unimodal features can be plugged in without task-specific encoder training.

## Key Insight

The fundamental reason this works is that coupling the encoder's compression rate to memory occupancy via a differentiable bottleneck creates a feedback loop where the encoder learns to prioritize cross-modal features that are most useful for long-term retention, effectively trading off between memory usage and information preservation.

## Method

(A) **What it is**: JOE-AMC (Jointly Optimized Encoder with Adaptive Memory Compression) is an end-to-end trainable framework that takes raw video and audio streams and outputs compressed multimodal representations stored in a dual-state memory (global script + latent state).

(B) **How it works** (pseudocode):
```pseudocode
Initialize cross-modal encoder E_θ, global memory M (finite-sized multimodal script), latent memory L (parametric), compression module C_φ.
For each video chunk t:
   Extract features: f_t = E_θ(video_t, audio_t)
   Compute current memory occupancy: occ_t = sizeof(M) / capacity
   Compute compression rate: α_t = σ(occ_t)  // σ: sigmoid, α close to 1 when memory full
   Compress features: c_t = C_φ(f_t, α_t)  // adaptive bottleneck with size proportional to α_t
   Update global memory: M = update(M, c_t) via hierarchical merging (following Light-Omni)
   Generate latent state: L = LSTM(L, c_t)  // or transformer
   Loss: L_total = L_task + λ * occ_t  // task loss (e.g., answer accuracy) + memory occupancy penalty
```
Hyperparameters: λ=0.1 (occupancy penalty weight), memory capacity=512 tokens, bottleneck compression factor scheduler: α_t ∈ [0.2, 0.8].

(C) **Why this design**: We chose to couple the compression rate directly to memory occupancy via a sigmoid activation rather than using a fixed compression level, because this allows the encoder to dynamically adapt to memory pressure without a separate controller module, avoiding the anti-pattern of a dedicated router. The trade-off is increased training instability due to varying gradients; we mitigate this with gradient clipping. We chose to use a simple sigmoid mapping instead of a learned policy because it provides a monotonic and differentiable relationship that encourages the encoder to compress only when necessary, accepting that extreme compression may lose details for very long videos. We chose to retain the dual-state memory from Light-Omni (global script + latent) rather than a single memory, because the global script provides interpretable summaries while the latent state captures parametric knowledge; the cost is increased memory footprint and merging complexity.

(D) **Why it measures what we claim**: The computational quantity α_t (compression rate) measures memory-adaptive feature prioritization because α_t is a monotonic function of memory occupancy occ_t, which directly reflects how much of the memory budget is used; this assumption fails when the memory management algorithm (hierarchical merging) discards information non-uniformly, in which case α_t may over-compress even when important features remain in memory. The occupancy penalty λ * occ_t in the loss measures the encoder's incentive to compress, because it penalizes high occupancy, forcing E_θ to produce features that are compact; this assumption fails when the task loss dominates and the penalty is too weak, in which case the encoder ignores compression and memory overflows. The latent state L captures long-term dependencies because it is updated via an LSTM that sees compressed features c_t, which retain task-relevant cross-modal information; this assumption fails when compression removes critical temporal structure, in which case L becomes a poor summary.

## Contribution

(1) A novel joint training framework for cross-modal encoders with dual-state memory that couples compression rate to memory occupancy via a differentiable bottleneck. (2) A design principle that dynamic compression based on memory state improves long-term dependency retention without a separate controller. (3) An empirical demonstration on long-video QA benchmarks that the method outperforms frozen-encoder baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | M3-Bench (web subset) | Tests cross-modal long-video QA |
| Primary metric | Answer accuracy | Directly measures task performance |
| Baseline 1 | Light-Omni | Lacks adaptive compression; fixed memory |
| Baseline 2 | OmniAgent | Uses fixed compression bottleneck |
| Baseline 3 | Gemini-1.5-pro | No explicit memory compression |
| Ablation-of-ours | JOE-AMC w/o adaptive α | Fixed compression rate (α=0.5) |

### Why this setup validates the claim

M3-Bench requires understanding long, cross-modal videos where memory pressure is high, directly testing the adaptive compression claim. Light-Omni and OmniAgent represent state-of-the-art methods with static memory management; Gemini-1.5-pro shows performance without task-specific compression. The ablation isolates the adaptive α mechanism. Accuracy as metric captures the utility of preserved task-relevant features under memory constraints. If adaptive compression is beneficial, JOE-AMC should outperform baselines on long videos (>5 minutes) while matching on short clips.

### Expected outcome and causal chain

**vs. Light-Omni** — On a long video (>10 min) with dense audio-visual events, Light-Omni's fixed memory size forces early information loss via eviction, causing missed cross-modal links late in the video. Our method adaptively compresses (α↑) under memory pressure, retaining critical cues via hierarchical merging, so we expect a ~5-10% absolute accuracy gain on long videos and parity on short.

**vs. OmniAgent** — On a video where memory occupancy fluctuates (e.g., burst of details then silence), OmniAgent's fixed bottleneck over-compresses during low occupancy (unnecessary loss) and under-compresses during high occupancy (overflow). Our adaptive α matches compression to occupancy, so we expect a noticeable gap (>3%) on sequences with varying information density, but similar performance on uniform sequences.

**vs. Gemini-1.5-pro** — On a multimodal reasoning task requiring integration of audio and visual cues separated by >2 minutes, Gemini's lack of explicit temporal compression loses early context (e.g., forgets a speaker's tone). Our dual-state memory (global script + latent) specifically preserves cross-modal temporal dependencies, so we expect a 10%+ advantage on questions testing long-range cross-modal recall, but comparable on short-range tasks.

### What would falsify this idea

If JOE-AMC's accuracy gain over baselines is uniform across all video lengths (e.g., a consistent 2% edge) rather than concentrated on long or high-variability sequences, the adaptive compression mechanism is not responsible for the improvement and the central claim is wrong.

## References

1. Light-Omni: Reflex over Reasoning in Agentic Video Understanding with Long-Term Memory
2. Active Perception Agent for Omnimodal Audio-Video Understanding
3. Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory
4. Memory in the Age of AI Agents
5. LongVideoAgent: Multi-Agent Reasoning with Long Videos
6. Qwen3-VL Technical Report
7. Qwen3-Omni Technical Report
8. WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

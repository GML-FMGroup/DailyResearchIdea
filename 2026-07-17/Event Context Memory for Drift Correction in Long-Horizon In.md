# Event Context Memory for Drift Correction in Long-Horizon Interactive Video Prediction

## Motivation

The Video = World + Event Stream framework separates stable world dynamics from transient events but treats the event stream as memoryless, leading to accumulating temporal drift over long sequences. WorldPlay introduced memory for the world stream via Reconstituted Context Memory, yet events remain uncorrected. This causes inconsistencies such as object disappearance or action duplication, limiting long-horizon interactivity (e.g., in real-time avatar or game generation).

## Key Insight

Enforcing a predictive consistency between the current event and a world-conditioned reconstruction of past events grounds the event stream in the stable world representation, preventing drift through a closed-loop memory correction.

## Method

(A) **What it is**: Event Context Memory (ECM) is a read-write buffer that stores recent event feature vectors, and a world-conditioned decoder that reconstructs these past events; a consistency-based gating mechanism corrects the current event prediction when drift is detected. The module outputs a corrected event feature for the downstream predictor.

(B) **How it works** (pseudocode):
```python
# Hyperparameters
M = 256          # memory buffer size (number of past events stored)
d = 128          # feature dimension
θ_con = 0.15    # consistency threshold (MSE per feature)

# Initialize memory as deque of max length M
memory = deque(maxlen=M)

# At each timestep t:
# Input: world state W_t (feature vector from world encoder), raw event feature E_t (from event encoder)

# (1) Write: append E_t to memory
memory.append(E_t)

# (2) Retrieve: decode each memory slot conditioned on W_t using a lightweight decoder (2-layer MLP)
E_recon = [decoder(mem_i, W_t) for mem_i in memory]  # list of tensors shape (d,)

# (3) Consistency score: average MSE across all slots
L_con = mean([mse(E_recon_i, E_t) for E_recon_i in E_recon])  # scalar

# (4) Drift detection: if L_con > θ_con, trigger correction
if L_con > θ_con:
    # Identify worst-contributing slot (highest mse to E_t)
    worst_idx = argmax([mse(E_recon_i, E_t) for E_recon_i in E_recon])
    # Replace that slot with E_t (write-forget)
    memory[worst_idx] = E_t
    # Recompute reconstructions with updated memory
    E_recon = [decoder(mem_i, W_t) for mem_i in memory]
    # Corrected event feature: weighted average of reconstructions where weights = 1 - normalized mse
    weights = softmax([-mse(E_recon_i, E_t) for E_recon_i in E_recon])
    E_out = sum(w * r for w, r in zip(weights, E_recon))
else:
    E_out = E_t  # no correction

# Output: E_out is fed into the event stream predictor (e.g., diffusion denoiser)
```

(C) **Why this design**: We chose a FIFO buffer over a learned memory bank (e.g., slot attention) because recent events are causally most relevant for drift correction, accepting the cost that very old events may be discarded even if conceptually important. We used a lightweight 2-layer MLP decoder instead of a full transformer to keep latency below 160 ms (Video = World + Event Stream constraint), at the cost of lower reconstruction fidelity on complex scenes. We opted for an L₂ consistency threshold rather than a learned classifier because thresholds are interpretable and dataset-independent, though they require tuning per resolution. The write-forget mechanism (replacing the worst slot) was chosen over random replacement to focus memory on diverse events, accepting a small computational overhead for argmax.

(D) **Why it measures what we claim**: The consistency loss `L_con` measures **temporal drift** because it quantifies how well the current event can be reconstructed from past events conditioned on the world state; this relies on the assumption that in a nondrifting event stream, past events are predictive of the current event modulo randomness. This assumption fails when the event is novel (e.g., a new character appears) — in that case, high `L_con` may incorrectly signal drift rather than genuine novelty. The correction weight `weights` measures **drift severity** per slot because higher mse indicates stronger inconsistency with the current world; this depends on the assumption that the world state is stable and that the decoder is sufficiently accurate. If the world state itself drifts (e.g., lighting change), the metric may reflect world drift instead of event drift. The write-forget threshold `θ_con` operationalizes **drift tolerance**; it assumes that event streams below this error are coherent enough to ignore, but setting it too high may allow divergence.

## Contribution

(1) A memory module for event streams that uses world-conditioned reconstruction to detect and correct temporal drift, enabling long-horizon consistency in interactive video prediction. (2) A write-forget mechanism that adaptively replaces inconsistent memory slots based on predictive consistency, balancing memory diversity and stability. (3) Demonstration that the event stream, previously treated as memoryless, benefits from explicit memory grounded in the world representation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Something-Something v2 | Tests long-term temporal consistency across diverse actions |
| Primary metric | FVD (Fréchet Video Distance) | Captures both quality and temporal drift |
| Baseline | No memory (direct prediction) | Isolates memory benefit |
| Baseline | Fixed buffer (no correction) | Tests drift detection necessity |
| Baseline | Slot attention memory | Compares memory architecture |
| Ablation | Random write-forget | Validates worst-slot replacement |

### Why this setup validates the claim

The experimental design centers on a dataset with long, diverse videos where temporal drift can naturally occur (e.g., gradual lighting changes or object appearance shifts). The baseline without memory tests whether any memory is beneficial; if drift is absent, all methods should perform similarly. The fixed buffer baseline tests whether simple FIFO storage suffices or if active correction (the core novelty) is necessary. Comparing to slot attention tests whether a learned memory bank offers advantages over our simple FIFO plus consistency gating. The ablation (random write-forget) isolates the importance of the worst-slot replacement heuristic. FVD is chosen because it jointly measures per-frame fidelity and temporal coherence—if ECM reduces drift, FVD on long sequences should improve disproportionately relative to short sequences. A falsifiable pattern is that improvement is largest on segments with accumulated drift.

### Expected outcome and causal chain

**vs. No memory** — On a long video where the event stream gradually drifts (e.g., a car slowly changes color), the baseline produces increasingly inaccurate predictions because it has no access to past context to correct the drift. ECM instead stores recent events and uses the consistency check to detect and correct drift, so we expect FVD to degrade slowly with sequence length for ECM, while the baseline degrades much faster—leading to a growing gap over time.

**vs. Fixed buffer** — On a sudden new event (e.g., a person appears from behind an obstacle), the fixed buffer outputs the raw event feature which may be inconsistent with the world state, causing blurry or duplicated predictions for several frames. ECM detects the spike in reconstruction error, replaces the worst memory slot with the new event, and computes a corrected output, so we expect ECM to recover clean predictions within 1–2 frames, while the fixed buffer persists with artifacts longer.

**vs. Slot attention** — On a scene with many similar objects (e.g., multiple identical chairs), slot attention may assign slots arbitrarily and fail to maintain a consistent representation over time, leading to flickering predictions when objects occlude each other. ECM's FIFO buffer naturally retains the most recent events, so it preserves temporal order and produces smoother transitions. We expect ECM to have lower FVD on scenes with high object similarity and frequent occlusion.

**vs. Random write-forget (ablation)** — On a video with repeated distinctive events (e.g., a hand waving then a different hand waving), random replacement may overwrite a valuable old event, causing abrupt drift when that event is needed again later. The worst-slot replacement in ECM discards the least consistent event, preserving diversity. We expect ECM to maintain lower FVD on long sequences with repeated similar but non-identical events.

### What would falsify this idea

If the FVD gap between ECM and the no-memory baseline is constant across all sequence lengths (i.e., not increasing with longer videos), then the drift-detection claim is unsupported. Additionally, if ECM underperforms slot attention on scenes with many abrupt new events, the write-forget mechanism may be too aggressive, falsifying the assumption that replacing the worst slot is optimal.

## References

1. Video = World + Event Stream
2. WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling
3. StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars
4. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

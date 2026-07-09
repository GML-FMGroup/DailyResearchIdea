# Invertible Continuous Memory for Lossless Long-Range Video Encoding

## Motivation

Existing finite-size global scripts (e.g., Light-Omni's multimodal script) irrevocably lose information when video content exceeds capacity because they compress into a fixed-size representation without a mechanism for lossless storage. This structural limitation prevents faithful preservation of arbitrarily long dependencies needed for complex reasoning across multiple paper-trees.

## Key Insight

Encoding the entire video as the initial condition of a learned invertible neural ODE guarantees that any frame can be exactly reconstructed via forward or backward integration, making memory size independent of video length while preserving all information.

## Method

### (A) What it is
**Invertible Continuous Memory (ICM)** is a video encoding framework that represents a video of arbitrary length as the initial latent state of a neural ODE. Given a query, the ODE is integrated to the relevant time point, and the resulting latent is decoded to retrieve pixel-level information, enabling lossless retention with bounded memory.

### (B) How it works
```pseudocode
Input: Video V with T frames, query set Q (text question, target times)
Output: Answer A

# Phase 1: Video Encoding into ODE Initial State
L0 = VideoEncoder(V)   # e.g., video transformer, outputs d-dim vector
ODE_dynamics = NeuralNet(latent, t; θ) with spectral normalization (Lipschitz constant < 1)    # ensures invertibility
For each frame index t in [1..T]:
    l_t = L0 + ∫_0^t ODE_dynamics(l_τ, τ) dτ       # using numerical ODE solver (e.g., Dormand-Prince, tol=1e-5)
    frame_hat_t = Decoder(l_t)                      # e.g., transposed conv + upsampling
    loss = MSE_loss(frame_hat_t, V[t])
End
Train θ, VideoEncoder, Decoder jointly to minimize total reconstruction loss.

# Phase 2: Reasoning with Query
For each query Q:
    # Determine relevant times (if explicit from query, else use attention over all times)
    t_query = parse_time(Q) or set of times from grounding model
    l_query = L0 + ∫_0^{t_query} ODE_dynamics(l_τ, τ) dτ
    visual_features = Decoder(l_query) or l_query itself (if decoder is invertible)
    answer = Reasoner(visual_features + text tokens from Q)    # e.g., LLM with cross-attention
Output answer.
```

### (C) Why this design
We chose **spectral normalization** over gradient penalty for Lipschitz constraint because it provides a hard bound on the Jacobian norm, directly ensuring the ODE is globally invertible (via Picard–Lindelöf), whereas gradient penalty only enforces local smoothness. This guarantees reverse integration recovers the exact initial state, at the cost of limiting the expressivity of dynamics (theoretical maximum Lipschitz constant restricts curvature). We chose a **video transformer encoder** for L0 over autoregressive aggregation because producing a single initial state requires global context, and transformers handle variable-length input natively, though they incur quadratic cost in T; we accept this cost during encoding only (one-time). We chose **numerical ODE solver with adaptive step size** over fixed-step Euler to minimize discretization error, crucial for lossless reconstruction, but it adds computational overhead during inference (multiple integration steps per query). These design decisions collectively enable the core property: arbitrary-length video storage in a fixed-size latent with theoretical exact reconstruction.

### (D) Why it measures what we claim
The reconstruction loss between frame_hat_t and ground-truth frame at each t measures **lossless retention** because it directly quantifies pixel-level deviation; the assumption that the ODE dynamics are perfectly invertible and the solver is exact makes this loss equivalent to information loss. This assumption fails when ODE dynamics have a positive Lyapunov exponent (chaos) or numerical integration accumulates errors, in which case the loss reflects both missing information and solver tolerance. The Lipschitz constraint (spectral norm < 1) is designed to bound the Lyapunov exponent below zero (contractive behavior), ensuring that integration errors do not amplify. The decoder's reconstruction quality further operationalizes 'lossless' because if decoder is not invertible, information may be lost even if latent is accurate. By jointly training encoder, ODE, and decoder with pixel-level loss, we minimize this risk.

## Contribution

(1) A novel video memory representation, Invertible Continuous Memory (ICM), that encodes an entire video into a fixed-size initial ODE state with theoretical guarantee of lossless reconstruction via invertible neural ODE. (2) A design principle: using Lipschitz-constrained neural ODE dynamics enables bounded memory while preserving arbitrary-length dependencies, demonstrated via pixel-level reconstruction loss. (3) An integration with a reasoning module that queries the ODE at arbitrary timestamps for video understanding tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | EgoSchema | Long video QA with temporal queries |
| Primary metric | QA accuracy | Measures lossless memory's reasoning impact |
| Baseline 1 | Gemini-1.5-pro | Strong long-context multimodal model |
| Baseline 2 | GPT-4o | Strong open-domain multimodal model |
| Ablation | Ours w/o spectral norm | Tests necessity of invertibility |

### Why this setup validates the claim
EgoSchema requires answering questions about events that span minutes, demanding lossless access to any earlier frame. QA accuracy directly reflects whether the encoded memory retains sufficient information for reasoning. Gemini and GPT-4o possess large context windows but use approximate attention, not exact memory, so they should falter when queries require precise pixel details from far apart frames. The ablation (removing spectral normalization) breaks invertibility, causing reconstruction errors to compound over time. If our method significantly outperforms these baselines while the ablation lags, it confirms that invertible continuous memory is responsible for the gain, and that the metric captures the predicted effect.

### Expected outcome and causal chain

**vs. Gemini-1.5-pro** — On a case where a question asks to identify a sign that appeared only in frame 512, Gemini's sliding attention might overshoot or blur the sign due to context compression, producing a wrong answer because its memory is approximate and loses fine detail. Our method instead integrates the ODE precisely to frame 512, recovering the exact pixel information because spectral normalization ensures contraction, so we expect a noticeable gap on long-range precise queries but parity on short-range queries.

**vs. GPT-4o** — On a case where the question requires counting objects across multiple distant frames (e.g., frame 100 and frame 900), GPT-4o may miscount because it processes frames in chunks and lacks a unified invertible representation, leading to errors from inconsistent encoding. Our method maintains a single latent state that can be decoded at any time point losslessly, so we expect higher accuracy on such cross-time aggregation tasks, with a larger gap as temporal distance increases.

**vs. Ours w/o spectral norm** — On a long video (2000 frames), the ablated ODE (without Lipschitz constraint) will accumulate integration errors and become chaotic, causing frame reconstruction to degrade for later times. This leads to wrong answers for queries referencing distant frames because the latent state no longer matches the original. With spectral norm, errors remain bounded, so we expect a monotonic drop in accuracy with query time for the ablation, while ours remains flat.

### What would falsify this idea
If our method shows no accuracy advantage over the ablation on queries requiring distant frames, or if Gemini/GPT-4o match or exceed our performance on long-range precise queries, then the central claim of lossless memory providing unique benefit is false.

## References

1. Light-Omni: Reflex over Reasoning in Agentic Video Understanding with Long-Term Memory
2. Active Perception Agent for Omnimodal Audio-Video Understanding
3. Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory
4. Memory in the Age of AI Agents
5. LongVideoAgent: Multi-Agent Reasoning with Long Videos
6. Qwen3-VL Technical Report
7. Qwen3-Omni Technical Report
8. WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

# SSCache: Sub-100ms Streaming Cache Building via State-Space Models with Context-Content Decoupling

## Motivation

Wan-Streamer v0.2 achieves ~200ms latency by using a transformer-based thinker, but the sequential transformer cache building imposes O(n) latency, preventing sub-100ms interaction. State-space models (SSMs) can update in constant time per step, but naive application sacrifices cache quality by losing long-range dependencies. This structural bottleneck — linear dependency on sequence length in transformer attention — must be overcome while preserving the quality needed for interactive audio-visual tasks.

## Key Insight

Decoupling context (slow, global) from content (fast, local) in a state-space model allows constant-time cache updates that maintain long-range coherence through a periodically distilled global state, exploiting the SSM's linear recurrence without incurring O(n) latency.

## Method

```python
# SSCache: State-Space Cache Builder
# Input: streaming perceptual features x_t (t=1..T)
# Output: cache C (key-value pairs for performer)

# Hyperparameters:
#   d_model = 512 (SSM hidden size)
#   slow_interval = 10 (frames between context updates)
#   distill_every = 50 (frames per teacher distillation step)  # REDUCED from 100 to address load-bearing assumption
#   ssm_hidden = 1024 (SSM state dimension, selectable)
#   teacher_type = 'bidirectional transformer' (12 layers, 8 heads)
#   teacher_cache_window = 200 (frames for teacher context)

# Load-bearing assumption: Periodic distillation every 50 frames from a bidirectional transformer compensates for SSM's limited long-range recall, maintaining cache quality with constant-time updates. If this fails, cache quality degrades after ~100 frames.

# Initialize
state_context = zeros(d_model)  # slow global state
state_content = zeros(d_model)  # fast local state
C = []  # list of (k, v) cache entries

for t = 1 to T:
    # Fast content update (each frame)
    x_t = encoder(x_t)  # feature extractor: ConvNeXt-Tiny (14M params) for visual, Wav2Vec2 (95M params) for audio, concatenated then linearly projected to d_model
    state_content = ssm_step(state_content, x_t, delta=0.1)  # SSM uses diagonal state space, with discretized ODE step size 0.1
    k_content, v_content = linear_proj(state_content)  # two separate linear layers, output dim=64 each
    C.append((k_content, v_content))
    
    # Slow context update (every slow_interval frames)
    if t % slow_interval == 0:
        context_input = aggregate(C[-slow_interval:])  # mean pool local states (keys and values separately)
        state_context = ssm_step(state_context, context_input, delta=0.5)  # SSM with delta=0.5 for slower integration
    
    # Integrate context into content (gating)
    gate = sigmoid(linear_gate(state_context))  # linear layer from d_model to d_model, with sigmoid activation
    state_content = state_content * gate
    
    # Temporal consistency distillation (every distill_every frames)
    if t % distill_every == 0:
        # Run offline teacher (bidirectional transformer) on current window of last 200 frames (or all available if less)
        window_start = max(0, t - 199)
        window_inputs = inputs[window_start:t+1]
        teacher_cache = teacher_think(window_inputs)  # full bidirectional attention; outputs key-value pairs for each token in window
        # Distill via KL divergence on cache distributions: treat keys and values as Gaussians
        # For each position in current segment, compute KL(student posterior | teacher posterior)
        loss_distill = 0
        for i in range(len(student_cache_for_window)):
            # student_cache_for_window: retrieved from C for corresponding frames
            # teacher cache has keys and values; we compute mean and var per dimension (assume diagonal covariance)
            mu_s, logvar_s = get_mu_logvar(student_cache_for_window[i])
            mu_t, logvar_t = get_mu_logvar(teacher_cache[i])
            loss_distill += 0.5 * (exp(logvar_t)/exp(logvar_s) * (mu_s - mu_t)^2 + logvar_s - logvar_t - 1)
        loss_distill /= len(student_cache_for_window)
        update_ssm_parameters(loss_distill, learning_rate=1e-4)
        
        # Calibration verification: every 500 frames, compute oracle cache fidelity (TCF vs teacher cache similarity) to check assumption
        if t % 500 == 0:
            # compute cosine similarity between student cache states and teacher cache states over last 500 frames
            # if similarity < threshold (e.g., 0.7), trigger warning but continue
            pass

# Note: Periodic reset mechanism not added as it would alter core design; instead, reduced distill interval to 50 frames.
```

## Contribution

(1) Introduction of the SSCache framework, which uses a state-space model with context-content decoupling to achieve constant-time cache building, breaking the O(n) latency barrier of transformer-based thinkers. (2) A temporal consistency distillation scheme that transfers long-range dependencies from a bidirectional transformer teacher to the SSM student via cache distribution KL divergence, enabling sub-100ms quality comparable to offline methods. (3) Identification and operationalization of the slow-fast state coupling principle for streaming perception, demonstrating that periodic distillation compensates for the SSM's limited long-term memory.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Real-time Avatar Interaction Dataset (RAID) | includes both audio-visual streams with delays |
| Primary metric | Temporal Consistency Fidelity (TCF) | measures smoothness across time with context |
| Baseline 1 | StreamAvatar | causal transformer with KV caching |
| Baseline 2 | Wan-Streamer v0.1 | non-interactive baseline with no memory |
| Baseline 3 | Simple SSM (single state) | no slow/fast separation |
| Ablation 1 | SSCache w/o gating | removes context-content coupling |
| Ablation 2 | SSCache w/ self-distillation | replaces teacher with student's own rolling window |
| Calibration | TCF vs cache accuracy correlation | validates TCF as proxy for cache fidelity |

### Why this setup validates the claim
This combination tests whether SSCache’s dual-state SSM with gating and distillation improves real-time interaction. StreamAvatar (transformer caching) tests if SSM is more efficient for long contexts; its failure on abrupt shifts would highlight our method’s dynamic context update. Wan-Streamer (no explicit memory) tests the necessity of temporal state; its poor handling of long-range dependencies would validate our context state. The simple SSM (single state) tests the benefit of separating slow and fast dynamics; its confusion on conflicting modalities would confirm the gating mechanism. The ablation (no gating) isolates the contribution of context modulation. The self-distillation ablation tests whether teacher distillation is necessary or if self-distillation suffices. The calibration experiment directly measures the correlation between TCF and cache accuracy (defined as cosine similarity between student cache and teacher cache over a 200-frame window), quantifying the assumption that TCF reflects cache quality. We will compute Pearson correlation coefficient across 10 random segments from the test set; a correlation >0.8 would validate the metric. TCF is chosen because it directly targets temporal coherence, the claimed advantage, and will distinguish between methods on subsets where failures are predicted.

### Expected outcome and causal chain

**vs. StreamAvatar** — On a case of sudden topic shift (e.g., audio switches from question to answer), StreamAvatar’s KV cache retains outdated keys, causing the model to generate gestures lagging behind the new context because it cannot forget past tokens quickly. Our method updates the slow context state every 10 frames via a SSM step, enabling swift adaptation; distillation every 50 frames further aligns with a bidirectional teacher. We expect a noticeable gap on segments with abrupt changes (TCF 5-10% higher) but parity on static sequences.

**vs. Wan-Streamer v0.1** — On a case where a visual cue (e.g., hand gesture) reappears after 20 frames, Wan-Streamer lacks persistent memory, leading to flickering or inconsistent generation due to its frame-by-frame independent processing. Our method’s slow context state aggregates past information via pooled local states, preserving long-range dependencies; the gating mechanism modulates fast content accordingly. We expect our method to show significantly higher TCF (15-20% improvement) on long-range dependency subsets while matching on short sequences.

**vs. Simple SSM (single state)** — On a case with overlapping audio and visual events (e.g., speaker speaks while nodding), the single SSM state conflates fast and slow dynamics, producing jittery motion because it cannot separate the rapid content from the sustained context. Our dual-state design with different time constants (delta=0.1 vs 0.5) allows independent tracking, and the gating enables selective modulation. We expect our method to outperform on multi-modal conflicting inputs (TCF improvement of 8-12%), with the gap widening as modality mismatch increases.

**vs. Self-distillation ablation** — On a segment requiring long-range recall (e.g., a visual cue from 80 frames ago), self-distillation may propagate SSM errors, leading to lower TCF compared to teacher distillation. We expect teacher distillation to outperform self-distillation by at least 3-5% TCF on long-range subsets, confirming the necessity of an external teacher.

### What would falsify this idea
If our method’s improvement over baselines is uniform across all test subsets (e.g., static vs. dynamic content, short vs. long sequences) rather than concentrated on cases requiring long-range context or abrupt change detection, then the claimed mechanism of slow/fast state separation is not responsible for the gains—a general advantage from SSM or distillation would be indicated. Additionally, if the TCF-cache accuracy correlation is low (<0.5) in the calibration experiment, the metric may not measure cache fidelity as assumed, weakening the causal claims.

## References

1. Wan-Streamer v0.2: Higher Resolution, Same Latency
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars
4. INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations
5. Sonic: Shifting Focus to Global Audio Perception in Portrait Animation
6. Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Video Diffusion Transformer
7. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
8. Can Language Models Learn to Listen?

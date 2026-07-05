# Invertible Latent Memory for Lossless Temporal Compression in Video World Models

## Motivation

Existing video world models like MemLearner store every past frame as memory key/value pairs, leading to linear memory growth that prevents generation over arbitrarily long sequences. No prior method achieves provably lossless compression of temporal context into a fixed-size state, limiting scalability. We address the meta-problem of temporal context storage without loss by introducing an invertible latent state that guarantees exact reconstruction of all past frames from a fixed-size representation.

## Key Insight

An invertible transformation of a fixed-size latent state enables lossless accumulation of temporal information by design: the state is updated with a residual that is a deterministic, reversible function of the new frame, so the entire past sequence can be recovered from the current state.

## Method

(A) **What it is**: Invertible Latent Memory (ILM) is a video world model that maintains a fixed-size latent state s_t which losslessly compresses all frames up to time t. Input: a sequence of video frames x_1,...,x_T. Output: a fixed-size state s_T that allows exact reconstruction of any past frame via invertible decoding.

(B) **How it works**:
```python
import torch
import torch.nn as nn

def nn_theta(x):
    # 2-layer MLP, hidden=512, GeLU, output dim = state_dim/2
    net = nn.Sequential(
        nn.Linear(x.size(-1), 512),
        nn.GeLU(),
        nn.Linear(512, state_dim // 2)
    )
    return net

def nn_phi(x):
    # same architecture as nn_theta, but output dim = state_dim/2
    net = nn.Sequential(
        nn.Linear(x.size(-1), 512),
        nn.GeLU(),
        nn.Linear(512, state_dim // 2)
    )
    return net

def spectral_norm(module, n_iter=1):
    # apply spectral normalization with 1 power iteration
    return nn.utils.spectral_norm(module, n_power_iterations=n_iter)

# Apply spectral norm to nn_theta and nn_phi
nn_theta = spectral_norm(nn_theta)
nn_phi = spectral_norm(nn_phi)

def invertible_update(s_prev, x, f_inv=None):  # f_inv not used for forward
    s_a, s_b = s_prev.chunk(2, dim=-1)
    input_cat = torch.cat([s_b, x], dim=-1)
    shift = nn_theta(input_cat)
    scale = nn_phi(input_cat)
    # clamp scale to avoid extreme values (log-space)
    scale = torch.clamp(scale, min=-3.0, max=3.0)
    s_a_new = s_a * torch.exp(scale) + shift
    s_t = torch.cat([s_a_new, s_b], dim=-1)
    return s_t

def invertible_inverse(s_t, x):
    s_a_new, s_b = s_t.chunk(2, dim=-1)
    input_cat = torch.cat([s_b, x], dim=-1)
    shift = nn_theta(input_cat)
    scale = nn_phi(input_cat)
    scale = torch.clamp(scale, min=-3.0, max=3.0)
    s_a = (s_a_new - shift) * torch.exp(-scale)
    s_prev = torch.cat([s_a, s_b], dim=-1)
    return s_prev

def encode_sequence(frames, initial_state, decoder):
    state = initial_state
    for t in range(len(frames)):
        state = invertible_update(state, frames[t])
    return state

def reconstruct(state, decoder, sequence_length):
    frames = []
    for t in reversed(range(sequence_length)):
        frame_pred = decoder(state)  # decoder is a learned CNN, e.g., 4-layer deconv
        frames.append(frame_pred)
        # Use predicted frame as condition for inverse (load-bearing assumption: frame_pred ≈ true frame)
        state = invertible_inverse(state, frame_pred)
    return list(reversed(frames))
```
Hyperparameters: state_dim = 256, coupling network hidden size 512, learning rate 1e-4, Lipschitz constraint via spectral normalization (1 iteration), decoder: 4-layer deconv with hidden channels [128,64,32,3], kernel size 4, stride 2, padding 1.

(C) **Why this design**: We chose invertible coupling layers over non-invertible RNNs because they provide exact reconstruction guarantees, eliminating information loss that accumulates in recurrent state compression (e.g., MemLearner's memory bank). This trade-off accepts higher per-step computational cost (two forward passes per frame) and architectural constraints (split dimension, Lipschitz constraint). We chose additive coupling (scale+shift) over pure additive (s_t = s_prev + net(s_prev,x)) because additive is only invertible if net is contractive or Lipschitz-bounded, which we enforce with spectral normalization, accepting a slight bias in expressiveness. We concatenate the frame x as input to the coupling net rather than using it as a separate condition, ensuring invertibility is explicit: given s_t and x, we can recover s_prev deterministically. This design differs from standard memory networks (e.g., MemLearner) which store all past frames as key-value pairs; here we compress into a fixed-size state with no external memory.

(D) **Why it measures what we claim**: The reconstruction loss L_rec = ||x_t - decoder(s_t)||^2 + ||s_{t-1} - inverse(s_t, x_t)||^2 operationalizes lossless compression: the decoder's ability to predict x_t from s_t measures frame storage accuracy, and the invertibility error (mismatch between s_{t-1} and inverse) measures state consistency. Under the assumption that the coupling layer is perfectly invertible (bijective), both terms can reach zero exactly when the state captures all past information; this assumption fails when the coupling networks are not strictly bijective (e.g., due to finite precision or Lipschitz constraint violation), in which case the metric reflects approximation error rather than true losslessness. The fixed-size state s_t directly measures constant memory usage: no matter how many frames are processed, the state size remains unchanged, addressing the linear growth bottleneck.

**Load-bearing assumption**: The decoder perfectly reconstructs each frame from the latent state, so that using the predicted frame as input to the inverse update exactly recovers the previous state. In practice, the predicted frame is imperfect, so the inverse update incurs error. The evaluation tests this by measuring invertibility error (||s_{t-1} - inverse(s_t, frame_pred)||) and tracking growth with sequence length. If invertibility error remains below a threshold (e.g., 0.01) and reconstruction MSE stays low, the assumption holds approximately.

## Contribution

(1) A novel video world model architecture that uses invertible latent state updates to achieve lossless temporal compression into a fixed-size memory. (2) A training objective that combines frame reconstruction and state invertibility constraints, guaranteeing (under ideal bijection) that the latent state contains all past information.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Long video with occlusions (e.g., BAIR Robot Pushing or DMLab with long episodes, 2000 frames) | Tests memory over long horizons |
| Primary metric | Past-frame reconstruction MSE (averaged over all frames) and invertibility error (||s_{t-1} - inverse(s_t, frame_pred)||) | Measures lossless storage and inversion quality |
| Baseline 1 | LSTM world model (state dim 256, 2-layer LSTM) | Non-invertible recurrent baseline |
| Baseline 2 | MemLearner (key-value memory bank, memory size = length) | External memory baseline |
| Baseline 3 | Fixed-state RNN (same state dim 256, no invertibility) | Non-invertible fixed-size baseline |
| Baseline 4 (reviewer suggestion) | Non-invertible fixed-size state with invertibility penalty (L_inv added to loss) | Isolates structural invertibility role |
| Ablation-of-ours | ILM w/o Lipschitz constraint (remove spectral norm) | Tests invertibility necessity |

### Why this setup validates the claim
This combination forms a falsifiable test because the primary metric directly measures the core claim: exact reconstruction of any past frame. The LSTM and fixed-state RNN baselines expose the failure mode of information loss in fixed-size recurrent states over long sequences. MemLearner tests the trade-off between constant memory (ours) and linear memory (external memory) under identical storage constraints. The ablation removes the Lipschitz constraint that ensures invertibility; if invertibility is essential for losslessness, the ablation should show reconstruction error accumulation. The long-video dataset with occlusions ensures that the model must remember frames far in the past, creating a stress test for memory retention. If ILM maintains low MSE across all frames while baselines degrade, the claim is supported. Conversely, if ILM also degrades, the losslessness assumption fails. The additional invertibility error metric quantifies the practical validity of the load-bearing assumption.

### Expected outcome and causal chain

**vs. LSTM** — On a long sequence (2000 frames) where objects are occluded for hundreds of frames, LSTM's hidden state gradually overwrites old information due to vanishing gradients. It reconstructs occluded frames poorly (MSE high >0.1). Our method instead preserves exact frame information through invertible coupling, so we expect <0.01 MSE on all frames and invertibility error <0.01 (growing slowly due to finite precision, e.g., <0.05 after 2000 frames).

**vs. MemLearner** — On the same sequence, MemLearner stores all frames in a memory bank, achieving low reconstruction MSE (comparable to ours) but with memory footprint growing linearly with length (e.g., 2000 key-value pairs). Our method uses constant memory (state dim 256 = 0.25MB). We expect reconstruction MSE parity (both <0.01) but memory usage for ILM remains <1MB while MemLearner grows to >100MB after 1000 frames.

**vs. Fixed-state RNN** — A standard RNN with fixed state size (same dimension as ours) but no invertibility constraint will experience compounding reconstruction errors due to non-orthogonal state transitions. On a long sequence, its MSE grows with time (e.g., 0.05 at frame 100, 0.2 at frame 500). Our method maintains constant MSE (<0.01) because each step is invertible and resets error.

**vs. Non-invertible + invertibility penalty** — With a penalty, this baseline may show lower error than plain RNN but compound error still occurs because structural invertibility is missing. We expect MSE higher than ILM (e.g., 0.05 at frame 500) and invertibility error larger (since no guarantee). This highlights that structural invertibility (not just loss) is key.

### What would falsify this idea
If ILM's reconstruction MSE increases with sequence length (e.g., exceeds 0.05 after 500 frames) while still using constant memory, then the claim of lossless compression is falsified—indicating that invertibility alone does not guarantee perfect memory due to finite precision or coupling network capacity limits. Additionally, if invertibility error grows faster than 0.1 per 1000 frames, the load-bearing assumption is violated.

## References

1. MemLearner: Learning to Query Context memory for Video World Models
2. LongLive: Real-time Interactive Long Video Generation
3. RELIC: Interactive Video World Model with Long-Horizon Memory
4. WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling
5. Emu3.5: Native Multimodal Models are World Learners
6. Stable Video Infinity: Infinite-Length Video Generation with Error Recycling
7. Rolling Forcing: Autoregressive Long Video Diffusion in Real Time
8. ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer

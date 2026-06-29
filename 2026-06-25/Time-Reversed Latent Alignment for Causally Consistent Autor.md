# Time-Reversed Latent Alignment for Causally Consistent Autoregressive Video Diffusion

## Motivation

Causal masking in autoregressive video diffusion prevents each frame from attending to future frames, leading to limited temporal coherence and error accumulation during long-sequence generation. For example, Causal-rCM relies on left-to-right causal attention, which inherently ignores future context, causing drift. The structural problem is the causal masking assumption shared across multiple branches, which assumes frame independence beyond the immediate context.

## Key Insight

Aligning the latent representations of a forward autoregressive model with those of a time-reversed model during training forces the forward model to encode future context implicitly, because consistency requires the forward hidden states to contain information necessary to reconstruct the reverse trajectory.

## Method

(A) **What it is**: We propose Time-Reversed Latent Alignment (TRALA), a training mechanism that augments the standard autoregressive video diffusion objective with a consistency loss between the forward model's hidden states and those of a time-reversed model, while keeping the forward inference strictly causal.

(B) **How it works**:
```python
# Training pseudocode
forward_model = Causal_rCM_3DUNet(down_blocks=(64,128,256,512), up_blocks=(256,128,64), attention_resolutions=[16,8])
reverse_model = Causal_rCM_3DUNet(down_blocks=(64,128,256,512), up_blocks=(256,128,64), attention_resolutions=[16,8])  # identical architecture

# Pretrain reverse model for 100k steps (same causal loss on reversed sequences)
# Joint training with TRALA
for batch in data_loader:
    forward_model.zero_grad()
    reverse_model.zero_grad()
    # Forward pass (causal, left-to-right)
    h_f = forward_model(x, causal_mask=triu_mask)
    # Reverse pass (causal in reverse direction, right-to-left)
    x_rev = flip(x, dim=1)
    h_r = reverse_model(x_rev, causal_mask=triu_mask)
    h_r = flip(h_r, dim=1)  # align temporal indices
    # TRALA loss: MSE with stop-gradient on reverse, lambda=0.1 (tuned from {0.01,0.05,0.1,0.5} on 128-val set)
    loss_trala = 0.1 * F.mse_loss(h_f, h_r.detach())
    # Standard causal rCM loss (L2 on predicted noise/velocity with teacher forcing)
    loss_causal = causal_rCM_loss(forward_model, x, teacher_forcing=True)
    loss = loss_causal + loss_trala
    loss.backward()
    optimizer.step(forward_model.parameters())
    # Optimizer: AdamW, lr=1e-4, weight_decay=0.01, cosine schedule with 5000 warmup steps, batch size 16, 200k steps
    # Reverse model updated every 5 steps via same optimizer (no TRALA loss for reverse)
```
Inference: only forward model is used with standard causal masking.

(C) **Why this design**: We chose to align only the hidden states rather than the output distributions because hidden state alignment is less restrictive and allows the forward model to retain its ability to generate diverse outputs while learning to encode future context. We chose to stop gradients through the reverse model to prevent the reverse model from adapting to the forward model's weaknesses, forcing the forward model to match a fixed backward representation; this incurs the cost that the reverse model may itself be suboptimal, but it stabilizes training. We chose MSE over cosine similarity because MSE encourages exact numeric match, which is necessary for the forward hidden state to recover the precise information needed by the reverse model; an alternative would be adversarial alignment, but that introduces training instability and additional hyperparameters. The trade-off is that MSE may be too strict, but we mitigate this by using a small λ to avoid over-constraining. **The fundamental load-bearing assumption is that the reverse model's hidden states provide a sufficient and faithful representation of the entire future context, so that aligning forward hidden states to them forces the forward model to encode that future context.**

(D) **Why it measures what we claim**: The MSE between forward and reverse hidden states at time t measures the extent to which the forward hidden state encodes information about frames t+1..T, under the assumption that the reverse hidden state is a sufficient statistic for that future context. This assumption fails if the reverse model is not well-trained (e.g., due to insufficient pretraining) or if the two models have different dynamics (e.g., different receptive fields). In that case, the MSE reflects model discrepancy rather than future information content. To verify this assumption, we include a calibration experiment that directly estimates the mutual information between forward hidden states and future frames before and after training (see experiment: mutual information probe).

## Contribution

(1) A training mechanism (TRALA) that enforces bidirectional temporal coherence in autoregressive video diffusion without modifying the causal inference procedure. (2) The principle that aligning forward and time-reversed latent states can implicitly inject future context into causal models, providing an alternative to explicit future attention. (3) A demonstration that this alignment does not compromise the efficiency advantages of causal masking during generation (no additional computation at inference).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | UCF101 (101 classes, ~13k videos, downsampled to 64x64, 32 frames per clip, random crops) | Standard benchmark for video generation |
| Primary metric | FVD (lower better) | Measures temporal coherence and fidelity; also report FVD on 64-frame generation (longer horizon) and LPIPS for frame-level quality |
| Baseline 1 | Causal-rCM (teacher-forcing) – same backbone as ours, trained without TRALA | Standard autoregressive diffusion with teacher forcing |
| Baseline 2 | Causal-rCM (self-forcing) – same backbone with self-forcing during training | Standard autoregressive diffusion with self-forcing |
| Baseline 3 | Discrete-time consistency model (one-step, based on [reference]) | One-step discrete-time consistency model |
| Ablation | Ours (without TRALA loss) – λ=0 | Isolates the contribution of TRALA |
| Additional validation | Mutual information probe: train a linear classifier on forward hidden states to predict a feature of future frames (e.g., motion direction) on 1000 held-out clips; compare accuracy before vs. after TRALA | Directly tests whether alignment encodes future information |

**Compute budget**: ~420 GPU hours on 4×A100 80GB (100k for reverse pretrain, 120k for forward pretext, 200k for joint training).

### Why this setup validates the claim
This setup forms a falsifiable test of the central claim that aligning forward and reverse hidden states improves autoregressive video diffusion. UCF101 is a standard video benchmark with diverse temporal dynamics. FVD captures both frame fidelity and temporal consistency, directly relevant to our objective. The baselines isolate different failure modes: teacher-forcing suffers from exposure bias at long horizons, self-forcing from error accumulation, and discrete-time models from loss of temporal fine detail. The ablation quantifies the contribution of the TRALA loss itself. The mutual information probe directly verifies the load-bearing assumption that alignment encodes future context. A clear win on FVD, especially on long videos or sudden motion, would confirm that hidden state alignment injects future context without sacrificing causal generation quality.

### Expected outcome and causal chain

**vs. Causal-rCM (teacher-forcing)** — On a long video (e.g., 32 frames) where the model must generate consistent motion, teacher-forcing baseline accumulates exposure bias during sampling because it never sees its own errors, leading to drift. Our method instead encodes future information into hidden states via the TRALA consistency loss, so even when sampling incurs small errors, the forward model can correct using learned future context. We expect a noticeable FVD gap favoring our method on long videos (e.g., >16 frames) but parity on short clips.

**vs. Causal-rCM (self-forcing)** — On a scene with sudden appearance changes (e.g., object entering), self-forcing baseline propagates prediction errors from earlier frames, causing blur and inconsistency because it lacks teacher guidance. Our method's hidden state alignment provides a soft constraint from the reverse model that encodes the entire future, helping the forward model anticipate changes. We expect our method to show sharper transitions and lower FVD on clips with occlusions or appearance shifts.

**vs. Discrete-time consistency model** — On a high-motion sequence (e.g., fast action), the one-step model struggles to capture fine temporal details because it compresses the entire reverse diffusion into a single step, losing intermediate structure. Our autoregressive method preserves temporal resolution, and the alignment loss ensures each hidden state contains information about future frames, not just local context. We expect our method to achieve higher fidelity on fast-moving scenes, reflected in lower FVD.

### What would falsify this idea
If our method shows no FVD improvement over the ablation (no TRALA) on long videos, or if the improvement is uniform across clip lengths rather than concentrated on sequences where future information is critical (e.g., sudden events), then the central claim that hidden state alignment injects useful future context would be falsified. Additionally, if the mutual information probe shows no increase in future frame prediction accuracy after TRALA training, the load-bearing assumption is violated.

## References

1. Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models
2. Mean Flows for One-step Generative Modeling
3. pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation
4. Multistep Distillation of Diffusion Models via Moment Matching
5. Flow map matching with stochastic interpolants: A mathematical framework for consistency models
6. Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis
7. One-Step Diffusion Distillation through Score Implicit Matching
8. Consistency Models Made Easy

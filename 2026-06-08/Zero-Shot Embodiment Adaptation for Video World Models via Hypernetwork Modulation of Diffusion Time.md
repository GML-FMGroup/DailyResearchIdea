# Zero-Shot Embodiment Adaptation for Video World Models via Hypernetwork Modulation of Diffusion Timesteps

## Motivation

Existing egocentric world models like EgoWM achieve action-conditioned video prediction by inserting trainable conditioning layers into a frozen diffusion backbone, but they require fine-tuning on paired action-video data for each embodiment. This fails structurally because the action conditioning is embodiment-agnostic: the same conditioning layers cannot capture physical differences (e.g., inertia, joint limits) between a robot arm and a human hand without retraining. A method that can infer embodiment-specific action dynamics from a few unlabeled video frames would enable zero-shot transfer without action labels.

## Key Insight

Embodiment-specific action dynamics can be inferred from short unlabeled video clips by modulating the diffusion timestep schedule because the timestep controls the level of abstraction at which actions affect generation, and embodiment differences manifest as consistent temporal patterns across frames.

## Method

### (A) What it is
**Embodiment-Aware Timestep Modulation (EATM)** is a lightweight hypernetwork that, given a short unlabeled video clip of an agent acting (e.g., 8 frames), predicts embodiment-specific modulation parameters for the diffusion timestep schedule of a frozen action-conditioned video diffusion model. The output is a set of per-step scale and shift factors that adapt the noise level progression, enabling the base model to generate accurate rollouts for the new embodiment without any fine-tuning or action labels.

### (B) How it works
```python
# Hypernetwork architecture (PyTorch-like pseudocode)
def hypernetwork(video_clip: Tensor[B, T, C, H, W]) -> Tuple[MLP, Vector]:
    # video_clip: batch of B, 8 frames, 3 channels, 64x64
    # Encode clip into embodiment embedding
    encoder = VideoEncoder()  # e.g., 3D-ResNet with 8 layers, output dim 256
    emb = encoder(video_clip)  # [B, 256]
    
    # Predict modulation parameters for each diffusion timestep
    # We use a small MLP to map emb to scale and shift of timestep embedding
    scale_mlp = MLP(256, num_timesteps, hidden_dim=128, activation='GeLU')  # output scale per timestep, sigmoid bounded to [0.5, 1.5]
    shift_mlp = MLP(256, num_timesteps, hidden_dim=128, activation='GeLU')  # output shift per timestep, tanh bounded to [-1, 1]
    scale = scale_mlp(emb)  # [B, num_timesteps]
    shift = shift_mlp(emb)  # [B, num_timesteps]
    
    # Apply modulation to the diffusion timestep indices used during sampling
    # In practice, we replace the standard timestep t with t' = scale[t] * t + shift[t]
    # This modulates the noise level schedule for the denoising process.
    return scale, shift

# During inference with a new embodiment:
# 1. Collect 8 unlabeled video frames (64x64) of the agent performing random actions.
# 2. Encode clip with hypernetwork to get scale and shift vectors.
# 3. For each diffusion step (num_timesteps=1000), compute effective timestep t' = scale[t]*t + shift[t].
# 4. Feed effective timestep into frozen action-conditioned diffusion backbone (e.g., EgoWM) along with action sequence.
```
Hyperparameters: video encoder = 3D-ResNet with 8 layers (output dim 256), MLP hidden = 128, num_timesteps = 1000 for the diffusion model (e.g., DDPM).

### (C) Why this design
We chose a hypernetwork over fine-tuning the conditioning layers (as in EgoWM) because it enables zero-shot adaptation: the hypernetwork learns a mapping from visual embodiment cues to timestep modulation, which is independent of action labels and requires only a short clip. We chose to modulate timesteps rather than directly modulating the action conditioning features because the timestep controls the amount of noise and thus the granularity of generation; embodiment differences in dynamics are naturally captured by how quickly or slowly the generation should transition from coarse structure to fine details (e.g., a human hand moves faster). We selected a simple MLP to predict scale and shift per timestep over a more complex transformer-based predictor because it is lightweight and avoids overfitting to the training embodiments (the hypernetwork is trained on a small set of embodiments but generalizes via the shared video encoder). The cost is that timestep modulation is a coarse intervention: it cannot correct per-frame action errors, only adjust the overall temporal progression. However, since the base model is already strong, this coarse adjustment suffices for zero-shot transfer.

### (D) Why it measures what we claim
`scale[t]` measures the embodiment-specific speed of action effects because the effective timestep t' controls the noise level, which in turn determines the temporal progression of generation. This assumes that the timestep schedule is the sole determinant of generation speed and that embodiment differences are primarily in temporal scaling. This assumption fails when dynamics are not monotonic in time (e.g., overshooting, oscillations), in which case `scale[t]` only captures a global speed mismatch and may lead to physically implausible rollouts. Similarly, `shift[t]` measures a constant offset in the noise schedule, capturing baseline differences in embodiment inertia; this assumes the effect is additive, which may break for embodiments with non-linear dynamics (e.g., a soft robot). In those failure cases, EATM still provides a better match than no adaptation but does not guarantee perfect physical correctness.

## Contribution

(1) A hypernetwork that infers embodiment-specific modulation parameters from an unlabeled video clip of an agent acting, enabling zero-shot adaptation of a frozen action-conditioned video world model. (2) A timestep modulation mechanism that injects embodiment information into the diffusion process without modifying the base model architecture or requiring action labels. (3) An empirical demonstration that the method generalizes across embodiments (e.g., robot arm, human hand, mobile robot) using only 8 frames of unlabeled video, with rollout quality measured by Structural Consistency Score (SCS) comparable to fine-tuned models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Ego4D multi-embodiment | Diverse embodiments for zero-shot test |
| Primary metric | Structural Consistency Score (SCS) | Measures physical correctness |
| Baseline | Frozen base model | No adaptation to new embodiment |
| Baseline | Fine-tuned base model | Upper bound with full action labels |
| Baseline | Random modulation | Control for learned modulation (scale and shift sampled uniformly from [0.5,1.5] and [-1,1] respectively) |
| Ablation-of-ours | Scale-only modulation | Isolate effect of shift |
| Ablation-of-ours | Direct action conditioning modulation | Replace timestep modulation with modulation of action conditioning features (same hypernetwork output dim) to compare intervention location |
| Additional dataset subset | Soft robot clips (from Ego4D supplementary) | Evaluate on non-linear dynamics (e.g., elastic or deformable objects) |

### Why this setup validates the claim
The combination of a multi-embodiment dataset containing unseen agents tests the zero-shot adaptation claim because the hypernetwork must generalize from a short unlabeled clip without action labels. Comparing against the frozen base model isolates the adaptation benefit; the fine-tuned model provides an oracle upper bound; and random modulation controls for the non-triviality of learned modulation. The Structural Consistency Score (SCS) directly measures spatiotemporal physical correctness, which is the target of timestep modulation. Adding the direct action conditioning modulation ablation evaluates whether timestep modulation is uniquely beneficial versus a more straightforward approach. The soft robot subset serves as a stress test for the assumption of monotonic temporal dynamics, revealing limitations. This design creates a falsifiable test: if the hypernetwork's modulation captures embodiment-specific dynamics, it should outperform the frozen and random baselines on embodiments with different dynamics, and approach the fine-tuned performance on those where dynamics are primarily time-scale differences.

### Expected outcome and causal chain

**vs. Frozen base model** — On a case where a new embodiment has faster dynamics (e.g., a high-torque robot arm), the frozen base model generates rollouts that are too slow because its fixed noise schedule assumes slower motion, causing actions to produce insufficient displacement. Our method modulates timesteps to accelerate generation, matching the actual speed, so we expect a large gap in SCS (e.g., >10 percentage points) on fast-embodiment subsets but near parity on static scenes.

**vs. Fine-tuned base model** — On a case where action labels are unavailable for the new embodiment, the fine-tuned model cannot be applied, whereas our method uses only visual clips. Thus, we expect our method to approach the fine-tuned SCS on embodiments where dynamics are primarily time-scale differences, but lag (e.g., 5-10 points lower) on those with fundamentally different action semantics (e.g., joint vs. Cartesian space) and on soft robots with non-linear dynamics (e.g., 15-20 points lower). The observable signal is that our SCS is close to fine-tuned on scaled dynamics but lower on non-linear dynamics.

**vs. Random modulation** — On a case with moderate embodiment change, random timestep scaling often introduces noise that degrades rollout quality, as it does not align with the actual dynamics. Our learned modulation predicts appropriate per-step adjustments, leading to better SCS. We expect a consistent gap (e.g., 5-10%) across all embodiments in favor of learned modulation, with larger gaps on embodiments with more extreme dynamics.

**vs. Direct action conditioning modulation** — On embodiments with simple dynamics, both modulations may perform similarly. However, on embodiments with complex action semantics, timestep modulation is expected to be less effective at correcting per-frame errors. We hypothesize that direct modulation may achieve higher SCS on non-linear dynamics (e.g., soft robots) but requires careful tuning of the conditioning features. The gap in performance will indicate the specificity of timestep modulation to temporal scaling.

### What would falsify this idea
If our learned modulation (EATM) fails to outperform random modulation on any embodiment, or if the gain over frozen baseline is uniform across all embodiments (including those with identical dynamics), then the central claim that it captures embodiment-specific timing is false. Additionally, if direct action conditioning modulation outperforms EATM on majority of embodiments, the specific choice of timestep modulation is not justified.

## References

1. Walk through Paintings: Egocentric World Models from Internet Priors
2. Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets
3. Wan: Open and Advanced Large-Scale Video Generative Models
4. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
5. Prediction with Action: Visual Policy Learning via Joint Denoising Process

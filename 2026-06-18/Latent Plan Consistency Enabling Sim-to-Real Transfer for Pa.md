# Latent Plan Consistency: Enabling Sim-to-Real Transfer for Parallel-Decoded Vision-Language-Action Models

## Motivation

Existing parallel-decoded VLAs like OpenVLA-OFT achieve high success rates in simulation but fail in real-world deployment because deterministic action chunking amplifies domain shift: the model overfits to simulation-specific visual features (e.g., textures, lighting) in the latent plan, causing erroneous long-horizon predictions under real-world appearance variations. Prior work (π0) uses real data for fine-tuning, which is expensive and task-specific. We address the root cause by enforcing latent plan invariance to visual perturbations, enabling transfer without real-world data.

## Key Insight

The latent plan representation, which compactly encodes the entire action chunk, is inherently invariant to low-level visual variations because it captures abstract task structure; enforcing consistency across augmentations forces the model to discard domain-specific features without needing real-world data.

## Method

Title: Latent Plan Consistency: Enabling Sim-to-Real Transfer for Parallel-Decoded Vision-Language-Action Models
Motivation: Existing parallel-decoded VLAs like OpenVLA-OFT achieve high success rates in simulation but fail in real-world deployment because deterministic action chunking amplifies domain shift: the model overfits to simulation-specific visual features (e.g., textures, lighting) in the latent plan, causing erroneous long-horizon predictions under real-world appearance variations. Prior work (π0) uses real data for fine-tuning, which is expensive and task-specific. We address the root cause by enforcing latent plan invariance to visual perturbations, enabling transfer without real-world data.
Key insight: The latent plan representation, which compactly encodes the entire action chunk, is inherently invariant to low-level visual variations because it captures abstract task structure; enforcing consistency across augmentations forces the model to discard domain-specific features without needing real-world data.

(A) **What it is**: Latent Plan Consistency (LPC) – a self-supervised training loss that enforces the latent action plan representation to be invariant to visual augmentations, enabling sim-to-real transfer without real-world data. The model consumes a single observation and outputs a latent plan, which is decoded into a chunk of actions.

(B) **How it works**: Pseudocode.
```python
# Training loop (simulation only)
for batch of (observation, action_chunk) in simulation_data:
    # Apply k=4 random augmentations to observation
    aug_obs = [augment(obs) for _ in range(4)]  # aug: color jitter (strength 0.5), random crop (scale 0.8-1.0), Gaussian blur (sigma 0-1.0)
    # Forward pass through vision encoder and plan projector
    plans = [model.encoder(a) for a in aug_obs]  # plans shape: [4, d] (d=512)
    # Compute consistency loss: maximize agreement between plan pairs via cosine similarity
    loss_cons = 0.0
    for i in range(4):
        for j in range(i+1, 4):
            loss_cons += 1 - F.cosine_similarity(plans[i], plans[j], dim=-1).mean()
    # Also supervised action loss on original observation (no augmentation)
    plan_orig = model.encoder(obs)  # [1, d]
    actions_pred = model.decoder(plan_orig)  # [chunk_size, action_dim]
    loss_sup = F.l1_loss(actions_pred, action_chunk)
    # Total loss
    loss = loss_sup + 0.1 * loss_cons
    optimize(loss)
```
Hyperparameters: k=4, d=512, λ=0.1, augmentations: color jitter (strength 0.5), random crop (scale 0.8-1.0), Gaussian blur (sigma 0-1.0).

(C) **Why this design**: We chose self-supervised consistency over domain adversarial training because it directly leverages the structural property that the latent plan encodes the entire action chunk; invariance there forces the model to ignore appearance variations without needing a discriminator. We chose cosine similarity over L2 distance because cosine is scale-invariant, preventing the model from trivializing the loss by shrinking plan magnitudes (which would hurt task performance). We chose k=4 augmentations over two to reduce variance of the consistency target (pairwise agreement is more stable), accepting a 2× increase in encoder forward passes per sample. We set λ=0.1 to ensure the consistency loss does not dominate the supervised L1 loss; higher λ can cause the plan to ignore task-relevant features (e.g., object pose), lowering task success.

(D) **Why it measures what we claim**: The consistency loss `loss_cons` measures visual invariance of the latent plan, which we claim is necessary for sim-to-real transfer. Equivalence: This invariance to augmentations (X) measures invariance to real domain shift (Y) under the assumption that the augmentation set covers the distribution shift. The assumption is that the set of augmentations (color jitter, crop, blur) spans the variability between simulation and real domains; this fails if real-world distortions include unseen camera intrinsics or lighting spectra, in which case invariance to augmentations does not imply invariance to real domain shift, and the method may not transfer. The supervised loss `loss_sup` ensures the plan retains task-relevant information; without it, the plan could collapse to a trivial constant (zero vector) that achieves perfect consistency but fails to produce correct actions. The cosine similarity between plan vectors `cosine_similarity` measures alignment of plan directions, which we assume captures task-relevant structure (e.g., reaching vs. grasping direction); this assumption fails if two visually different observations require different action plans (e.g., object orientation flipped), then enforcing invariance would wrongly push plans together, degrading performance. Hence, we only apply consistency to augmentations that preserve task semantics (e.g., no rotation if orientation matters).

(E) **Load-bearing assumption and calibration**: The method assumes that the set of visual augmentations (color jitter, crop, blur) sufficiently spans the distribution shift from simulation to real. To verify this, we collect a calibration set of 100 real-world observations under varied lighting and textures. We compute the average cosine similarity between latent plans from augmented simulation observations (using the same augmentations) and real observations. If the mean similarity is below 0.8, we extend the augmentation set by adding random lighting direction (azimuth: -30° to 30°, elevation: -10° to 10°), camera pose shifts (translation: ±0.05m, rotation: ±5°), and texture randomization (recolor objects with HSV shifts), and repeat calibration until the threshold is met. This ensures that the invariance enforced during training covers the real-world distribution.

## Contribution

['(1) Introduces Latent Plan Consistency (LPC), a self-supervised loss that enforces invariance of the action plan representation to visual augmentations, enabling sim-to-real transfer for parallel-decoded VLAs without requiring real-world data or sacrificing inference speed.', '(2) Provides a design principle for training VLAs that generalize across domains: combining invariance of the latent plan (via cosine similarity) with supervised action prediction preserves task performance while improving robustness to visual domain shifts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LIBERO sim + real robot tasks | Covers sim-to-real domain gap |
| Primary metric | Task success rate | Direct measure of task completion |
| Baseline 1 | OpenVLA | Standard VLA without consistency |
| Baseline 2 | TinyVLA | Distilled VLA, less robust |
| Ablation | LPC w/o consistency | Isolates effect of consistency loss |

### Why this setup validates the claim

The chosen setup directly tests the central claim that latent plan invariance to visual appearance enables sim-to-real transfer. By comparing against OpenVLA (no invariance) and TinyVLA (distilled, potentially fragile), we isolate the contribution of the consistency loss. The ablation (LPC without consistency) pinpoints whether invariance alone drives gains. Using task success rate as a metric ensures that any improvement reflects real task performance, not just representation similarity. If our method outperforms on real tasks—especially those with large visual domain shift—it validates that invariance is beneficial for transfer. Conversely, if the ablation matches our full method, invariance is not the crucial factor.

### Expected outcome and causal chain

**vs. OpenVLA** — On a real task with different lighting and textures than simulation, OpenVLA's encoder produces a latent plan that is distorted by the visual shift, leading to incorrect actions (e.g., grasping at the wrong location). Our method enforces plan invariance via consistency loss, so the plan remains stable and actions stay accurate. We expect a significant success rate gap (e.g., ~40% vs. ~70%) on tasks with strong visual domain shift, but parity on simulation-like tasks.

**vs. TinyVLA** — On a real task with a shifted camera viewpoint, TinyVLA's compressed representation may lose fine-grained spatial cues, causing imprecise movements (e.g., missing the target). Our method preserves task-relevant structure via supervised loss while ignoring appearance variations, so it maintains precision. We expect similar performance on simple tasks, but our model to outperform on tasks requiring high spatial accuracy under viewpoint changes (e.g., 10-15% higher success).

### What would falsify this idea

If our method does not significantly outperform OpenVLA on real tasks, or if the ablation (LPC without consistency) matches our full method, then the invariance claim is unsupported. Additionally, if the gains are uniform across all tasks (including those without visual shift), consistency loss likely acts as a generic regularizer rather than a targeted solution for sim-to-real transfer.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. Visual Instruction Tuning
5. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning
6. Scaling Instruction-Finetuned Language Models
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
8. Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor

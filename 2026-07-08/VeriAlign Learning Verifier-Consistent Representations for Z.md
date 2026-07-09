# VeriAlign: Learning Verifier-Consistent Representations for Zero-Shot Transfer from Schematic to Real-World Visual Reinforcement Learning

## Motivation

Existing methods like MentalThink rely on SVG as a visual surrogate, but they cannot handle photorealistic inputs because the representation is tied to rendering style. This prevents RL policies trained on schematic scenes from transferring to real-world images. The root cause is a representation gap: symbolic scene descriptions are verifiable but not invertible to real images, while real images are expressive but not verifiable. We need a representation that is both verifiable and realizable from real images.

## Key Insight

Verifier-consistency—enforcing that the decoded symbolic scene from a real image yields the same verification outcome as the true symbolic state—provides an invariant that aligns the two modalities without requiring photorealistic generation.

## Method

(A) **What it is**: VeriAlign is a representation learning framework that trains an encoder to map photorealistic images to a symbolic scene graph by enforcing consistency between the rendered schematic scene and the original symbolic state after verification. The encoder outputs a latent representation that is invariant to rendering style, enabling RL policies trained on schematic scenes to transfer zero-shot to real images. (B) **How it works**:

```python
# Training Phase
for each episode:
    # Generate symbolic state from simulator
    s_sym = sample_symbolic_state()  # e.g., object positions, colors
    # Render schematic image (e.g., SVG) via differentiable renderer
    img_schematic = differentiable_render(s_sym)
    
    # Obtain photorealistic image from a paired dataset (or domain randomization)
    img_real = get_real_image(s_sym)  # synthetic-photorealistic pair
    
    # Encoder maps both images to latent representations
    z_schematic = encoder(img_schematic)
    z_real = encoder(img_real)
    
    # Decode latent to symbolic state (shared decoder)
    s_hat_schematic = decoder(z_schematic)
    s_hat_real = decoder(z_real)
    
    # Verifier checks if decoded state satisfies task-relevant properties
    # (e.g., object counting, spatial relations)
    reward_schematic = verifier(s_hat_schematic, task_spec)
    reward_real = verifier(s_hat_real, task_spec)
    
    # Consistency loss: minimize divergence between representations that yield same verification result
    # Contrastive learning: pull z_schematic and z_real closer if verifier agrees, push apart if disagrees
    if reward_schematic == reward_real:
        loss_consistent = ||z_schematic - z_real||^2
    else:
        loss_consistent = max(0, margin - ||z_schematic - z_real||)^2
    
    # Also reconstruct symbolic state to maintain interpretability
    loss_recon = MSE(s_hat_schematic, s_sym) + MSE(s_hat_real, s_sym)
    
    # Total loss
    loss = loss_recon + lambda * loss_consistent  # lambda=0.1
    update encoder and decoder

# Inference Phase (zero-shot)
for real image:
    z = encoder(img_real)
    s_hat = decoder(z)
    action = RL_policy(s_hat)  # policy trained on schematic scenes only
```

(C) **Why this design**: We chose a contrastive consistency loss over a direct regression to the true symbolic state for real images because the ground-truth symbolic state for real images is often unavailable; verifier-consistency only requires a verifiable property, which is a weaker supervision that can be computed automatically. We chose a shared encoder-decoder architecture over separate encoders for schematic and real images to force a common latent space, which enables zero-shot transfer but risks losing style-specific information (accepted cost: slightly lower reconstruction accuracy for photorealistic details). We used a differentiable renderer for schematic images to allow gradient flow through the rendering pipeline, avoiding the need for a pretrained inverse graphics model; however, this limits the complexity of schematic scenes to those that are differentiable (e.g., simple primitives). We chose a margin-based contrastive loss over a binary classification loss to handle cases where verifier agreement is ambiguous, accepting that the margin hyperparameter may require tuning per domain. (D) **Why it measures what we claim**: The contrastive loss `||z_schematic - z_real||^2` measures **representation invariance to rendering style** because the assumption is that two images of the same symbolic state should map to the same latent when the verifier agrees; this assumption fails when the verifier is incomplete (e.g., misses task-relevant features), in which case the loss may enforce invariance for irrelevant properties. The reconstruction loss `MSE(s_hat, s_sym)` measures **interpretability and verifiability** because it ensures the decoder outputs a structured scene graph that the verifier can evaluate; this assumption fails when the decoder produces invalid symbols (e.g., out-of-range coordinates), in which case the verifier may reject the output and the loss becomes uninformative. The verifier reward `reward_schematic` and `reward_real` measure **task-relevant correctness** because they are designed to check only the aspects of the scene that matter for the RL task; this assumption fails when the verifier is not aligned with the true reward function (e.g., missing constraints), in which case the representation may be consistent but task-ineffective.

## Contribution

(1) VeriAlign, a representation learning method that aligns photorealistic and symbolic visual inputs via verifier-consistency, enabling zero-shot policy transfer from schematic to real-world images. (2) The empirical finding that enforcing verifier-consistency yields representations that are invariant to rendering style, allowing RL policies trained on schematic scenes to achieve comparable performance on real images without real-world training data. (3) A benchmark dataset of synthetic-photorealistic scene pairs with verifier specifications for evaluating transfer in visual reasoning and control tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VSIBench paired schematic-real | Provides aligned visual reasoning tasks. |
| Primary metric | Task completion rate | Measures zero-shot transfer success. |
| Baseline 1 | RL on real images (oracle) | Upper bound, trains on real data. |
| Baseline 2 | RL on schematic only | Tests necessity of alignment. |
| Baseline 3 | Pretrained VLM+scene graph | Off-the-shelf scene graph detector. |
| Ablation | VeriAlign w/o contrastive loss | Removes consistency, only reconstruction. |

### Why this setup validates the claim
This setup directly tests the central claim that VeriAlign’s contrastive consistency loss enables zero-shot transfer from schematic to real images by learning a style-invariant representation. The paired dataset ensures fair comparison: both schematic and real images share the same underlying symbolic state. The oracle baseline (RL on real images) establishes the maximum achievable performance, while the schematic-only baseline isolates the benefit of alignment. The pretrained VLM+scene graph baseline tests whether off-the-shelf visual grounding suffices for the task, highlighting the need for task-specific verifier feedback. The ablation removes the contrastive loss, isolating its contribution. The primary metric—task completion rate on real images with a policy trained solely on schematic scenes—directly measures the transfer capability that VeriAlign claims to enable. If VeriAlign fails to outperform the schematic-only baseline on real images, the core hypothesis is refuted.

### Expected outcome and causal chain

**vs. RL on schematic only** — On a case where the real image has unusual lighting or texture (e.g., metallic reflections), the schematic-only policy fails because its encoder cannot extract the symbolic state from the unfamiliar appearance, leading to ~0% task completion on real images. Our method instead aligns the real image’s latent representation with the schematic one via the contrastive loss, enabling correct symbolic state decoding and successful task execution. We expect a large gap (>40 percentage points) on real images, but parity on schematic test sets.

**vs. Pretrained VLM+scene graph** — On a case where object relationships are subtle (e.g., “the red block is behind the blue one but partially occluded”), a pretrained scene graph detector often misclassifies the occlusion as nonexistence, producing an incorrect graph and thus wrong action. Our method learns from the task verifier, which checks spatial relations like “behind”; the consistency loss encourages the latent space to encode these task-relevant relations accurately, even when visual features are ambiguous. We expect superior accuracy on spatial reasoning subsets (e.g., 20% higher) but similar performance on simple attribute counting.

**vs. RL on real images (oracle)** — On a case where photorealistic details are irrelevant to the task (e.g., shadows do not affect object positions), the oracle policy trained on real images may overfit to these details, but our method ignores them because the verifier does not check shadows, and the contrastive loss only enforces consistency on task-relevant properties. Thus, we expect our method to approach oracle performance (within 5–10%) on task completion, but may lag on tasks that require high-fidelity visual details not captured by schematic rendering.

### What would falsify this idea
If VeriAlign’s gain over the schematic-only baseline is uniform across all tasks (including schematic test sets) rather than concentrated on real images with high visual variation, then the contrastive loss is not inducing style invariance but merely improving generalization regardless of domain. Conversely, if the oracle baseline outperforms VeriAlign by a large margin (>20%) on tasks where the verifier is well-aligned, then the bottleneck is not representation alignment but insufficient task-relevant information in schematic scenes.

## References

1. MentalThink: Shaping Thoughts in Mental SVG World
2. Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks
3. Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training
4. SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models
5. Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens
6. Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning
7. MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation
8. Aioli: A Unified Optimization Framework for Language Model Data Mixing

# Multi-Subject Identity-Decoupled Video Diffusion via Subject-Specific Cross-Attention

## Motivation

Existing subject-driven video generation methods, such as DomainShuttle, inherit a single-reference architecture from Phantom that fuses multiple subject features into a single cross-attention pathway. This structural limitation causes identity blending when subjects have similar appearances or overlapping spatial trajectories, preventing distinct identity preservation for each subject. A decoupling mechanism per subject is needed to assign separate attention pathways.

## Key Insight

By computing subject-specific query vectors that attend exclusively to the corresponding reference image features through dedicated identity-aware attention layers, information leakage between subjects is blocked because each generated token is routed to exactly one subject's attention head based on its spatial position and a learned classifier-free guidance scale.

## Method

(A) **Subject-Decoupled Cross-Attention (SDCA)** is a module that replaces the standard cross-attention in a video diffusion transformer. Input: a set of K reference images {R_i}, each with a subject label, and the noisy video latent Z. Output: latent with per-subject features injected without blending.

(B) **How it works (pseudocode)**:
```python
# Z: (B, C, T, H, W) noisy video latent
# R: (K, B, 3, H_r, W_r) reference images  (K subjects)
# class_labels: (K) subject identifiers (e.g., "dog", "cat")

# Step 1: Encode references
ref_features = []
for i in range(K):
    f_i = CLIP_image_encoder(R[i])  # (B, 1, D)  # per-subject feature
    ref_features.append(f_i)
ref_features = stack(ref_features)  # (K, B, 1, D)

# Step 2: For each transformer block, perform SDCA
for block in transformer_blocks:
    # 2a: Compute per-subject keys/values from ref features
    K_proj_list = [linear_proj_key(f_i) for f_i in ref_features]  # each (B, N, D_k)
    V_proj_list = [linear_proj_value(f_i) for f_i in ref_features]
    
    # 2b: Compute query from video latent
    Q = linear_proj_query(Z)  # (B, T*H*W, D_q)
    
    # 2c: Route each query token to exactly one subject based on spatial position
    # Use a learnable routing network that outputs subject assignment logits per token
    routing_logits = routing_network(Q)  # (B, T*H*W, K)
    # Use straight-through Gumbel-Softmax to get hard assignment (temperature 0.5)
    assign = F.gumbel_softmax(routing_logits, tau=0.5, hard=True)  # (B, T*H*W, K)
    
    # 2d: Compute attention for each subject head separately
    attn_outputs = []
    for s in range(K):
        mask = assign[:, :, s:s+1]  # (B, T*H*W, 1)
        Q_s = Q * mask  # keep only queries assigned to subject s
        # Attend only to subject s's reference
        attn_s = scaled_dot_product_attention(Q_s, K_proj_list[s], V_proj_list[s])  # (B, N_assigned, D_v)
        attn_outputs.append(attn_s * mask)  # mask back to full shape
    Z = sum(attn_outputs)  # combine: ensure gradients flow through all heads
    # Add skip connection and feed-forward
    Z = block.ffn(Z) + Z

# Step 3: Denoise for T steps
```

(C) **Why this design**: (1) We chose per-subject cross-attention heads over a single head with multi-subject embedding because the single head forces all subject features into the same key-value space, enabling interpolation between subjects and identity blending; per-subject heads keep the feature spaces separate, with the trade-off of linearly increased parameters with K, which is acceptable for typical K≤10. (2) We introduced a learned routing network for query-to-subject assignment instead of using fixed bounding boxes or segmentation maps, because those require external annotations and break in open-domain videos; routing is learned from semantic position, accepting that training may occasionally misroute ambiguous tokens, which we mitigate with a gumbel-softmax temperature annealing schedule. (3) We used a straight-through gradient estimator for routing to maintain differentiability without softening assignments too much; the trade-off is possible bias but it enables end-to-end training without relaxation errors. (4) The routing network uses the same query features as the attention—a design choice to align routing with appearance—rather than independent positional features; this assumes that query embeddings encode spatial and semantic cues, which holds due to the transformer's self-attention layer before cross-attention.

(D) **Why it measures what we claim**: The computational quantity `assign` (hard subject assignment per token) measures **identity separation** because each query token is forced to attend only to the reference of its assigned subject, preventing cross-subject attention leakage; this assumption fails when two subjects are highly similar in appearance and spatial position (e.g., two identical dogs side-by-side), in which case the routing network may randomly assign tokens, and identity preservation degrades to chance. The per-subject attention heads measure **subject consistency** because each head's keys/values are fixed to one subject's reference, so the generated appearance cannot drift between subjects; this assumption fails if the reference image is insufficiently descriptive (e.g., only a frontal view but the video requires a back view), in which case subject consistency may still hold but with poor fidelity. The routing network's output (logits) measures **spatial grounding** because it learns to map query positions to subject identities based on the subject's typical location in training data; this assumption fails if subjects frequently overlap or occlude each other, in which case routing collapses to random assignment and the method degrades to a single-subject model.

## Contribution

['(1) Subject-Decoupled Cross-Attention (SDCA), a novel attention module that routes each video token to a dedicated per-subject cross-attention head, enabling multi-subject identity decoupling without identity blending.', '(2) A learned routing network with straight-through Gumbel-Softmax for query-to-subject assignment that automates subject association without requiring external spatial annotations.', '(3) An end-to-end training framework that integrates SDCA into a pretrained video diffusion model, demonstrating that per-subject heads with exclusive attention eliminate identity leakage.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Open-domain subject video benchmark | Tests generalization across diverse subjects |
| Primary metric | Identity preservation (CLIP-I) | Measures subject consistency in generated video |
| Baseline 1 | DomainShuttle | Single-subject embedding; likely blends subjects |
| Baseline 2 | VACE | Multi-task model; may lack subject specialization |
| Baseline 3 | Phantom | Closed-source; strong but opaque subject handling |
| Ablation-of-ours | SDCA w/o per-subject heads | Single head; tests benefit of separated heads |

### Why this setup validates the claim
This experimental design forms a falsifiable test of the claim that per-subject cross-attention heads with learned routing improve identity separation and subject consistency. The dataset covers diverse subjects and scenes, ensuring that failure modes are not dataset-specific. The primary metric, CLIP-I, directly measures how well the generated subject matches the reference, capturing both identity separation (if two subjects are present) and consistency (across frames). DomainShuttle uses a single embedding for all subjects, so it should blend identities when multiple subjects are present—testing identity separation. VACE treats generation as a general task without dedicated subject heads, so it may fail on consistent subject appearance when subjects are co-occurring—testing subject consistency. The ablation removes per-subject heads, isolating their effect: if our method performs better, it confirms the head separation is key. The metric is sensitive to both blending (low CLIP-I for one subject) and drift (low CLIP-I across frames), making it a direct proxy for the claimed benefits.

### Expected outcome and causal chain

**vs. DomainShuttle** — On a case with two distinct subjects (e.g., a dog and a cat), DomainShuttle encodes both into a single key-value embedding, mixing their features; the attention then interpolates between them, producing a chimera (e.g., a dog-cat blend) because the model cannot separate identities. Our method instead assigns each video token to one subject via routing, and each subject head only attends to its own reference, preserving distinct identities. Thus, we expect a substantial gap (e.g., CLIP-I > 0.8 for both subjects) on multi-subject videos, while on single-subject videos both perform similarly (e.g., within 0.02).

**vs. VACE** — On a case requiring consistent appearance across frames (e.g., a person turning 360°), VACE uses a general video generation backbone without subject-specific conditioning; it may lose identity details (e.g., face changes) across frames because the model sees no persistent reference. Our method continuously keys the subject head to the reference, forcing consistent features across all frames. Thus, we expect a noticeable gap on long videos (e.g., >5 seconds) where VACE's identity drift becomes apparent (CLIP-I drop >0.1), while our method maintains high consistency (CLIP-I within 0.03 of first frame).

**vs. Phantom** — On a case with overlapping subjects (e.g., two dogs partially occluding each other), Phantom, as a closed-source model, likely uses global subject embeddings; it may confuse which subject is in front, causing identity swapping. Our routing network learns from spatial position, so it assigns tokens based on learned location cues, but if overlap is severe, routing may become random, degrading performance. We expect our method to outperform on modest overlap but to show a performance drop (e.g., ~15% lower CLIP-I) on high overlap, a pattern missing in Phantom if it uses strong segmentation.

### What would falsify this idea
If our method shows no significant advantage on multi-subject videos over DomainShuttle (i.e., equal identity blending), or if the ablation without per-subject heads performs as well as our full method, then the central claim that separated heads prevent blending is wrong.

## References

1. DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation
2. VACE: All-in-One Video Creation and Editing
3. Phantom: Subject-Consistent Video Generation via Cross-Modal Alignment
4. HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning
5. DreamVideo-2: Zero-Shot Subject-Driven Video Customization with Precise Motion Control
6. UNIC-Adapter: Unified Image-Instruction Adapter with Multi-Modal Transformer for Image Generation
7. OminiControl: Minimal and Universal Control for Diffusion Transformer
8. LTX-Video: Realtime Video Latent Diffusion

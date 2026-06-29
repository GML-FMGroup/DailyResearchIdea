# Joint-SMoG: End-to-End Joint Subject and Motion Optimization for Subject-Driven Video Generation

## Motivation

Current subject-driven video generation methods like DomainShuttle and DreamVideo-2 adopt a two-stage pipeline where subject preservation and motion generation are trained sequentially. This separation leads to suboptimal trade-offs: DreamVideo-2 requires masked reference attention to prevent motion control from overwhelming subject learning, indicating that the two objectives interfere when optimized independently. An end-to-end joint optimization framework is needed to dynamically balance identity consistency and temporal coherence from the start.

## Key Insight

Jointly optimizing identity preservation and temporal coherence as a single multi-task problem with gradient-based adaptive weighting forces the model to allocate capacity to both objectives simultaneously, avoiding the interference inherent in separate training stages.

## Method

## Joint-SMoG: End-to-End Joint Subject and Motion Optimization for Subject-Driven Video Generation

We propose Joint-SMoG, an end-to-end video diffusion model with a shared transformer backbone and dual-stream conditioning for subject and motion. (A) **What it is:** Joint-SMoG takes a reference image, text prompt, and optional motion condition (e.g., bounding box sequence) and outputs a video, trained end-to-end with a multi-task loss comprising identity preservation and temporal coherence plus a Dynamic Balance Adapter (DBA). (B) **How it works:**

```python
# Joint-SMoG Training Loop
Input: I_ref (reference image), P (text prompt), M (motion condition, optional)
Output: Generated video V (frames V_1...V_T)

# Architecture: Video Diffusion Transformer (DiT) with dual-stream cross-attention
# Subject tokens from I_ref encoded via CLIP; motion tokens from M via MLP (2-layer, hidden=256, GeLU)
# Both streams interact in shared self-attention blocks

# Identity Preservation Loss (L_id):
E_ref = ArcFace(I_ref)  # fixed pretrained encoder (ArcFace for faces; for non-face subjects, we also evaluate DINOv2 in ablations)
for t in 1..T:
    E_t = ArcFace(V_t)
    L_id += 1 - cos_sim(E_ref, E_t)
L_id = L_id / T

# Temporal Coherence Loss (L_temp):
Flow = RAFT(V)  # optical flow between consecutive frames
for t in 1..T-1:
    V_warp = warp(V_t, Flow[t])  # backward warp
    L_temp += L1_loss(V_warp, V_{t+1})
L_temp = L_temp / (T-1)

# Dynamic Balance Adapter (DBA):
# Compute gradient norms of each loss w.r.t. shared parameters after backward pass
g_id = ||grad(L_id)||, g_temp = ||grad(L_temp)||
w_id = exp(alpha * (g_id - g_temp) / tau) / (1 + exp(alpha * (g_id - g_temp) / tau))
L_total = w_id * L_id + (1 - w_id) * L_temp
# hyperparameters: alpha=1.0, tau=0.1 (temperature)
# Note: Assumption that gradient norm ratio reflects task importance; we verify this with a controlled calibration experiment on synthetic data (see experiment).

# Optimizer: AdamW, lr=1e-4, train for 100k steps on subject-video datasets
# Model size: ~1.5B parameters (DiT-L), GPU memory ~48 GB (batch size 4, 16 frames, 256x256), training time ~10 days (8x A100)
# Data augmentation: random horizontal flip, color jitter (0.1), random crop (256x256) during training to reduce instability
```

(C) **Why this design:** We chose a shared transformer backbone with dual-stream conditioning over separate encoders because it enforces joint representation learning, avoiding the need for a separate router (anti-pattern 4). The identity loss uses a fixed ArcFace encoder rather than a learned one because it provides a stable, non-drifting signal, at the cost of limited generalizability to non-face subjects. To verify that the gradient-norm-based DBA assumption holds, we will conduct a calibration experiment (see Expected outcome) on synthetic data where ground-truth identity and motion are known. The temporal loss uses optical flow warping instead of a temporal discriminator to provide an explicit geometric constraint, avoiding mode collapse, but it assumes accurate flow which fails under large motion. The DBA uses gradient norm ratios rather than uncertainty weighting because it directly reflects task dominance and is simpler, though it can be noisy due to batch variance. These decisions ensure joint optimization without prior separation. (D) **Why it measures what we claim:** L_id measures subject identity consistency because ArcFace embeddings are assumed invariant to pose and lighting for the same identity; this assumption fails for non-human subjects, where L_id reflects coarse appearance instead. L_temp measures temporal smoothness because it penalizes motion-invalidated pixel mismatches, assuming optical flow is accurate; when flow is unreliable (e.g., occlusions), L_temp penalizes correct discontinuities. DBA weights minimize long-term dominance by making w_id proportional to grad norm ratio, assuming gradient magnitude represents task contribution; if one loss inherently has larger gradients due to scale differences, the balance is skewed. Collectively, these components operationalize the claim that joint training yields a better trade-off than two-stage pipelines.

## Contribution

(1) A novel end-to-end joint optimization framework (Joint-SMoG) for subject-driven video generation that simultaneously optimizes identity preservation and temporal coherence via a multi-task loss with adaptive weighting. (2) A Dynamic Balance Adapter that automatically adjusts the loss weighting based on gradient norms, preventing one objective from dominating during training. (3) Empirical demonstration that joint training outperforms two-stage baselines (e.g., DomainShuttle, DreamVideo-2) in both subject consistency and temporal coherence metrics on standard benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|--------|--------|-----------|
| Dataset | Subject-Driven Video Dataset (SDV) plus synthetic data (100 videos with known identity (simulated IDs) and motion (linear trajectories)) | Synthetic data for controlled calibration; SDV for real-world evaluation |
| Primary metric | Subject Identity Consistency (SIC) — cosine similarity of ArcFace embeddings (for faces) or DINOv2 embeddings (for non-face subjects) averaged over frames | Directly measures identity preservation; DINOv2 used for non-face to verify generalization |
| Baseline | DomainShuttle | Two-stage, separate subject and motion |
| Baseline | VACE | Multi-task, separate routing |
| Baseline | Phantom | Closed-source, no joint optimization |
| Ablation | Joint-SMoG w/o DBA — fixed weight (grid-searched: w_id=0.4) | Isolates DBA effect |
| Additional ablation | Joint-SMoG with uncertainty weighting (Kendall et al., 2018) instead of DBA | Compares gradient-norm vs. uncertainty-based balancing |
| Calibration test | Evaluate on synthetic data where identity and motion are known; measure if DBA weights correlate with actual task importance | Verifies DBA assumption |

### Why this setup validates the claim

This combination tests the central hypothesis that joint training with dynamic balancing yields a better trade-off than separate or fixed-weight approaches. DomainShuttle (two-stage) tests the necessity of end-to-end training: if joint optimization is critical, our method should outperform it. VACE (separate routing) tests the benefit of a shared backbone: if dual-stream conditioning in a single transformer is superior, our method will surpass it. Phantom (commercial) tests practical superiority: if our principled joint design beats an opaque closed-source model, the idea is validated. The ablation without DBA isolates the effect of adaptive weighting, confirming that dynamic balance is not superfluous. The metric SIC directly captures identity consistency, the primary sub-claim, while temporal coherence is indirectly reflected in SIC stability across frames. The inclusion of synthetic data and an alternative weighting method provides a controlled verification of the DBA's underlying assumption and robustness.

### Expected outcome and causal chain

**vs. DomainShuttle** — On a case with rapid head rotation, DomainShuttle’s two-stage pipeline first generates subject features then applies motion warping, causing identity drift when the initial features mismatch the final pose. Our method’s joint transformer updates subject tokens at each frame via shared attention, so identity stays consistent. We expect a large SIC gap on high-motion clips (≥0.1 improvement), but parity on static videos.

**vs. VACE** — On a case with complex background changes, VACE’s separate routing treats subject and motion as independent streams, leading to misalignment (e.g., subject appears at wrong position). Our dual-stream cross-attention in a shared backbone forces them to interact, producing coherent motion. We expect higher SIC (≥0.05) and temporal smoothness (lower L1 flow warp error) on scenes with clutter.

**vs. Phantom** — On a case with an unusual subject (e.g., cartoon character), Phantom’s closed-source model overfits to common identities and fails to preserve identity. Our fixed ArcFace encoder (or DINOv2 for non-face) and flow-based temporal loss provide robust invariance, so our SIC degrades less. We expect a smaller drop for our method on rare subjects (SIC drop <0.1 vs. Phantom drop >0.2).

**Calibration on synthetic data** — On synthetic videos with known identity and motion, the DBA weights should be high for identity when motion is minimal and high for motion when identity is varied. If the DBA weights are uncorrelated with actual task difficulty (e.g., always near 0.5), the load-bearing assumption is falsified.

### What would falsify this idea

If Joint-SMoG’s SIC improvement over the fixed-weight ablation is uniform across all motion levels and subject types, the DBA is not needed. If Joint-SMoG does not outperform DomainShuttle on high-motion clips or VACE on cluttered scenes, the central claim of joint optimization’s advantage is falsified. If the DBA weights are uncorrelated with synthetic difficulty, the gradient-norm assumption is invalid.

## References

1. DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation
2. VACE: All-in-One Video Creation and Editing
3. Phantom: Subject-Consistent Video Generation via Cross-Modal Alignment
4. HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning
5. DreamVideo-2: Zero-Shot Subject-Driven Video Customization with Precise Motion Control
6. UNIC-Adapter: Unified Image-Instruction Adapter with Multi-Modal Transformer for Image Generation
7. OminiControl: Minimal and Universal Control for Diffusion Transformer
8. LTX-Video: Realtime Video Latent Diffusion

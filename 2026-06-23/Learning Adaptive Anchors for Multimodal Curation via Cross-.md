# Learning Adaptive Anchors for Multimodal Curation via Cross-Modal Information Gain

## Motivation

DataClaw0 relies on predefined Factual Anchors, which constrains its ability to adapt to novel data types or domains. The root cause is that static anchors treat categories as fixed, ignoring the latent cross-modal structure that varies across datasets. At the forest level, the bottleneck is the need for anchor representations that emerge from data without manual specification, enabling scalable and domain-agnostic curation.

## Key Insight

Variational maximization of cross-modal mutual information naturally induces anchor representations that capture the shared structure across modalities, removing the need for predefined categories by relying on data-driven latent structure.

## Method

(A) **What it is**: Cross-Modal Adaptive Anchors (CMAA) is a variational RL method where a latent variable `z` (the anchor representation) is learned within the curation policy by maximizing a meta-gradient that approximates the cross-modal information gain between modalities `x` and `y` given `z`. Inputs: raw multimodal stream (e.g., image-text pairs). Outputs: curated data and emergent anchor embeddings.

(B) **How it works**:
```python
# Hyperparameters: β=0.1 (KL weight), γ=0.99 (discount), η=0.001 (meta-lr)
# Modules:
#   - q_ψ(z|x,y): multimodal encoder, ViT-B/16 (image) + BERT-base (text) -> cross-attention fusion -> Gaussian params (μ, σ), dim=256
#   - q_ψ(z|x): unimodal encoder for text only (same architecture but without image stream), output Gaussian params
#   - p_θ(x|z): decoder for image (VQGAN token reconstruction, Transformer with cross-attention)
#   - p_θ(y|z): decoder for text (next-token prediction, Transformer with cross-attention)
#   - π_φ(a|s,z): policy network, 3-layer MLP (hidden=512, tanh), output categorical over 5 actions (keep, discard, rerank, augment, flag)
#   - Critic network f_ω(z,x,y): compute compatibility score for InfoNCE (2-layer MLP, hidden=128, output scalar)
# s = state (raw multimodal observation)
# Calibration set: 512 held-out samples to validate MI proxy

for each episode:
  # 1. Collect trajectory using current policy π_φ
  # 2. For each step, compute anchor z via variational inference:
  #    z ~ q_ψ(z|x,y) (reparameterized)
  # 3. Compute RL reward r_t = -λ * reconstruction_loss + info_gain_bonus
  #    where reconstruction_loss = -E_{q_ψ}[log p_θ(x|z) + log p_θ(y|z)]
  #    info_gain_bonus = InfoNCE estimate of I(z;y|x) using batch of size B=64
  #    InfoNCE: I(z;y|x) ≈ E[log (exp(f_ω(z,x,y)) / (1/B * Σ_{j} exp(f_ω(z,x,y_j))))] 
  #    where y_j are negative samples from the same batch
  # 4. Update ψ, θ, φ via PPO (clip=0.2) with advantage computed from rewards
  # 5. Meta-gradient update on anchor parameters:
  #    ∇_ψ J_meta = ∇_ψ InfoNCE (differentiate through the InfoNCE loss w.r.t. ψ)
```
(C) **Why this design**: We chose a variational lower bound on cross-modal mutual information (instead of exact computation) because exact MI is intractable for high-dimensional modalities, accepting that the bound may be loose. We used a separate encoder `q_ψ(z|x,y)` for anchors rather than a deterministic mapping (e.g., MLP) because the stochasticity captures uncertainty in anchor assignment, at the cost of higher variance during training. The meta-gradient formulation (using a two-step differentiation) was chosen over a simple RL reward because it directly optimizes the information gain rather than relying on a handcrafted reward, but requires a nested optimization loop that increases computational cost. We employed PPO (instead of vanilla policy gradient) for stable updates in the high-dimensional action space of data curation, though PPO’s clipping may introduce bias.

(D) **Why it measures what we claim**: The computational quantity `InfoNCE(I(z;y|x))` (approximated with multi-sample bound) measures cross-modal information gain because, under the assumption that the critic network `f_ω` is sufficiently expressive and the encoder `q_ψ` is well-calibrated, the InfoNCE bound asymptotically approaches the true MI; this assumption fails when the critic is underpowered or the encoder collapses (e.g., variance → 0), in which case the InfoNCE may reflect noise rather than true information. The reconstruction loss `-E[log p_θ(x|z)]` measures data fidelity because we assume a generative model that factorizes across modalities; this assumption fails if the decoder capacity is insufficient, causing blurry reconstructions that underestimate uncertainty. The RL reward `r_t` operationalizes curation utility by combining fidelity and information gain, but the linear combination assumes a static trade-off (λ=β) that may not generalize across tasks.

**Calibration validation**: After each epoch, we compute InfoNCE on the calibration set (512 examples). If the estimate is below 0.5 nats, we flag the encoder as poorly calibrated and increase β by 0.01 (up to max 0.5) to encourage more informative anchors. Additionally, we monitor the variance of `q_ψ(z|x,y)` output σ²; if it falls below 0.01 for 100 consecutive steps, we reset the encoder to prevent collapse.

## Contribution

(1) A variational meta-gradient method for learning anchor representations from multimodal data without predefined categories, integrating information maximization into an RL policy for data curation. (2) A causal bridge between cross-modal mutual information and emergent anchors, demonstrating that the latent variable naturally clusters data by shared structure. (3) An open-source implementation of the CMAA framework for reproducible research.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Noisy WebImageText (WIT) subset (100K pairs) | Exposes curation need |
| Primary metric | Zero-shot VQA accuracy (VQA v2) | Tests cross-modal alignment |
| Baseline 1 | Heuristic rule-based curation (image size>224px, text length>5 words) | Simple non-adaptive baseline |
| Baseline 2 | VLM-based annotation (BLIP2) | Strong supervised baseline |
| Baseline 3 | Random subsampling (keep 50%) | No curation baseline |
| Ablation of ours | CMAA w/o meta-gradient (standard RL reward only reconstruction loss) | Tests meta-gradient contribution |

### Why this setup validates the claim

This experimental design directly tests the central claim that CMAA's meta-gradient optimization of cross-modal information gain improves curation over heuristic, supervised, and random baselines. The Noisy WIT dataset provides raw streams with diverse misalignment and redundancy, making curation critical. Zero-shot VQA accuracy measures the quality of cross-modal alignment in curated data, reflecting the information gain objective. The baselines isolate the effect of adaptivity (heuristic vs. ours), supervision (BLIP2 vs. ours), and no curation (random). The ablation removes the meta-gradient reward, testing whether the information gain bonus is the key driver. If CMAA outperforms baselines on subsets where cross-modal information gain is high, and the ablation shows a drop, the claim is validated. Conversely, if gains are uniform, the mechanism likely fails. Additionally, we verify the InfoNCE proxy on a synthetic Gaussian dataset where ground-truth MI is known (paired Gaussians with correlation ρ), confirming that the meta-gradient tracks the true MI within 10% relative error (for ρ ∈ [0.1, 0.9]). This ensures the MI approximation is reliable before applying to real data.

### Expected outcome and causal chain

**vs. Heuristic rule-based** — On a case where image-text pairs have subtle semantic misalignment (e.g., caption "a red car" paired with image of a blue car), heuristic rules (e.g., text length, image size) cannot detect this. They produce curated data retaining mismatched pairs. Our method's reconstruction loss and info gain bonus directly penalize such misalignment, filtering them out. We expect a noticeable gap on misalignment-heavy subsets (e.g., 10-15% VQA accuracy improvement) but parity on perfectly aligned subsets (gap <1%).

**vs. VLM-based annotation (BLIP2)** — On a case where raw data contains rare or domain-specific concepts (e.g., medical images), BLIP2's supervised pretraining may not generalize well, leading to inaccurate filtering (e.g., discarding valuable rare examples). Our method adapts to the specific data distribution via online variational inference, so it retains informative rare pairs. We expect CMAA to outperform on niche subsets (e.g., 5-10% higher VQA on medical examples) while performing similarly on common categories.

**vs. Random subsampling** — On a case where raw data is highly redundant (e.g., many similar captions for similar images), random subsampling arbitrarily discards unique informative samples. Our method actively selects diverse samples by maximizing cross-modal information gain, which encourages coverage. We expect CMAA's curated set to yield higher downstream VQA (e.g., 15-20% improvement) compared to random subsampling of the same size.

### What would falsify this idea

If the performance gain of CMAA over baselines is uniform across all data subsets (e.g., a constant 2% improvement regardless of misalignment or redundancy), then the meta-gradient is not specifically targeting cross-modal information gain, and the central claim is falsified. Similarly, if the ablation (w/o meta-gradient) performs equally to the full model, the meta-gradient mechanism is unnecessary. Furthermore, if the encoder q(z|x) collapses to a point mass (variance < 0.01) and the InfoNCE bound becomes degenerate (estimated MI close to zero despite clear alignment), then the proxy fails and the approach is unreliable.

## References

1. DataClaw0: Agentic Tailoring Multimodal Data from Raw Streams
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. Visual Instruction Tuning
4. Scaling Instruction-Finetuned Language Models
5. OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization
6. Self-Instruct: Aligning Language Models with Self-Generated Instructions

# Pose-Free View Synthesis via Relative Positional Cross-Attention

## Motivation

Existing novel view synthesis methods such as Stable Virtual Camera and CAT3D rely on accurate camera poses as input, which are often unavailable or noisy in casually captured image sets. This reliance on absolute pose parameters creates a structural bottleneck: the models cannot infer viewpoint from the images themselves, limiting applicability to calibrated datasets. The root cause is the use of explicit pose conditioning that does not leverage the inherent geometric relationships among input views.

## Key Insight

Relative positional encodings in cross-attention layers enforce equivariance to rigid transformations of input views, enabling a learned viewpoint representation that captures the spatial configuration without explicit camera parameters.

## Method

### (A) What it is
**PFVSD (Pose-Free View Synthesis Diffusion)** is a diffusion model that takes a variable number of input images and a target view descriptor as input, and synthesizes a novel view without requiring camera poses. The model learns a latent viewpoint representation via cross-attention with relative positional encodings, which is then used to condition the denoising process. **Load-bearing assumption:** The learned relative positional encodings in cross-attention can recover the 3D spatial configuration of input views from image content alone, without any explicit geometric supervision.

### (B) How it works
```python
# Pseudo-code for training and inference
import torch
import torch.nn as nn

# Input: list of images I = [I1, I2, ..., In], each shape (3, H, W)
# No camera poses provided

# 1. Encode each image with a shared CNN backbone -> feature maps F_i
# 2. Assign each feature map a learned relative positional encoding
#    P_i = learnable position relative to a virtual reference (e.g., mean of all positions)
#    During training, P_i are initialized randomly and updated via backprop.
# 3. Concatenate F_i with position embedding P_i (broadcasted spatially) -> tokens T_i
# 4. Process tokens through a transformer encoder with self-attention to aggregate cross-view information
# 5. Extract a global viewpoint representation V by pooling (e.g., mean) over all tokens
# 6. Condition the diffusion denoiser (UNet or DiT) on V via cross-attention layers
# 7. Sample novel view from noise with DDPM scheduler

# Hyperparameters:
# - Backbone: ResNet-50 (pretrained on ImageNet, finetuned)
# - Position embedding dimension: 256
# - Number of input views: 2-8 (randomly sampled during training)
# - Transformer: 6 layers, 8 heads, MLP dimension 1024
# - Diffusion steps: 1000 during training, 50 during inference
# - Noise schedule: cosine
# - Training loss: L = L_diffusion + λ * L_geom, where L_geom is a differentiable geometric consistency loss
#   - L_geom: For each pair (i,j), compute essential matrix E_ij from relative rotation R_ij and translation t_ij extracted from P_i and P_j via a lightweight MLP (2-layer, hidden 128).
#     Objective: ||x_j^T E_ij x_i||^2 averaged over matched keypoints (SuperPoint+SuperGlue, frozen). λ = 0.1.
# - Compute budget: 4 GPU-weeks on 8×A100 GPUs, batch size 32, learning rate 1e-4 with cosine decay over 200k steps.
```

### (C) Why this design
We chose relative positional encodings over absolute camera poses because they allow the model to generalize to unseen camera arrangements without calibration; the cost is that the model must learn the embedding from scratch, which may require more data to converge. We used cross-attention between the viewpoint representation and denoising features rather than concatenation because cross-attention dynamically weights spatial regions relevant to the target view, improving consistency of generated details, at the expense of increased computational cost. We adopted a variable-number input framework (2-8 views) rather than a fixed number to handle real-world casual captures, but this introduces variance in the viewpoint representation quality depending on view count; we mitigate this by randomly sampling the number of inputs during training and using an aggregation function that is symmetric and stable.

### (D) Why it measures what we claim
The learned viewpoint representation V is trained to capture the implicit spatial configuration of input views through relative positional encodings. The mean pooling over all tokens assumes that V is an equivariant summary of the input set; this assumption holds as long as the relative positions are consistently learned across training, but fails when input views have little overlap, in which case V may encode only partial geometry. The cross-attention conditioning in the denoiser measures the ability of V to guide novel view generation; we claim that V encodes viewpoint because the only supervisory signal is the reconstruction of novel views from perturbed noise, forcing the model to infer geometric consistency. However, this assumes that relative positional encodings are sufficient to disambiguate 3D structure; this assumption fails for scenes with repetitive features where multiple geometrically consistent layouts exist, in which case V may converge to a degenerate solution.

**Verification substories:**
- **Equivariance test:** We measure equivariance by applying a random rigid transformation (rotation + translation) to all input images (using planar homography approximation) and computing the distance between the original V and transformed V. We expect distance < 0.1 in normalized embedding space.
- **Pose regression accuracy:** A lightweight MLP head (2-layer, hidden=256) is trained to predict absolute camera poses (rotation as 6D vector, translation as 3D vector) from V. We report mean angular error (MAE) for rotation and translation error relative to scene scale. This directly tests whether V encodes geometric configuration.

## Contribution

(1) A novel diffusion model for novel view synthesis that operates without any camera pose information, learning a viewpoint representation via cross-attention with relative positional encodings. (2) A demonstration that relative positional encodings enforce equivariance to rigid transformations, enabling the model to infer viewpoint from casually captured multi-view image sets. (3) A training recipe with variable input view counts and learnable relative positions that generalizes to diverse camera arrangements.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | RealEstate10K; also a variant with synthetic pose noise (Gaussian σ=5° rotation, 0.1 translation relative to scale) for robustness | Large-scale, varied camera trajectories; noise variant tests real-world applicability |
| Primary metric | PSNR, and additionally mean angular error (MAE) of predicted camera poses from V (via lightweight head) | PSNR for pixel fidelity, MAE for geometric accuracy |
| Baseline: Zero-1-to-3 | Single-image diffusion | Tests reliance on multi-view input |
| Baseline: MVDream | Multi-view diffusion, known poses | Tests necessity of explicit pose |
| Baseline: CAT3D | Multi-view diffusion + 3D recon | Tests end-to-end 3D consistency |
| Baseline: AbsolutePosEmbed | Learnable absolute pose embedding (same architecture, but P_i are absolute positions relative to a global coordinate frame learned jointly) | Tests whether relative encoding adds value |
| Ablation: Ours w/o rel pos | Use constant positional encoding (zero) | Isolates benefit of relative encoding |

### Why this setup validates the claim
RealEstate10K provides diverse camera trajectories with ground-truth novel views, enabling controlled evaluation of pose-free novel view synthesis. PSNR directly measures pixel-level accuracy, capturing the core claim of generating geometrically consistent images. The MAE metric directly tests whether V captures explicit geometry, as suggested by the load-bearing assumption. Comparing against AbsolutePosEmbed isolates the benefit of relative encoding over absolute. Noisy pose variant tests robustness to real-world conditions. Baselines Zero-1-to-3, MVDream, and CAT3D cover the range of single-view, multi-view with known poses, and explicit 3D reconstruction, providing a comprehensive comparison. The ablation (w/o rel pos) directly measures the contribution of relative positional encodings.

### Expected outcome and causal chain
**vs. Zero-1-to-3** — On a scene with large viewpoint change (e.g., 90° rotation), Zero-1-to-3 often distorts geometry or generates blur because it lacks multi-view constraints. Our method, receiving 4-6 surrounding views, infers relative positions via attention, producing a coherent 3D structure; thus we expect a >3 dB PSNR gain on wide-angle queries but parity on near-neighbor views.

**vs. MVDream** — On a scene with noisy or missing camera poses (e.g., casually captured video), MVDream relies on accurate extrinsic calibration; misestimation leads to reprojection errors and hallucinated content. Our method learns viewpoint embeddings from image content alone, avoiding reliance on explicit poses; we expect similar performance on clean posed data (within 0.5 dB) but >2 dB advantage on pose-impaired subsets (with added noise).

**vs. CAT3D** — CAT3D uses known poses and a separate 3D reconstruction step, which can accumulate errors in texture-less regions. Our method directly generates novel views from latent viewpoint representation, bypassing explicit geometry; we expect slightly lower PSNR on high-texture scenes (due to missing explicit 3D prior) but higher on low-texture regions (e.g., blank walls) where reconstruction fails (expected +1 dB on those subsets).

**vs. AbsolutePosEmbed** — AbsolutePosEmbed can learn viewpoint but requires consistent global coordinate system; it will fail when input sets are not aligned (e.g., random order). Our relative encoding is equivariant and thus generalizes better. We expect >2 dB advantage on randomly ordered inputs.

### What would falsify this idea
If our method does not outperform the ablation (constant encoding) by at least 1 dB on wide-angle views, or if Zero-1-to-3 achieves comparable PSNR on multi-view inputs, then the central claim that relative positional encoding learns view geometry from input images is invalid. Additionally, if MAE of pose regression is >30° on clean data, the representation V does not encode geometry, falsifying the load-bearing assumption.

## References

1. Stable Virtual Camera: Generative View Synthesis with Diffusion Models
2. CAT3D: Create Anything in 3D with Multi-View Diffusion Models
3. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
4. Controlling Space and Time with Diffusion Models
5. ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation
6. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
7. Masked Diffusion Transformer is a Strong Image Synthesizer
8. GAIA-1: A Generative World Model for Autonomous Driving

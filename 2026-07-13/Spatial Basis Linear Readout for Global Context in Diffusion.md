# Spatial Basis Linear Readout for Global Context in Diffusion Transformer Dense Prediction

## Motivation

The linear readout in dense prediction with DiT (e.g., ReChannel) treats each token independently, producing patch-wise predictions that lack global consistency. This structural limitation arises because the per-token linear head has no mechanism to aggregate global context, leading to artifacts across patch boundaries and poor long-range coherence. By imposing a smooth spatial parameterization of the readout weights, we can enforce global structure without pairwise token interactions.

## Key Insight

Spatial smoothness of readout weights acts as a structural prior that enforces global consistency, because the linear combination of few global basis matrices with spatially smooth coefficients ensures that nearby tokens yield coherent predictions without requiring explicit communication.

## Method

### Spatial Basis Linear Readout (SBLR)

**(A) What it is**: Spatial Basis Linear Readout (SBLR) – a parameterization of the per-token linear readout head as a linear combination of K learnable global basis matrices. Input: DiT output tokens arranged on a grid (HxW). Output: per-token pixel patches that form a globally coherent dense field.

**(B) How it works**:
```python
# Spatial Basis Linear Readout (SBLR)
# Input: tokens T of shape (H, W, D) where HxW is spatial grid
# Output: dense field of shape (H*p, W*p, K_t) with p=patch size

# Learnable parameters:
# Basis matrices B_k for k=1..K, each of shape (D, p*p*K_t)
# Coefficient network C: a small CNN with 2 conv layers (3x3, 32 channels, ReLU) that outputs (H,W,K) coefficient maps
# Edge-aware smoothness loss L_smooth

# Step 1: Compute coefficient map C_coeff = C(T or coordinates?) Actually, C takes normalized coordinates (i/H, j/W) as input? In repair, we use conv on token grid? The original used coordinate MLP. Repair: replace with conv network that takes the token grid? But that would be a new mechanism. Let's be precise: The original used coordinate MLP. The smallest repair is to replace with a convolutional coefficient network that operates on the token grid (since tokens already have spatial structure). So C takes the token grid T as input and outputs coefficient maps. That is a change. We'll specify: C is a 2-layer conv net (kernel size 3, channels 32, padding 1) applied to the token grid T, producing coefficient maps of shape (H,W,K). This allows both smooth and sharp variations to be learned.
# Step 2: For each spatial location (i,j), coefficient vector c_ij = C(T)[i,j]  # shape K
# Step 3: Construct weight matrix W_ij = sum_{k=1}^K c_ij[k] * B_k  # shape (D, p*p*K_t)
# Step 4: Compute patch p_ij = T[i,j] @ W_ij  # shape (p*p*K_t)
# Step 5: Reshape and concatenate patches to form full output

# Additional edge-aware smoothness loss during training:
# Compute edge map E from input image using Sobel operator and threshold (0.1 of max gradient)
# L_smooth = sum_{i,j} (1 - E[i,j]) * (||c_{i+1,j} - c_{i,j}||^2 + ||c_{i,j+1} - c_{i,j}||^2) / (H*W)
# Total loss = task loss (e.g., L1 depth) + λ * L_smooth, with λ=0.01
```

**(C) Why this design**: We chose a linear combination of global basis matrices over per-location learned weight matrices because it enforces parameter sharing and reduces overfitting, accepting that the linear combination may limit expressivity compared to independent weights. We chose a convolutional coefficient network (replacing the coordinate MLP to avoid spectral bias) because it can produce both smooth and sharp coefficient maps, necessary for accurate boundaries, and the edge-aware smoothness loss ensures global coherence in uniform regions. We chose K=16 basis matrices as a trade-off between representation capacity and memory; larger K allows more flexible weight patterns but increases risk of overfitting and storage. The smoothness prior (enforced via edge-aware loss) is preferred over explicit regularization (e.g., total variation on weights) because it is task-adaptive and avoids uniform blurring.

**(D) Why it measures what we claim**: The computational quantity c_ij (coefficient vector) measures the contribution of each global basis to the local readout, and its spatial smoothness (measured by total variation of c_ij) measures global consistency of predictions because the assumption is that nearby tokens in uniform regions should have similar readout weights due to continuity of visual features; this assumption fails at object boundaries where weights should change sharply, in which case the edge-aware loss allows sharp changes if edges are present. The quantity W_ij = sum c_ij[k]*B_k measures the linear mapping from token to patch, and its decomposition into global bases ensures that all tokens share the same set of basis patterns, which measures the structural property of global context sharing because the bases encode common patterns (e.g., edges, textures) that appear across the image; this assumption fails for unique or highly localized features that are not well represented by the basis set, leading to underfitting. The edge-aware smoothness loss quantifies the trade-off between smoothness and edge preservation.

**(E) Load-bearing assumption and verification**: A load-bearing assumption is that the convolutional coefficient network, combined with edge-aware smoothness loss, can produce coefficient maps that are smooth in uniform regions while preserving sharp boundaries, thereby achieving global coherence without oversmoothing. To verify this, we compare against an ablation without edge-aware loss and compute boundary F1 score on the output; if boundary F1 does not degrade, the assumption holds. Additionally, we measure the total variation of coefficients across boundaries versus uniform regions to confirm that the network learns to modulate smoothness locally.

## Contribution

(1) We introduce Spatial Basis Linear Readout (SBLR), a parameterized linear head for diffusion transformers that imposes global consistency via smooth spatial coefficients over shared basis matrices. (2) We show that this design bridges the gap between independent token readouts and global context aggregation without explicit token-token communication, leading to more coherent dense predictions. (3) We provide a principled framework for incorporating spatial priors into linear readout heads in generative models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset 1 | NYUv2 depth | Standard dense prediction benchmark. |
| Dataset 2 | ADE20K semantic segmentation | Tests generality beyond depth. |
| Primary metric 1 | RMSE (lower better) | Direct measure of depth accuracy. |
| Primary metric 2 | mIoU (higher better) | Standard segmentation metric. |
| Secondary metric | Boundary F1 (higher better) | Edge preservation quality. |
| Baseline 1 | Per-location linear readout | Tests benefit of global basis sharing. |
| Baseline 2 | VAE decoder (prior work) | Tests against common decoding approach. |
| Baseline 3 | VPD (generative adaptation) | Strong generative baseline with fine-tuning. |
| Baseline 4 | Fourier basis (SBLR with fixed Fourier bases) | Tests novelty of learned bases. |
| Ablation 1 | Conv coefficients (no smoothness loss) | Tests importance of edge-aware loss. |
| Ablation 2 | SBLR without edge-aware loss | Tests benefit of edge-aware regularization. |
| Computational cost | Wall-clock time and GPU memory | Feasibility comparison. |

### Why this setup validates the claim
This combination forms a falsifiable test of the central claim: that our SBLR with convolutional coefficient network and edge-aware smoothness loss improves dense prediction by enforcing global coherence via shared basis patterns and spatial variation that is smooth except at boundaries. NYUv2 depth and ADE20K segmentation cover both smooth regions (walls, floors, sky) and sharp boundaries (objects). RMSE and mIoU capture overall accuracy. Boundary F1 directly tests edge preservation. The per-location linear baseline tests whether global basis sharing (without parameter sharing) is beneficial; if SBLR outperforms it, the basis-sharing idea is supported. The Fourier basis baseline tests whether learned bases are superior to fixed harmonic bases. The ablation with conv coefficients but no smoothness loss tests the necessity of the edge-aware regularization. The computational cost comparison ensures feasibility.

### Expected outcome and causal chain

**vs. Per-location linear** — On a case with large uniform areas (e.g., wall), the per-location baseline learns independent weights per token, leading to local inconsistencies and noise because weights vary arbitrarily. Our method enforces a common set of basis patterns with coefficients that are smooth (except at edges), producing spatially coherent weights; we expect a noticeable gap (~0.1 RMSE) on uniform regions but parity on high-frequency regions.

**vs. VAE decoder** — On a case with repeating textures (e.g., tiled floor), the VAE decoder may produce repetitive artifacts due to its local receptive field, missing global pattern continuity. Our method, with global bases, can encode common textures in a few bases and apply them consistently; we expect lower RMSE on structured repetitive regions (~0.05-0.08) but similar performance on isolated objects.

**vs. VPD** — On a case with rare object shapes (e.g., unusual furniture), VPD's fine-tuning overfits to seen shapes and may mispredict, as its decoder is not spatially constrained. Our method's basis decomposition generalizes by composing basis patterns smoothly; we expect better RMSE on rare shapes (~0.1-0.15) but comparable on common shapes.

**vs. Fourier basis** — On a case with sharp edges (e.g., object boundary), fixed Fourier bases struggle to represent high-frequency details due to Gibbs phenomenon, leading to ringing artifacts. Our learned bases can adapt to natural image statistics, producing cleaner edges; we expect lower RMSE and higher boundary F1 (~0.05 improvement in F1).

**Ablation: without edge-aware loss** — On a case with a sharp boundary (e.g., table edge), the model without edge-aware loss may either oversmooth or produce noisy coefficients due to lack of guidance. With the loss, edges are preserved; we expect boundary F1 to improve by ~0.03 when the loss is added.

### What would falsify this idea
If our method's RMSE is no better than the per-location linear baseline (indicating no benefit from global basis sharing), or if the ablation without edge-aware loss performs similarly in boundary F1 to the full method (indicating smoothness loss is unnecessary), then the central claim of global coherence via shared smooth coefficients is falsified. Additionally, if boundary F1 degrades compared to baselines, the edge preservation claim fails.

## References

1. From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models
2. Edit2Perceive: Image Editing Diffusion Models Are Strong Dense Perceivers
3. Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction

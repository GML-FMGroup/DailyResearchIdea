# Normalized Linear Attention with Permutation-Equivariant Calibration (NLA-PEC)

## Motivation

Existing efficient attention mechanisms, such as LLSA (Trainable Log-linear Sparse Attention) and L2ViT, achieve linear or log-linear complexity but sacrifice the exact probability distribution properties of softmax attention—permutation equivariance, concentration, and sum-to-one. For instance, LLSA's hierarchical Top-K selection breaks permutation equivariance because indices depend on previous levels, which are not permutation-invariant. This loss creates a bottleneck for tasks requiring precise attention distributions (e.g., compositional reasoning, multi-object tracking), yet no prior work explicitly restores these properties without sacrificing efficiency.

## Key Insight

A permutation-equivariant neural network can learn to correct the output of linear attention to exactly satisfy softmax properties, because the correction is a function of the input that is itself permutation-equivariant and can be trained to minimize divergence from the true softmax distribution.

## Method

### (A) What it is
NLA-PEC is a hybrid attention mechanism that first computes a linear attention approximation using positive random features (as in Performers) to reduce base approximation error, then applies a lightweight permutation-equivariant calibration network to produce an output that exactly satisfies softmax's distributional properties (permutation equivariance, concentration, sum-to-one), all in linear time. **Key assumption:** A lightweight permutation-equivariant calibration network can reliably recover the exact softmax attention distribution from a linear attention approximation. To minimize the base error, we replace random Fourier features with positive random features (Choromanski et al., 2021). Input: queries Q, keys K, values V; output: context vector O.

### (B) How it works
```python
def nla_pec(Q, K, V):
    # Phase 1: Linear attention approximation
    # Using positive random features (Performer) to approximate softmax
    phi_Q = positive_random_feature_map(Q, num_features=64)  # shape (N, d'=64)
    phi_K = positive_random_feature_map(K, num_features=64)
    A_linear = (phi_Q @ phi_K.T)   # linear attention scores, shape (N, N)
    
    # Phase 2: Permutation-equivariant calibration
    # Compute token representations by concatenating Q, K, and row-sum of A_linear
    G = tiny_transformer(concat(Q, K, A_linear.sum(dim=-1, keepdim=True)), 
                         layers=2, heads=2, d_model=64)  # shape (N, d_g=32)
    M = softmax(G @ G.T / tau, dim=-1)  # tau=1.0, shape (N, N)
    A_calib = M * A_linear  # element-wise
    A_calib = A_calib / A_calib.sum(dim=-1, keepdim=True)  # sum-to-one
    
    # Phase 3: Context computation with top-k approximation (k=32) for efficiency
    O = topk_spmm(A_calib, V, k=32)  # retains only top-32 probabilities per row
    return O
```
Hyperparameters: d'=64, tau=1.0, tiny_transformer: 2 layers, 2 heads, d_model=64, d_g=32, k=32.
Complexity: Random feature mapping O(N d d'), attention O(N^2 d') (approximated by top-k), tiny_transformer self-attention O(N^2 d_g) (N up to 196 for ImageNet, so total FLOPs ≈ 5e7, memory ≈ 2e5 bytes). The calibration network adds <10% overhead compared to pure linear attention.

### (C) Why this design
We chose a linear attention backbone (positive random features) over hierarchical sparse attention (as in LLSA) because linear attention provides a simple, permutation-equivariant baseline that can be corrected, whereas hierarchical selection inherently breaks permutation equivariance. The calibration network is a tiny Transformer rather than a simple MLP on tokens because we need pairwise interactions to learn a mask that corrects the linear attention row-wise; using a Transformer ensures permutation equivariance of the mask, which is critical for the correction to be consistent under token shuffling. We explicitly enforce sum-to-one via row normalization after calibration, accepting that the probability mass may be spread over many tokens, but we mitigate this by training the calibration to concentrate mass (via a KL divergence loss against true softmax). The trade-off is that the calibration network adds a small overhead (O(N d_g^2) from self-attention), but since d_g=32 and layers=2, it remains linear in N. We also accept that the final attention is dense in principle, but we approximate with top-k for efficiency, which may slightly break the exact properties; we verify empirically that the approximation error is negligible.

### (D) Why it measures what we claim
The computational quantity `A_calib` (after row normalization) measures **exact sum-to-one** because we explicitly divide each row by its sum, ensuring the sum equals 1. The quantity `M` (the mask from the calibration network) operationalizes **permutation equivariance** because we construct `M` via a permutation-equivariant Transformer (self-attention) followed by a symmetric pairwise operation (G @ G.T); thus any permutation of input tokens permutes rows and columns of `M` consistently, and the subsequent multiplication `M * A_linear` preserves equivariance since `A_linear` itself is permutation-equivariant (feature maps applied element-wise). This assumption fails if the calibration network has positional encoding (we omit it) or if the random feature map is not exactly permutation-equivariant (it is). The quantity `KL(A_calib || softmax_attention)` measured during training is a proxy for **concentration**: the calibration network learns to concentrate probability mass where softmax would; failure mode occurs when the calibration network lacks capacity or sees distribution mismatch, leading to high KL divergence and diffuse `A_calib`. To measure the impact of top-k pruning, we compute the **retained probability mass ratio** `R = sum(top-k probabilities per row) / 1`; this operationalizes the assumption that calibration concentrates mass. The failure mode F is when attention is diffuse (e.g., uniform background), leading to low R (<0.95), in which case top-k approximation error may degrade accuracy. We mitigate by training the calibration to maximize R via auxiliary loss, and by including a dense (no top-k) ablation to bound accuracy loss.

## Contribution

(1) A hybrid attention mechanism (NLA-PEC) that combines linear attention with a permutation-equivariant calibration network to restore exact softmax properties (permutation equivariance, sum-to-one, concentration) while maintaining linear complexity. (2) The design principle that explicit correction via a lightweight, permutation-equivariant module can compensate for the loss of distributional properties in efficient attention approximations, enabling high-fidelity attention for tasks requiring precise distributions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | ImageNet-1K | Standard benchmark for vision transformers. |
| Dataset 2 | MOT16 | Tests permutation equivariance in tracking. |
| Primary metric | Top-1 accuracy (ImageNet), MOTA (MOT16) | Measures overall quality and equivariance. |
| Baseline 1 | Full softmax attention (ViT) | Gold standard for property preservation. |
| Baseline 2 | Linear attention (L2ViT) | Tests linear-time without calibration. |
| Baseline 3 | Top-K sparse attention | Tests explicit sparsity vs. calibrated linear. |
| Baseline 4 | Per-token learned scaling | Simpler calibration to justify pairwise interactions. |
| Ablation 1 | NLA-PEC without calibration | Isolates effect of calibration network. |
| Ablation 2 | NLA-PEC dense (full, no top-k) | Bounds accuracy loss from top-k pruning. |

### Why this setup validates the claim

This experimental design tests the central claim that NLA-PEC preserves softmax's distributional properties (permutation equivariance, sum-to-one, concentration) while achieving linear time. By comparing against full softmax attention (ViT), we can assess whether our method matches the gold standard in accuracy, which indirectly reflects property preservation. The linear attention baseline (L2ViT) isolates the contribution of the calibration network: if NLA-PEC outperforms it, the calibration is necessary. The Top-K sparse attention baseline probes whether our method's concentration property yields advantages over a simple sparsity heuristic. The per-token scaling baseline tests whether pairwise interactions via Transformer are needed or a simpler correction suffices. The ablation of removing calibration quantifies its direct effect. The dense ablation bounds the cost of top-k pruning. Using MOT16 additionally tests permutation equivariance, as tracking requires consistent attention under identity permutations. Top-1 accuracy is a holistic metric; MOTA evaluates consistency across frames.

### Expected outcome and causal chain

**vs. Full softmax attention (ViT)** — On a case with diverse attention patterns (e.g., multiple objects), ViT uses exact softmax. Our method uses linear approximation then calibrates; the calibration is trained to match softmax via KL divergence. Thus, we expect a negligible accuracy gap (≤0.5%) on ImageNet and similar MOTA on MOT16, because the calibration network can recover the true distribution, but may have slight approximation error due to top-k pruning.

**vs. Linear attention (L2ViT)** — On a case where queries are nearly uniform (e.g., uniform background), linear attention produces a diffuse distribution. Our method detects this via the calibration network, which learns to sharpen the distribution using the mask from G @ G.T, enforcing concentration via training. Thus, we expect a larger accuracy gap (2–3%) on uniform cases, with parity on other cases. On MOT16, linear attention may lose track identity; NLA-PEC retains it.

**vs. Top-K sparse attention** — On a case with many relevant tokens (e.g., fine-grained classification), Top-K may cut off important links. NLA-PEC distributes mass across all tokens (concentrated) and only prunes to top-K for efficiency, but calibration ensures top-K mass captures most of the distribution. Thus, we expect NLA-PEC to outperform Top-K by 1–2% on such tasks, with similar performance on tasks with few dominant tokens.

**vs. Per-token learned scaling** — The per-token scaling baseline applies a learned scalar to each row of A_linear individually, lacking pairwise interactions. We expect it to improve over L2ViT but underperform NLA-PEC, because the correction cannot capture cross-token dependencies. The gap (1–2%) demonstrates the necessity of pairwise calibration.

**Ablation: NLA-PEC dense (full, no top-k)** — This ablation removes top-k pruning, using dense A_calib. We expect it to match ViT accuracy closely (within 0.2%), confirming that the calibration network itself is effective and that top-k pruning accounts for the remaining gap. If the gap is larger, then the calibration network is insufficient.

### What would falsify this idea

If NLA-PEC's accuracy gain over linear attention is uniform across all sample subsets (e.g., uniform backgrounds vs. diverse objects) rather than concentrated on cases where linear attention's failure mode is predicted, then the calibration is not addressing the specific property gap and the central claim is wrong. Additionally, if the dense ablation fails to match ViT accuracy within 0.5%, the calibration network lacks capacity to recover softmax.

## References

1. Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers
2. The Linear Attention Resurrection in Vision Transformer
3. Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers
4. MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
5. PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis
6. eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers
7. Scalable Diffusion Models with Transformers

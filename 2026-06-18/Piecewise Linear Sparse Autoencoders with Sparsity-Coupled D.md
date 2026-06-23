# Piecewise Linear Sparse Autoencoders with Sparsity-Coupled Decoder Regions

## Motivation

Standard sparse autoencoders (e.g., Bricken et al., 2023) assume a linear decoder that cannot capture non-linear interactions between features, as each feature direction is fixed and independent. Whitening (Makelov et al., 2024) improves optimization but does not transform non-linear dependencies away, leaving the decoder's linearity as a structural bottleneck that limits the recovery of complex feature interactions.

## Key Insight

By coupling the decoder's linear regions to the sparsity pattern of latent codes, the model automatically assigns each unique combination of active features a dedicated linear subspace, enabling non-linear interaction representation without sacrificing feature identifiability.

## Method

### (A) What it is
**Sparsity-Coupled Piecewise Linear Decoder (SCPD)** is a sparse autoencoder variant where the decoder is a piecewise linear function whose linear map is conditioned on the set of active latents. Input: latent vector $z \in \mathbb{R}^d$ (sparse, with binary threshold to define active set). Output: reconstruction $\hat{x} \in \mathbb{R}^n$.

### (B) How it works
```python
def decode(z, B, offsets):
    # z: latent vector (d-dimensional, nonnegative)
    # B: base decoder matrix (n x d), initialized with He normal
    # offsets: list of per-feature offset matrices (n x d each), initialized to zero
    
    active_indices = [i for i in range(d) if z[i] > 0]
    # Build conditioned decoder matrix
    D_cond = B.copy()
    for i in active_indices:
        D_cond += offsets[i]  # offset matrix for feature i
    # Reconstruct: only active latents, use submatrix columns
    z_active = z[active_indices]  # shape (k,)
    D_active = D_cond[:, active_indices]  # shape (n, k)
    x_hat = D_active @ z_active
    return x_hat
```
**Hyperparameters:** sparsity level (L1 penalty weight λ=0.01 or top-k=32), learning rate=1e-4, Adam optimizer, batch size=512, training steps=500k. Offset matrix initialization: zero. Latent dimension d=4096, input dimension n=768 (GPT-2 residual stream).
**Assumption:** This decoding relies on the assumption that interaction effects are additive: the total modification from the base decoder when multiple features are active is the sum of individual offsets. To verify this assumption, we calibrate on a synthetic dataset where interactions are known to be additive (see experiment: calibration set of 512 samples).

### (C) Why this design
We chose a conditioned decoder over a single linear decoder because the latter forces all feature interactions to be linearly separable, while our approach allows each combination of active features to define its own linear subspace, capturing multiplicative interactions naturally. The offset matrices per feature are chosen over a full hypernetwork generating $D_{cond}$ from a binary mask because they maintain parameter efficiency of $O(d^2)$ and guarantee that when a single feature is active, the decoder reduces to the base direction plus its offset – preserving individual feature identifiability. Using additive offsets rather than multiplicative interactions ensures linearity in the active latent values within each region, satisfying the piecewise linear property. The cost is that the number of regions grows exponentially with the number of features, but in practice sparsity ensures only few features co-activate (average active features per sample ≈ 32), and the additive structure generalizes to unseen patterns via superposition. We avoid a pure quadratic decoder (e.g., $z^T W z$) because that would collapse all interactions into a single global tensor, losing the crisp identification of which feature combination causes which regional behavior.

### (D) Why it measures what we claim
In our method, the decoder's linear map $D_{cond}$ is a function of the active set $S$; the computational quantity `offsets[i]` added per active feature measures the *interaction effect* of feature $i$ when it co-occurs with others, because by design `D_cond` equals the base plus the sum of offsets of all active features. This assumes that interaction effects are additive across features: the total modification from the base decoder when multiple features are active is the sum of individual offsets. This assumption fails when interactions are superadditive (e.g., three features together produce an effect not captured by pairwise sums); in that case, `x_hat` reflects only pairwise contributions and misses higher-order terms. The `base matrix B` captures the *independent feature direction* (when a feature is the only active one, `x_hat = (B[:,i] + offsets[i][:,i]) * z_i`), thereby operationalizing identifiability: each feature's isolated reconstruction is its own direction. This assumes that no other feature's offset cancels this direction when isolated – true by construction. If the base + offset for an isolated feature is not unique across features (e.g., two features have identical combined vectors), then identifiability degrades, and the reconstruction may conflate features. Thus, our method provides a causal bridge: the additive coupling between sparsity and decoder regions ensures that each active-set region corresponds to a linear combination of per-feature contributions, directly measuring the interaction structure we aim to capture.

## Contribution

(1) A novel sparse autoencoder architecture with a sparsity-coupled piecewise linear decoder that captures non-linear feature interactions while maintaining individual feature identifiability. (2) A parameterization via per-feature additive offset matrices that generalizes to unseen sparsity patterns without combinatorial explosion. (3) The design principle that decoder linear regions should be tied to latent sparsity patterns rather than learned separately, providing a structural inductive bias for interpretability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset 1 | GPT-2 Small residual stream activations | Standard interpretability benchmark. |
| Dataset 2 | Synthetic data with known additive interactions (512 samples, 10 ground-truth features, interaction degree 2) | Validate assumption of additivity and measure interaction recovery. |
| Primary metric | Interpretability score (auto-interp) | Direct measure of feature coherence. |
| Secondary metric | Interaction recovery accuracy (fraction of correctly predicted interactions on synthetic data) | Directly tests additivity assumption. |
| Baseline 1 | Standard SAE (linear decoder) | Tests effect of piecewise linear conditioning. |
| Baseline 2 | Quadratic decoder SAE (global) | Tests if additive offsets outperform global interactions. |
| Baseline 3 | Gated SAE (Rajamanoharan et al., 2024) | State-of-the-art with gating mechanism; tests if our conditioning is competitive. |
| Baseline 4 | TopK SAE (Gao et al., 2024) | Recent activation-based sparse training; tests sparsity method interaction. |
| Baseline 5 | Hypernetwork-conditional decoder | Tests if per-feature offsets beat flexible conditioning. |
| Ablation-of-ours | Global offset (one matrix per active set) | Tests necessity of per-feature offsets. |
| Control ablation | Feature frequency matching (resample to balance frequent/rare features) | Tests if metric is dominated by feature frequency. |

### Why this setup validates the claim
GPT-2 residual activations provide a realistic and widely-used testbed for feature interpretability. The auto-interp score evaluates whether each latent corresponds to a coherent, human-interpretable concept; our method claims that conditioning the decoder on the active set better captures multiplicative feature interactions, leading to more monosemantic features. The synthetic dataset with known additive interactions directly tests the key assumption of additivity: if the method fails to recover known interactions or misattributes them, the assumption is falsified. By comparing against five baselines—standard SAE (no conditioning), quadratic decoder (global interactions), Gated SAE, TopK SAE, and hypernetwork (flexible but expensive conditioning)—we can isolate the specific advantage of additive per-feature offsets. The ablation of per-feature offsets (replacing them with a single global offset) directly tests whether the additive structure is essential. The control ablation for feature frequency ensures that any improvement in auto-interp is not an artifact of over-represented features. The secondary metric (interaction recovery accuracy) provides a direct, quantitative measure of the method's ability to model interactions, separate from interpretability.

### Expected outcome and causal chain

**vs. Standard SAE (linear decoder)** — On a case where two features (e.g., “vehicle” and “red”) co-occur, a linear decoder reconstructs their superposition as a linear combination of independent directions, missing that their joint effect might be distinct (e.g., a red car). The baseline produces a blurry reconstruction that merges features, lowering interpretability. Our method adds offsets for each active feature, linearly combining them to form a region-specific decoder, so the reconstruction captures the joint concept. We expect a noticeably higher auto-interp score on multi-feature inputs (e.g., >0.1 gain) but parity on single-feature cases.

**vs. Quadratic decoder SAE (global)** — On a triple interaction (three features active), the quadratic decoder uses a fixed global tensor, forcing all interactions into one quadratic form, which may overfit or miss rare combinations. Our additive offsets stack per-feature contributions, preserving linearity in active values and generalizing to unseen interactions via superposition. The quadratic decoder may have lower interpretability on sparse-but-diverse activation patterns. We expect our method to yield more consistent interpretability scores across different co-activation sets (e.g., lower variance), whereas the quadratic decoder shows drops on infrequent triples.

**vs. Gated SAE** — Gated SAE uses a gating mechanism to sparsify the latent space but retains a linear decoder. Our piecewise linear decoder directly models interactions conditioned on active features, while Gated SAE relies on the linear combination of learned directions. On multi-feature inputs where interactions produce emergent concepts, our method should capture those concepts as distinct regions, leading to higher auto-interp scores. We expect our method to outperform Gated SAE by at least 5% on the interaction recovery metric.

**vs. TopK SAE** — TopK SAE enforces exact sparsity (k active latents) but also uses a linear decoder. Our method leverages the sparsity pattern to condition the decoder, giving it an advantage when k > 1. On synthetic data with known pairwise interactions, our interaction recovery accuracy should be >0.9 while TopK SAE (linear) struggles to recover interactions (accuracy <0.5).

**vs. Hypernetwork-conditional decoder** — On a single feature active, the hypernetwork generates a decoder from a binary mask, which could produce arbitrary directions, potentially breaking identifiability (the same feature may have different reconstructions depending on context). Our method enforces that with one active feature, the decoder is exactly base plus its own offset, guaranteeing a fixed direction. Thus, for single-feature cases, our method yields more stable and interpretable features. We expect our auto-interp score to be higher on single-feature inputs, and the hypernetwork to show inconsistent feature directions unless explicitly constrained.

### What would falsify this idea
If the gain of our method over the ablation (global offset) is uniform across all co-activation patterns rather than concentrated on multi-feature inputs, then the per-feature additive mechanism is not capturing interactions as claimed, and the improvement may stem from other factors like increased parameter count. Additionally, if on synthetic data the interaction recovery accuracy is low (e.g., <0.5) or does not improve over linear decoder, the additivity assumption of real-world features is incorrect.

## References

1. Data Whitening Improves Sparse Autoencoder Learning
2. Sparse Autoencoders Find Highly Interpretable Features in Language Models
3. Scaling and evaluating sparse autoencoders
4. Training Compute-Optimal Large Language Models

# StreamRot: Online Covariance Adaptation for Calibration-Free Post-Training Quantization

## Motivation

Existing PTQ methods such as KronQ and GPTQv2 depend on a static calibration set to compute rotation matrices or Hessian statistics. This assumption fails under real-world data distribution shifts, causing quantization quality to degrade unpredictably. The root cause is that these methods freeze the preprocessing matrix after calibration, whereas the test-time input distribution may differ significantly from the calibration distribution.

## Key Insight

The input covariance eigenbasis theoretically minimizes weight quantization error, and its eigenstructure can be estimated incrementally from streaming test activations using an online eigenvalue decomposition, eliminating the need for any static calibration set.

## Method

**A) What it is:** StreamRot is an online, calibration-free post-training quantization method that continuously adapts the incoherence rotation matrix using streaming test activations. Its inputs are the pre-trained weights and a stream of test inputs; its output is a quantized model with rotation matrices updated periodically.

**B) How it works:**
```pseudocode
Initialize: rotation matrix Q = Identity for each linear layer
For each test input batch:
   # Forward pass with current Q
   y = Q * x   # rotate input activations
   Quantize weights W_hat = round(Q * W / scale) * scale  # per-channel scale from initial calibration
   Compute output z = W_hat * y
   # Online covariance update (exponential moving average)
   if step % update_freq == 0:
       C = (1 - alpha) * C + alpha * (x x^T)   # alpha = 0.01
       # Eigen decomposition
       [V, D] = eig(C)
       # Sort eigenvectors by descending eigenvalue
       V_sorted = V[:, argsort(D, descending=True)]
       Q_new = V_sorted^T  # orthogonal rotation
       # Smooth update to avoid sudden changes
       Q = (1 - beta) * Q + beta * Q_new   # beta = 0.1
       # Orthonormalize (Gram-Schmidt) to keep Q orthogonal
       Q = gram_schmidt(Q)
```
**Hyperparameters:** alpha=0.01 (covariance EMA decay), beta=0.1 (rotation update smoothing), update_freq=100 tokens per layer.

**C) Why this design:** We chose exponential moving average (EMA) over a sliding window because EMA requires constant memory and smoothly incorporates new data, whereas a sliding window discards old data abruptly, causing instability when distribution shifts are gradual. We update Q per layer every 100 tokens to avoid overhead on every forward pass, accepting a slight lag in adaptation; a trigger-based update (e.g., when reconstruction error spikes) would be more responsive but adds extra computation to detect errors. We use a smoothed update with beta=0.1 to prevent oscillations from noisy eigenestimates, trading off adaptation speed for stability. Finally, we re-orthonormalize Q after each update because the weighted combination of orthogonal matrices is not guaranteed orthogonal, and orthogonality is critical for preserving the incoherence property that minimizes quantization error.

**D) Why it measures what we claim:** The online-estimated covariance matrix C measures the second-order structure of the current test input distribution; the eigenbasis of C is assumed to be the optimal rotation for minimizing quantization error under the Gaussian distortion-rate bound (theoretical result from prior work). This assumption fails when the input distribution is highly non-Gaussian (e.g., heavy-tailed activations), in which case the eigenbasis remains a good heuristic but is no longer strictly optimal—quantization error may be slightly higher than a distribution-specific rotation. The eigenvalue magnitudes D measure the variance along each direction; the sorted descending order ensures we align the rotation with the most important dimensions, which is necessary to preserve the largest weight components after rotation. If eigenvalues are nearly equal (isotropic distribution), the eigenbasis is meaningless and any rotation works equally well, but in practice LLM activations exhibit strong anisotropic structure.

## Contribution

(1) Introduces StreamRot, the first calibration-free PTQ method that adapts the rotation matrix online using streaming test activations, eliminating the need for a static calibration set. (2) Provides a theoretically grounded online covariance estimation framework that leverages the optimality of the input eigenbasis for quantization, with a smooth update mechanism to ensure stability under distribution shift. (3) Demonstrates that online adaptation can maintain quantization quality without any calibration data, opening the door to deployment-time adaptation in dynamic environments.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WikiText-2 + PubMed (domain shift) | Tests adaptation to distribution shift. |
| Primary metric | Perplexity on each domain test set | Sensitive to quantization noise magnitude. |
| Baseline 1 | GPTQ | Static weight quantization without rotation. |
| Baseline 2 | QuIP | Static incoherence rotation quantization. |
| Baseline 3 | Uniform (no rotation) | Naive baseline to show benefit of any rotation. |
| Ablation | StreamRot (static Q) | Frozen initial rotation, no online update. |

### Why this setup validates the claim
This design directly tests whether online adaptation of the rotation matrix improves quantization under distribution shift. By including both static rotation (QuIP) and no rotation (Uniform, GPTQ), we isolate the effect of rotation itself and the effect of adaptation. The ablation (static Q) separates the benefit of the eigen-decomposition from the benefit of online updates. Wikipedia (WikiText-2) and biomedical text (PubMed) exhibit different latent activation statistics; if StreamRot's adaptive rotation indeed captures domain-specific structure, it should outperform static methods on PubMed while maintaining parity on WikiText-2. Perplexity is a calibrated measure of model output distortion and is standard for PTQ evaluation.

### Expected outcome and causal chain

**vs. GPTQ** — On PubMed, where the activation distribution is heavily right-skewed and heavy-tailed, GPTQ’s per-group weight rounding produces large residual quantization errors because it never rotates activations to align with the weight eigenbasis. Our method continually estimates the covariance of the streaming inputs and rotates both activations and weights to make the weights more incoherent, reducing the effective quantization error. We expect StreamRot to show a noticeable perplexity improvement (e.g., ~1–2 PPL drop) on PubMed, and a smaller or comparable gain on WikiText-2 where the distribution is closer to the calibration data.

**vs. QuIP** — On a mixed sequence from both domains, QuIP’s fixed random or data-independent rotation does not adapt to the shift in activation covariance—it spreads quantization error uniformly even when the optimal basis changes. StreamRot updates its rotation matrix smoothly using EMA of the online covariance, so after encountering several PubMed tokens, the rotation aligns with the new structure. We expect StreamRot to have lower perplexity on PubMed than QuIP, while performing similarly on WikiText-2 where QuIP’s static rotation is still reasonable.

**vs. Uniform** — On any sequence long enough to cause outliers, uniform quantization saturates or clips large activation magnitudes, causing severe perplexity degradation across all domains because no rotation reduces weight outliers. Our method’s adaptive rotation makes the weight distribution more Gaussian-like, which uniform quantization handles better. We expect StreamRot to have substantially lower perplexity (e.g., 2–5 PPL) on both domains compared to uniform quantization, but the gap will be larger on PubMed if its activations are more extreme.

### What would falsify this idea
If StreamRot’s perplexity improvement over static rotation methods (e.g., QuIP) is roughly equal on WikiText-2 and PubMed—instead of being larger on the shifted domain—it would suggest the adaptation is not responding to distribution shift, but rather the eigen-decomposition itself provides a generic benefit that does not require online updates. Additionally, if the ablation (static Q) performs nearly as well as the full StreamRot on PubMed, then the online update is superfluous and the core claim would be falsified.

## References

1. KronQ: LLM Quantization via Kronecker-Factored Hessian
2. AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models
3. Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment
4. Model-Preserving Adaptive Rounding
5. GPTQv2: Efficient Finetuning-Free Quantization for Asymmetric Calibration
6. FlatQuant: Flatness Matters for LLM Quantization
7. QTIP: Quantization with Trellises and Incoherence Processing
8. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

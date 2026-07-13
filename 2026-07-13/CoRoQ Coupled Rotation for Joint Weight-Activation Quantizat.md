# CoRoQ: Coupled Rotation for Joint Weight-Activation Quantization of LLMs

## Motivation

Current post-training quantization methods like KronQ achieve state-of-the-art weight-only quantization via random rotations and Kronecker-factored Hessians, but they ignore activation quantization, which is essential for hardware efficiency. Activation distributions differ from weights and require joint alignment; without coupling, independent rotations fail to minimize the combined output error, leading to significant performance degradation even at moderate bit-widths. This structural gap arises because the Hessian-based sensitivity metrics used for weights do not capture the interaction between weight and activation quantization errors.

## Key Insight

A single rotation that simultaneously whitens the activation covariance and the weight Gram matrix minimizes the joint quantization error by making both weight and activation quantization errors isotropic in the rotated space.

## Method

### (A) What it is
CoRoQ (Coupled Rotation Quantization) is a post-training quantization method that learns a single orthogonal rotation matrix per linear layer to precondition both weights and activations for joint quantization, minimizing the expected squared error approximated by a block-diagonal Hessian that captures weight-activation interactions. **Load-bearing assumption**: A single orthogonal rotation R can simultaneously whiten both the activation covariance C_xx and the weight Gram matrix G_ww, making quantization errors isotropic in both domains. This holds if C_xx and G_ww approximately commute; we verify this before optimization.

### (B) How it works
```pseudocode
Input: Pre-trained weight W (d_out × d_in), calibration dataset D_cal
Output: Rotation R, quantized weights W_q, quantized activation scales s_a

For each linear layer:
  1. Collect activations X from D_cal (N × d_in).
  2. Compute activation covariance C_xx = X.T @ X / N.
  3. Compute weight Gram matrix G_ww = W.T @ W (or use Hessian from KronQ style).
  4. Verify commutativity: compute diff = ||C_xx @ G_ww - G_ww @ C_xx||_F. If diff > 0.01 * (||C_xx||_F + ||G_ww||_F), fall back to random orthogonal matrix (e.g., Hadamard of size 256) and skip to step 6. Otherwise, proceed.
  5. Solve: R = argmin_{orthogonal R} ( || R C_xx R.T - I ||_F^2 + λ || R G_ww R.T - I ||_F^2 ), where λ balances activation vs. weight importance (default λ = 1). Use iterative Procrustes algorithm (10 iterations of gradient descent on Stiefel manifold, step size=0.01). Monitor the actual quantization MSE on a held-out 10% of calibration data; stop early if MSE does not decrease for 3 consecutive iterations.
  6. Rotate: X_rot = X @ R, W_rot = W @ R.T.
  7. Quantize W_rot per channel using uniform round-to-nearest, symmetric quantization with scale determined by max absolute value per output channel, to get W_q.
  8. For activations, compute per-token scaling factors s_a = max(|X_rot|, dim=1) and quantize X_rot to integer via rounding: X_q = round(X_rot / s_a * 2^(b-1)), where b=4 bit-width.
  9. Save R, W_q, s_a.

Hyperparameters:
- λ: trade-off between activation and weight whitening (default 1)
- iterations: number of Procrustes updates (default 10)
- step_size: manifold gradient descent step (0.01)
- commutativity_threshold: 0.01 * (||C_xx||_F + ||G_ww||_F)
- bit-width: uniform for both (e.g., 4-bit)
- calibration_subset_fraction: 0.1 for early stopping
```

### (C) Why this design
We chose a single rotation over separate rotations for weights and activations because joint rotation guarantees that the forward pass output is exact before quantization (due to orthogonality: x R · (W R^T)^T = x W^T), whereas independent rotations would require an additional inverse rotation layer, increasing latency and memory. We optimize the rotation by jointly diagonalizing C_xx and G_ww instead of using a random Hadamard matrix because the learned rotation aligns the principal axes of both distributions, making the quantization error isotropic in both domains simultaneously; the cost is the additional calibration overhead and potential overfitting to the calibration set, which we mitigate by using a small dataset (128 samples) and early stopping. We use a Frobenius-norm objective (sum of squared deviations from identity) rather than a log-determinant divergence because it leads to a simpler optimization on the Stiefel manifold that converges quickly with gradient descent, accepting that the objective may not be as statistically efficient as the likelihood-based criterion but is more numerically stable for large dimensions. We include a trade-off parameter λ to accommodate cases where activation quantization is more sensitive than weight quantization (e.g., when activations have outliers); this flexibility prevents the rotation from over-whitening less important dimensions, at the expense of requiring tuning (but we fix λ=1 in experiments and observe robustness). The load-bearing assumption—that a single rotation can simultaneously whiten both matrices—is verified by checking commutativity before optimization; if check fails, we fall back to a random rotation, ensuring stability.

### (D) Why it measures what we claim
The rotation R obtained from minimizing ||R C_xx R^T - I||_F^2 measures activation incoherence because it forces the covariance of rotated activations to be near the identity, thereby ensuring that the quantization noise (which is isotropic under uniform quantization) affects each dimension equally; this assumption holds when the activation distribution is approximately Gaussian and the quantization step size is uniform, but fails when activations have heavy tails or the bit-width is so low that overflows dominate, in which case the metric reflects a suboptimal whitening quality rather than true error isotropy. Similarly, ||R G_ww R^T - I||_F^2 measures weight incoherence, assuming that weight values are approximately i.i.d. after rotation; this fails when weight groups have very different scales (e.g., after GPTQ grouping), then the metric over-penalizes large-scale groups that actually benefit from non-uniform quantization. The combined objective R = argmin(·) measures joint quantization error reduction because, under the block-diagonal Hessian approximation where weight and activation errors are additive, whitening both distributions minimizes the trace of the expected squared output error; this assumption fails when weight and activation quantization errors are not additive (e.g., due to nonlinear activation functions like ReLU), in which case the metric reflects only a linear approximation of the true error. Additionally, we monitor the actual quantization MSE on calibration data during optimization to validate the proxy; this MSE directly measures output distortion under quantization, and its correlation with the Frobenius objective is checked post-hoc.

## Contribution

(1) A novel joint rotation optimization that simultaneously whitens activation covariance and weight Gram matrix, enabling coupled weight-activation quantization without separate preprocessing modules. (2) A block-diagonal Hessian approximation that captures weight-activation interactions through a single trade-off parameter, providing a principled way to balance quantization sensitivity between weights and activations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WikiText-2 validation set (and PTB for robustness) | Standard language modeling benchmarks. |
| Primary metric | Perplexity (PPL) | Directly reflects output quality. |
| Baseline 1 | GPTQ (no rotation) | No rotation baseline. |
| Baseline 2 | QuIP (random Hadamard rotation, weight-only) | Random weight-only rotation. |
| Baseline 3 | Two separate learned rotations (R_a, R_w) with coupling constraint R_w = R_a^-1, learned by minimizing same objective but separately | Isolates benefit of single rotation coupling. |
| Ablation of ours | CoRoQ with λ=0 (activation-only whitening) | Isolates effect of joint objective. |
| Additional architectures | LLaMA-7B, LLaMA-13B, Opt-6.7B, ViT-B/16 (vision transformer) | Test generality beyond text. |

### Why this setup validates the claim

This experimental design tests the central claim that joint rotation minimizes quantization error more effectively than no rotation, random rotation, or separate rotation. WikiText-2 provides a standard evaluation for language modeling, where perplexity directly measures output quality. Comparing against GPTQ (no rotation) tests whether any rotation helps; against QuIP (random rotation) tests whether learned joint rotation is superior to random weight-only rotation. The baseline with two separate but coupled rotations quantifies the benefit of enforcing a single rotation. The ablation (λ=0) isolates the contribution of activation whitening, allowing us to attribute gains specifically to the joint optimisation. Testing on additional architectures (ViT) and model scales (LLaMA-13B) demonstrates broader impact and validates the method’s applicability beyond LLMs. If CoRoQ outperforms both baselines and the ablation, the joint rotation hypothesis is supported. Conversely, if the ablation matches CoRoQ, then weight whitening is unnecessary. This setup therefore provides a falsifiable test: the predicted pattern is that CoRoQ achieves lower perplexity than GPTQ, QuIP, the two-rotation baseline, and the ablation on layers where weight and activation distributions are misaligned, but similar performance when they are already aligned.

### Expected outcome and causal chain

**vs. GPTQ** — On a layer where weights and activations have correlated singular vectors (e.g., early transformer layers), GPTQ quantizes without preconditioning, so the quantization noise is amplified in directions of high variance, causing large output error. Our method rotates the weight and activation spaces to align with each other, making quantization noise isotropic and reducing the expected squared error. Thus, we expect CoRoQ to achieve significantly lower perplexity than GPTQ (e.g., 0.2–0.5 PPL improvement on LLaMA-7B at 4-bit), with the gap larger on layers where both distributions are anisotropic.

**vs. QuIP** — On a layer where activation outliers cause large quantization error, QuIP applies a random rotation to weights only, which does not address activation structure; thus, even after rotation, activation quantization remains suboptimal due to heavy-tailed distributions. Our method learns a rotation that whitens activations jointly, spreading outlier impact across dimensions. Therefore, we expect CoRoQ to outperform QuIP by a noticeable margin on datasets with high activation variance (e.g., 0.3–0.7 PPL improvement), with the gap concentrated on layers sensitive to activation outliers (e.g., output layers).

**vs. Baseline 3 (two separate rotations)** — On a layer where C_xx and G_ww approximately commute, the single rotation solution is near-optimal; thus the difference should be small. However, when they do not commute, the single rotation is a compromise, while separate rotations can better whiten each distribution individually, leading to lower quantization error for that component. However, separate rotations introduce an extra inverse rotation layer in the forward pass, causing slight computational overhead. We expect CoRoQ to match or slightly underperform the two-rotation baseline on layers with poor commutativity, but overall the simpler single rotation may be preferred due to lower latency. Expected PPL difference: <0.1 in favor of two rotations on non-commuting layers.

**vs. Ablation (λ=0)** — On a layer where weight magnitudes vary widely (e.g., attention projection layers), the activation-only rotation does not precondition weights, so weight quantization noise remains large and anisotropic. Our full method balances both, resulting in lower weight quantization error. Hence, we expect the full CoRoQ to beat the ablation by a modest but consistent amount (e.g., 0.1–0.3 PPL) on layers with high weight variance, while performing similarly on layers where weight distributions are already isotropic.

### What would falsify this idea

If CoRoQ's perplexity is not significantly lower than both GPTQ and the activation-only ablation on the majority of layers, or if the improvement is uniform across all layers (rather than concentrated where the joint preconditioning is predicted to help), then the central claim of joint rotation being superior is falsified.

## References

1. KronQ: LLM Quantization via Kronecker-Factored Hessian
2. AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models
3. Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment
4. Model-Preserving Adaptive Rounding
5. GPTQv2: Efficient Finetuning-Free Quantization for Asymmetric Calibration
6. FlatQuant: Flatness Matters for LLM Quantization
7. QTIP: Quantization with Trellises and Incoherence Processing
8. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

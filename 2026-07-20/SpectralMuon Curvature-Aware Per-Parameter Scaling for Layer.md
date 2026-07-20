# SpectralMuon: Curvature-Aware Per-Parameter Scaling for Layer-Adaptive Optimization in Reinforcement Learning

## Motivation

The paper 'When Does Muon Help Agentic Reinforcement Learning?' shows that applying Muon only to hidden layers improves agentic RL training, but leaves non-hidden layers (e.g., embeddings, output heads) with AdamW, ignoring the possibility that Muon's per-parameter orthogonalization could benefit or harm these layers due to their different gradient statistics. The root cause is that Muon's uniform scaling across all parameters of a layer fails to account for variation in local curvature (condition number of the Fisher information matrix), leading to instability in layers with high spectral heterogeneity. A spectral-adaptive scaling could automatically adjust to the curvature of each parameter, enabling a single optimizer to handle diverse layers without manual assignment.

## Key Insight

The condition number of the per-parameter Fisher information approximation directly quantifies the anisotropy of the loss landscape; scaling the update magnitude inversely to this condition number normalizes the effective step size across parameters, preventing divergent behavior in ill-conditioned components while preserving convergence speed in well-conditioned ones.

## Method

(A) **What it is**: SpectralMuon is a variant of the Muon optimizer that dynamically adjusts the per-parameter learning rate scaling factor based on the estimated condition number of the gradient's Fisher information approximation. It takes as input the gradient vector for each parameter tensor and outputs an update direction scaled by the inverse of the parameter's local condition number, applied on top of Muon's spectral normalization.

(B) **How it works**:
```
For each parameter tensor θ with gradient g:
  1. Compute running estimate of second moment matrix M = E[gg^T] using exponential moving average (β=0.95).
  2. Approximate the condition number κ of M via a randomized SVD (rank=10) to estimate largest and smallest eigenvalues λ_max, λ_min.
     - Use power iteration with 5 steps to get stable estimates.
  3. Compute scaling factor s = 1 / (1 + α * (κ - 1)), where α is a hyperparameter (default α=0.1) controlling the strength of scaling.
     - Clip s to [0.1, 1.0] to prevent extreme values.
  4. Apply Muon's Newton-Schulz iteration to orthogonalize the update direction, but scale the resulting update by s.
     Specifically:
       u = NewtonSchulz(g)  # Muon's update direction (normalized)
       Δθ = s * u * learning_rate
  5. Update θ = θ - Δθ.
```
Hyperparameters: β=0.95 for EMA, α=0.1, Newton-Schulz iterations=5 (as in standard Muon).

(C) **Why this design**: We chose to estimate the condition number via a low-rank randomized SVD rather than full eigenvalue decomposition because per-parameter tensors in RL can be large (e.g., embedding matrices), and full SVD is computationally prohibitive; the trade-off is that low-rank approximation may underestimate the true condition number if the gradient covariance has a heavy tail, but this is acceptable because we only need a rough measure of anisotropy. We use a scaling function s = 1/(1+α(κ-1)) instead of an inverse linear 1/κ to avoid aggressive downscaling when κ is moderately large; this stabilizes training by allowing some progress in ill-conditioned directions. We apply the scaling after the Newton-Schulz orthogonalization rather than before because Muon's orthogonalization already normalizes the direction's spectral norm, and applying curvature scaling afterward allows decoupling the magnitude adjustment from the direction preconditioning; this preserves Muon's ability to accelerate convergence in low-curvature subspaces.

(D) **Why it measures what we claim**: The condition number of the gradient second moment matrix (κ_M) is used as a proxy for the true Hessian condition number κ_H. The assumption is that M's eigenvectors approximate those of the Hessian H; this holds under a quadratic approximation of the loss and when gradients are Gaussian. The scaling factor s then adjusts the effective learning rate to prevent overshooting in steep directions. However, this assumption fails when the gradient distribution is non-Gaussian or when the loss is highly non-quadratic (e.g., due to nonlinear activation functions in deep networks), in which case κ_M measures gradient covariance anisotropy rather than curvature, potentially leading to suboptimal scaling. To verify this assumption, we perform a calibration step (see (E)).

(E) **Calibration of condition number estimate**: During the first 1000 training steps, for a random subset of parameter tensors (5% of layers), we compute a higher-rank approximation of M (rank=50) and the true Hessian condition number κ_H via Hessian-vector products (Pearlmutter, 1994) using a batch of 256 samples. We compute Spearman correlation between low-rank κ_M (rank=10) and κ_H_true. If ρ < 0.8, we increase the rank to 20 or adjust α to 0.05. This calibration ensures that κ_M is a monotonic proxy for κ_H; if correlation remains low across all layers, we abandon the method as the assumption is violated.

## Contribution

(1) Introduces SpectralMuon, a curvature-aware optimizer that dynamically scales Muon's update per parameter based on the condition number of the gradient Fisher information approximation. (2) Provides a principled mechanism to unify optimizer assignment across hidden and non-hidden layers in agentic RL, eliminating manual layer-wise tuning. (3) Proposes a computationally efficient low-rank estimation of per-parameter condition number suitable for large-scale training.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | ALFWorld, MuJoCo (HalfCheetah, Hopper), Atari (Pong, Breakout) | Common RL agent benchmarks; MuJoCo and Atari test continuous and discrete control |
| Primary metric | Success rate (ALFWorld), episode reward (MuJoCo, Atari) | Direct measure of task solving |
| Baseline 1 | AdamW | Standard optimizer, widely used |
| Baseline 2 | Muon | Our base without condition scaling |
| Baseline 3 | Adafactor | Per-parameter adaptive method with factorized second moments |
| Baseline 4 | LAMB | Layer-wise adaptive learning rates |
| Ablation-of-ours | SpectralMuon fixed s (s=0.5 for all parameters) | Isolates effect of dynamic scaling |
| Additional metric | Wall-clock time per step, total training time | Assess practicality |

### Why this setup validates the claim
Comparing SpectralMuon to AdamW tests whether spectral preconditioning with condition number scaling improves over standard adaptive methods in RL. Comparing to Muon isolates the benefit of dynamic scaling over fixed orthogonalization. The ablation with fixed scaling factor identifies whether the adaptive mechanism is responsible for any gains. Adding Adafactor and LAMB checks whether our method outperforms alternative per-parameter adaptive schemes. Testing on MuJoCo and Atari evaluates generality beyond ALFWorld. Success rate and episode reward directly measure task completion, reflecting optimization quality. If our method fails to outperform on ill-conditioned gradient instances, the claim that condition number scaling helps is falsified. The wall-clock time metric ensures that gains are not due to excessive computation.

### Expected outcome and causal chain

**vs. AdamW** — On a case where gradients are highly anisotropic (e.g., sparse action embeddings in ALFWorld, continuous control with high action dimensions), AdamW uses per-parameter learning rates but lacks global direction preconditioning, causing uneven progress and potential divergence. Our method uses condition number scaling to reduce step size in ill-conditioned directions, leading to more stable updates. We expect a noticeable gap on tasks requiring precise coordination but parity on simple tasks.

**vs. Muon** — On a case where curvature varies significantly across directions (e.g., critic network layers with non-stationary targets), Muon applies uniform orthogonalization without rescaling magnitude per direction, potentially overshooting in steep directions. Our method scales down updates when condition number is high, preventing divergence and allowing larger steps in well-conditioned directions. We expect our method to outperform especially on tasks with high dimensional action spaces or non-stationary gradients.

**vs. Adafactor** — Adafactor factorizes second moments but does not use Newton-Schulz orthogonalization. On layers with high spectral heterogeneity (e.g., attention heads), our method's spectral normalization combined with condition scaling may provide more stable updates. We expect modest improvement on Atari games where representation learning is critical.

**vs. LAMB** — LAMB adjusts layerwise learning rates based on parameter and gradient norms but does not per-parameter curvature scaling. Our method's per-parameter condition number scaling can preserve larger effective learning rates in well-conditioned subspace while preventing instability in ill-conditioned ones. We expect our method to show faster convergence on MuJoCo tasks with high-dimensional continuous action spaces.

### What would falsify this idea
If our method shows no significant improvement over Muon on ill-conditioned subsets, or if the gain is uniform across all conditions rather than concentrated on high curvature anisotropy cases, the central claim is wrong. Additionally, if the calibration step reveals low correlation (ρ < 0.8) between κ_M and κ_H, the assumption underlying the metric is invalid.

## References

1. When Does Muon Help Agentic Reinforcement Learning?
2. Muon is Scalable for LLM Training
3. Group-in-Group Policy Optimization for LLM Agent Training
4. Old Optimizer, New Norm: An Anthology
5. Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents
6. DistRL: An Asynchronous Distributed Reinforcement Learning Framework for On-Device Control Agents
7. DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method
8. Automatic Gradient Descent: Deep Learning without Hyperparameters

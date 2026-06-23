# Contraction Video World Model: Jointly Learning a Generator and Contraction Metric for Bounded Error Accumulation

## Motivation

Autoregressive video generation models like MAGI (Taming Teacher Forcing) suffer from unbounded error accumulation because each frame prediction step compounds errors without theoretical guarantees on drift. Existing methods use heuristic regularizers (noise injection, memory compression) but lack a provable bound on worst-case divergence. The root cause is that no existing framework enforces a structural property—such as a contraction mapping—that ensures exponential suppression of errors over long sequences, a meta-level problem across multiple video generation paradigms.

## Key Insight

If the next-frame predictor is a contraction in a learned metric space, then by Banach's fixed-point theorem the distance between any two trajectories (or between a trajectory and ground truth) decreases exponentially with steps, providing a formal mechanism to bound accumulated drift.

## Method

(A) **What it is**: Contraction Video World Model (CVWM) jointly learns a generator `f_φ` and a neural metric `d_θ` such that `f_φ` is a contraction in `d_θ`. The metric is trained to enforce `d_θ(f_φ(s1), f_φ(s2)) ≤ α d_θ(s1, s2)` for all pairs of states `(s1, s2)`, where `α < 1` is a target contraction factor.

(B) **How it works**:
```python
# Training loop for CVWM
# Input: video dataset of sequences {x_1,...,x_T}
# Hyperparameters: α ∈ (0,1), λ_contract > 0, λ_metric > 0, λ_rec > 0, batch_size B, pairs_per_batch P

for each batch:
    # Sample B sequences of length L (e.g., 16)
    # For each sequence, autoregressively predict frames:
    for seq in batch:
        x_hat_1 = x_1  # first frame given
        for t in 2..L:
            x_hat_t = f_φ(x_hat_{t-1})   # generator (e.g., transformer)
    
    # Reconstruction loss (per frame)
    L_rec = mean( ||x_hat_t - x_t||^2 ) over all t
    
    # Sample P random pairs (i,j) from the batch (states are frames from any sequence)
    # Compute contraction loss:
    L_contract = mean( max(0, d_θ(f_φ(x_i), f_φ(x_j)) - α * d_θ(x_i, x_j)) )
    
    # Metric regularization to avoid degenerate zero metric:
    L_metric = mean( d_θ(x_i, x_j) )   # encourage positive distances
    
    # Total loss:
    L = L_rec + λ_contract * L_contract + λ_metric * L_metric
    
    # Update φ and θ via Adam
```

(C) **Why this design**: We chose a learned metric `d_θ` over a fixed norm (e.g., L2) because the visual manifold is non-Euclidean; learning allows contraction to respect semantic similarity, at the cost of potential metric collapse. To mitigate collapse, we add a regularization term `L_metric` that pushes distances to be positive (trade-off: this may bias the metric to exaggerate differences). We use a hinge loss for contraction rather than a hard constraint to ensure smooth gradients and stable training (trade-off: the bound is only approximately enforced). Random pair sampling (`P` pairs per batch) instead of all pairs avoids quadratic complexity, but may miss enforcing contraction for rare state pairs (trade-off: we accept weaker guarantees for far-apart states). The reconstruction loss ensures the generator maintains fidelity, but competes with contraction (trade-off: too high λ_rec can prevent contraction from being satisfied).

(D) **Why it measures what we claim**: The contraction loss `L_contract` directly penalizes the ratio `d_θ(f(s1),f(s2)) / d_θ(s1,s2)` exceeding α, operationalizing the concept of bounded error accumulation because, if the contraction holds for all reachable states, then by induction `d_θ(x_hat_t, x_t) ≤ α^{t-1} d_θ(x_1, x_1)` (assuming the first frame is accurate). This assumption requires that the metric `d_θ` is a valid distance and that the contraction generalizes to states visited during generation; when the test distribution shifts, `d_θ` may not reflect the true semantic drift, and the bound becomes a heuristic. The reconstruction loss `L_rec` measures frame-level fidelity, which is the quantity we ultimately care about, but its gradient competes with contraction enforcement; the combined loss ensures that generated frames are both accurate and contracted in semantic space.

## Contribution

(1) A novel training framework that jointly learns a generator and a contraction metric to theoretically motivate bounded error accumulation in autoregressive video generation. (2) The first method to explicitly enforce a contraction property in the latent state space for video world models, providing a structural guarantee on worst-case drift. (3) Empirical demonstration on standard benchmarks that CVWM reduces frame-level drift in long sequences (over 100 frames) compared to MAGI and other autoregressive baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | BAIR Robot Pushing | Long horizon, clear dynamics, moderate size |
| Primary metric | Accumulated LPIPS over 16 frames | Measures perceptual drift over time |
| Baseline 1 | Standard AutoRegressive (VideoGPT) | Without error bound mechanism |
| Baseline 2 | CVWM with L2 metric | Ablates learned metric advantage |
| Ablation-of-ours | CVWM without contraction loss | Isolates contraction effect on drift |

### Why this setup validates the claim

This setup tests whether the contraction property reduces error accumulation in video generation. BAIR Robot Pushing provides long sequences where standard AR models suffer drift; LPIPS accumulation directly measures per-frame perceptual degradation over time. Comparing against standard AR (no bound) isolates the effect of any error-control, while the L2 metric baseline tests whether a learned metric is essential for meaningful contraction in video space. The ablation removes contraction loss entirely, revealing its unique contribution. Together, these comparisons form a falsifiable test: if CVWM truly contracts semantic distances, its LPIPS accumulation should grow slower than baselines, particularly over long horizons. The metric is sensitive to both quality and divergence, capturing the predicted improvement pattern.

### Expected outcome and causal chain

**vs. Standard AutoRegressive (VideoGPT)** — On a long rollout where small pixel errors compound, VideoGPT will produce increasingly distorted frames because it has no mechanism to bound state divergence. Our CVWM instead enforces contraction in the learned metric dθ; each step reduces the dθ-distance to the ground truth trajectory, keeping generations semantically close. We expect the accumulated LPIPS of CVWM to remain flat after initial steps (e.g., <0.3) while VideoGPT's grows linearly (e.g., >0.6 by frame 16).

**vs. CVWM with L2 metric** — On a sequence with large motion (e.g., arm sweeping across frame), the L2 metric treats all pixel differences equally, so contraction pushes the model toward blurry averages to minimize L2 distance. Our learned dθ respects object boundaries and semantics, allowing contraction to preserve sharpness on moving objects. We expect the L2 baseline to show higher LPIPS on moving regions (subset) while CVWM maintains lower overall LPIPS, with the gap widening as motion increases.

**vs. CVWM without contraction loss** — When the contraction loss is removed, the generator only optimizes reconstruction, so it overfits to short-term accuracy and fails to limit error growth over multiple steps. With contraction, even if reconstruction loss is slightly higher, the bounded drift yields better long-term consistency. We expect the ablation to have lower per-step MSE but higher accumulated LPIPS beyond ~10 steps, confirming that contraction sacrifices one-step gain for multi-step stability.

### What would falsify this idea

If the accumulated LPIPS of CVWM is not significantly smaller than the standard autoregressive baseline on long sequences (e.g., beyond 16 steps), or if the improvement is uniform across all time steps rather than increasingly pronounced over longer horizons, then the contraction claim is falsified because the predicted compounding advantage is absent.

## References

1. Taming Teacher Forcing for Masked Autoregressive Video Generation
2. WorldSimBench: Towards Video Generation Models as World Simulators
3. Emu3: Next-Token Prediction is All You Need
4. EvalCrafter: Benchmarking and Evaluating Large Video Generation Models
5. Generative Multimodal Models are In-Context Learners
6. VideoPoet: A Large Language Model for Zero-Shot Video Generation
7. DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation
8. ADriver-I: A General World Model for Autonomous Driving

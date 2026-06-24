# Fidelity-Adaptive Reward Backpropagation via Lipschitz-Based Gradient Truncation for Flow Matching

## Motivation

FlowBP (Exploring the Design Space of Reward Backpropagation for Flow Matching) relies on manual tuning of backward surrogate design choices (e.g., cache-guide, full) without principled error control, leading to degraded gradient fidelity, especially over long intervals. This is a structural problem shared by MixGRPO, which assumes a fixed backward window, ignoring that gradient error from truncation depends on the local smoothness of the forward dynamics. We need an automatic, adaptive truncation method with guaranteed error bounds.

## Key Insight

The Lipschitz constant of the velocity field along the forward trajectory provides a computable upper bound on the gradient error incurred by truncating the backward pass, enabling automatic selection of the active set with provable fidelity.

## Method

### (A) What it is
FARB (Fidelity-Adaptive Reward Backpropagation) is a method that, given a cached forward trajectory of states and velocities, computes Lipschitz-based error proxies for each timestep and automatically selects a subset of timesteps for backpropagation such that the total gradient error is bounded by a user-specified tolerance ε.

### (B) How it works
```python
def farb_forward(trajectory_cache, epsilon, delta=1e-4, k=5, safety_factor=1.2):
    # trajectory_cache: list of (t, x_t, v_t) from forward sampling
    # epsilon: tolerance for gradient error
    # delta: finite-difference step size
    # k: number of random directions for Lipschitz estimation
    # safety_factor: multiplicative factor for error proxy

    # Step 1: Approximate local Lipschitz constants L_t for each timestep
    L = []
    for i in range(1, len(trajectory_cache)):
        t_i, x_i, v_i = trajectory_cache[i]
        max_ratio = 0.0
        for _ in range(k):
            direction = random_normal_vector_like(x_i)
            direction_norm = norm(direction) + 1e-8
            x_perturbed = x_i + delta * direction / direction_norm
            v_perturbed = model.velocity(t_i, x_perturbed)
            ratio = norm(v_perturbed - v_i) / delta
            if ratio > max_ratio:
                max_ratio = ratio
        L.append(max_ratio)

    # Step 2: Compute per-step error proxy e_i = safety_factor * L_i * dt_i * grad_proxy
    # grad_proxy: estimated max gradient norm of reward w.r.t. state, updated via EMA
    # Initialize grad_proxy as 1.0, then update after each backward pass using actual gradient norms
    dt = [trajectory_cache[i+1][0] - trajectory_cache[i][0] for i in range(len(trajectory_cache)-1)]
    error_proxies = [safety_factor * L[i] * dt[i] * grad_proxy for i in range(len(dt))]

    # Step 3: Select active timesteps to ensure omitted error <= epsilon
    indexed_errors = list(enumerate(error_proxies))
    indexed_errors.sort(key=lambda x: x[1], reverse=True)
    total_error = sum(error_proxies)
    cumulative = 0.0
    active_indices = []
    for idx, err in indexed_errors:
        active_indices.append(idx)
        cumulative += err
        if cumulative >= total_error - epsilon:
            break
    return sorted(active_indices)
```
During backpropagation, only timesteps in `active_indices` are used for gradient computation; for others, the cached velocity is used as a constant (stop-gradient).

### (C) Why this design
We chose Lipschitz-based error proxies over fixed window sizes (as in MixGRPO) because the local smoothness of the velocity field varies across the trajectory; early steps may have high curvature requiring finer backprop, while later steps are smoother, and manual tuning cannot capture this variation. We chose finite-difference estimation from the forward cache rather than global Lipschitz bounds (which would be overly conservative) because the cache already provides states and velocities at no extra cost, giving local estimates that are tighter and computationally cheap. We use k=5 random directions and take the maximum to reduce underestimation risk, with a safety factor of 1.2 to hedge against directional sampling noise. The greedy selection algorithm (keeping the largest error-producing timesteps until cumulative error meets the bound) ensures the total omitted error is within ε, providing a clear fidelity guarantee; the alternative of thresholding (keep timesteps with error > threshold) would not bound the total error. The trade-off is that the active set may be non-contiguous, complicating implementation, but gradient computation can be organized via checkpointing (e.g., recompute forward passes for omitted steps without gradient). We also chose to estimate Lipschitz constants via random-direction finite differences rather than full Jacobian computation because the latter is prohibitively expensive for high-dimensional states; finite differences give a practical proxy, accepting noise that we mitigate by averaging multiple directions and using the maximum.

### (D) Why it measures what we claim
The local Lipschitz constant L_t measures the sensitivity of the velocity to changes in state; combined with the timestep size dt_t and an assumed bound on the gradient of the reward with respect to the state (grad_proxy), the product L_t * dt_t * grad_proxy upper-bounds the norm of the gradient contribution from that timestep under a mean-value theorem argument for the backward pass; this upper bound is a proxy for the error incurred if that timestep is omitted from backpropagation because the gradient of the reward with respect to the initial state can be expressed as a recursive product of velocity Jacobians, and omitting a timestep is equivalent to replacing the Jacobian with the identity (error proportional to deviation). We assume the gradient error from omitting step i is bounded by L_i * dt_i * ||dR/dx_i||. This holds if the velocity Jacobian is Lipschitz continuous; failure mode: non-Lipschitz regions cause error bound violation. Mitigate by using a safety factor (1.2) and calibrating epsilon on a held-out set of 100 trajectories where ground-truth gradient error is computed via full backprop. The grad_proxy is estimated as an exponential moving average of observed gradient norms from prior iterations (decay=0.9), initialized to 1.0, and updated after each backward pass.

## Contribution

(1) A novel framework for automatic fidelity-adaptive gradient truncation in reward backpropagation for flow matching, using Lipschitz-based error proxies computed from the forward trajectory alone. (2) A closed-form expression for the truncation error bound in terms of local Lipschitz constants, enabling users to specify a fidelity tolerance ε. (3) A practical algorithm that eliminates manual tuning of backward surrogate design (e.g., cache-guide, full) while maintaining alignment quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Pick-a-Pic v2 (1M prompts, 50-step FLOW matching, resolution 256x256) | Fine-grained alignment needed; standard benchmark for reward backprop |
| Primary metric | HPS v2 score at fixed compute (100 gradient update steps) | Measures alignment vs efficiency; reflects human preference |
| Secondary metric | Gradient error (L2 norm difference between FARB gradient and dense gradient) on 100 synthetic trajectories with known ground truth | Validates error bound directly; isolates proxy accuracy |
| Baseline 1 | MixGRPO (fixed window size=10, matched to average FARB active set size) | Tests adaptivity hypothesis |
| Baseline 2 | LeapAlign (connector: 2-layer MLP, hidden=256, trained on 10K reward-state pairs) | Tests direct gradient vs proxy |
| Baseline 3 | Dense backprop (full trajectory of 50 steps) | Upper bound on performance |
| Ablation | FARB with uniform selection (same number of active timesteps as FARB, but chosen uniformly) | Isolates benefit of Lipschitz-based adaptivity |

### Why this setup validates the claim
This setup tests FARB's central claim: that Lipschitz-based error proxies can identify which timesteps matter most for gradient fidelity. Comparing against MixGRPO (fixed window) isolates whether adaptivity improves efficiency; LeapAlign tests whether direct backprop through selected steps beats learned approximations; full dense backprop sets the gold standard. The ablation (uniform selection with same compute) controls for the compute budget itself. HPS v2 is the right metric because it reflects human preference alignment, which is the ultimate goal of reward backpropagation, and measuring at fixed compute ensures we evaluate the tradeoff rather than raw improvement. The secondary metric on synthetic trajectories directly measures gradient error, allowing us to verify that the error bound holds empirically and that the proxy correlates with actual error.

### Expected outcome and causal chain

**vs. MixGRPO** — On a trajectory with high curvature in early timesteps (e.g., generating a complex scene) and smooth late steps, MixGRPO with a fixed window of 10 steps either truncates early high-error steps (if short) or wastes compute on late low-error steps (if long). FARB adaptively selects the early high-Lipschitz steps and skips the smooth later ones, capturing critical gradients efficiently. Thus we expect FARB to achieve HPS v2 scores that are at least 5% higher per unit compute, with the gap widening as trajectory curvature varies. Empirically, the average active set size for FARB is expected to be 20 out of 50 steps (i.e., 40% reduction vs dense), with epsilon=0.05.

**vs. LeapAlign** — LeapAlign learns a connector to approximate the reward gradient, which can introduce bias when the reward function is non-smooth (e.g., aesthetic score with sharp thresholds). On such a case, the connector's error distorts the gradient, leading to suboptimal updates. FARB directly backpropagates through selected timesteps using exact Jacobians (only omitting low-error steps), preserving fidelity. We expect FARB to outperform LeapAlign by at least 3 HPS v2 points when the reward landscape is complex (e.g., using a reward that combines multiple criteria).

**vs. Dense backprop** — Dense backprop provides the exact gradient but requires full compute (50 steps). With epsilon=0.01, FARB should achieve HPS v2 within 98% of dense, while using only 40% of the backward steps (20 vs 50). If epsilon is relaxed to 0.1, compute savings increase to 60% reduction but reward may drop to 95% of dense. The expected pattern is that FARB's reward improvement approaches dense as ε → 0, validating the Lipschitz proxy's accuracy. On synthetic trajectories, the gradient error (L2 norm) should be below epsilon for 95% of cases after calibration.

### What would falsify this idea
If FARB's reward gain relative to MixGRPO is uniform across all trajectories (i.e., no correlation between Lipschitz variation and gradient error), then the adaptive selection does not exploit the predicted error structure, falsifying the claim that Lipschitz proxies enable efficient fidelity-adaptive backpropagation. Specifically, if the Pearson correlation between per-step Lipschitz and actual gradient error (measured via full backprop on synthetic trajectories) is below 0.5, the proxy is unreliable.

## References

1. Exploring the Design Space of Reward Backpropagation for Flow Matching
2. Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is Secretly a GAN Discriminator
3. Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference
4. MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE
5. SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers
6. One-Step Diffusion with Distribution Matching Distillation
7. Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation
8. Training Diffusion Models Towards Diverse Image Generation with Reinforcement Learning

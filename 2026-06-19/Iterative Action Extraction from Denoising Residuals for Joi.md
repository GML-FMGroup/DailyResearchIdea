# Iterative Action Extraction from Denoising Residuals for Joint Planning and World Simulation

## Motivation

Existing methods like DiT4DiT generate actions in a single forward pass from current video features, without simulating future states or iteratively refining plans. This structural limitation prevents the model from correcting action plans based on simulated future outcomes, as the denoising process in diffusion models is not leveraged for planning. We address this by extracting actions from the residuals between consecutive denoising steps, enabling iterative planning where action generation and future-state simulation co-occur in a single process.

## Key Insight

The denoising residuals in a diffusion transformer encode the gradient of the video distribution with respect to the current noisy state, and the difference between residuals at consecutive noise levels provides a planning gradient that can be projected onto action space to yield the minimal intervention steering the simulation toward a desired future.

## Method

### Iterative Action Extraction from Denoising Residuals for Joint Planning and World Simulation

**Title:** Iterative Action Extraction from Denoising Residuals for Joint Planning and World Simulation

**Motivation:** Existing methods like DiT4DiT generate actions in a single forward pass from current video features, without simulating future states or iteratively refining plans. This structural limitation prevents the model from correcting action plans based on simulated future outcomes, as the denoising process in diffusion models is not leveraged for planning. We address this by extracting actions from the residuals between consecutive denoising steps, enabling iterative planning where action generation and future-state simulation co-occur in a single process.

**Key insight:** The denoising residuals in a diffusion transformer encode the gradient of the video distribution with respect to the current noisy state, and the difference between residuals at consecutive noise levels provides a planning gradient that can be projected onto action space to yield the minimal intervention steering the simulation toward a desired future.

**Load-bearing assumption (stated explicitly):** The difference between consecutive denoising residuals (`delta = residual(t) - residual(t-1)`) in a diffusion transformer trained on video trajectories is linearly related to the actions that generated those trajectories, i.e., there exists a linear transformation `W` such that `action ≈ W * delta`. This holds under the premise that the denoising process approximates a linearized dynamics around the data manifold, and that the noise schedule is sufficiently fine-grained. We verify this assumption via a calibration step before deployment: on a held-out set of 1000 trajectory snippets (each 10 steps long), we compute delta and perform linear regression to predict ground-truth actions; if the Pearson correlation between predicted and true actions exceeds 0.8, the assumption is considered valid and we proceed; otherwise, we fall back to using a separately trained inverse dynamics model (as in Cosmos-Predict1) as a safeguard, though this is not our primary method.

**Calibration procedure (concrete):**
1. Collect 1000 trajectory snippets of length L=10 from the target environment, each consisting of frames and actions.
2. For each snippet, run the denoising process from t=T to 0, compute delta at each step, and average delta over the snippet to get a single delta vector per step.
3. Train a linear ridge regression model (alpha=1.0) to map delta to action, using 800 snippets for training and 200 for validation.
4. Compute Pearson correlation on validation set. If correlation > 0.8, we use the learned linear projection layer (with weights from regression) for action extraction; otherwise, we use a pretrained inverse dynamics model (a 3-layer MLP with hidden=256, ReLU) as a fallback, noting that our core contribution is contingent on the assumption holding.

(A) **What it is:** Iterative Diffusion Planning (IDP) extracts actions from the delta between consecutive denoising residuals in a video diffusion transformer. Input: current observation, goal condition, and a pretrained video diffusion model. Output: a sequence of actions that iteratively refine the simulated future.

(B) **How it works:**
```python
def plan_actions(current_obs, goal_cond, diffusion_model, T=100, K=10, calib_linear=None):
    # Initialize noisy video at highest noise level
    x = add_noise(current_obs, noise_level=T)
    prev_residual = None
    actions = []
    for t in range(T, 0, -1):  # denoising loop (DDIM scheduler, eta=0.0)
        residual = diffusion_model(x, t, cond=goal_cond)  # predicts noise (epsilon)
        if prev_residual is not None:
            # Compute delta residual (planning gradient)
            delta = residual - prev_residual
            # Project to action space via calibrated linear layer (from calib_linear) if assumption holds, else fallback
            if calib_linear is not None:
                action = calib_linear(delta)  # linear projection with learned W
            else:
                # Fallback: use pretrained inverse dynamics model on denoised frames
                action = inv_dynamics(denoise_step(x, residual, t))  # not default
            actions.append(action)
        # Denoising step using DDIM scheduler
        x = denoise_step(x, residual, t)
        prev_residual = residual
    # Sample K actions uniformly from the sequence (step = T//K)
    return actions[::T//K]
```
**Hyperparameters:** T=100 denoising steps, K=10 output actions, linear projection layer: 512 input dim (residual dim) → action dim (e.g., 7 for Robosuite), trained via ridge regression (alpha=1.0) during calibration. Fallback inverse dynamics: 3-layer MLP, hidden=256, ReLU, trained on same calibration data.

(C) **Why this design:** We chose to extract actions from the delta of consecutive residuals rather than from intermediate features because the delta directly captures the change in the denoising direction across noise levels, which corresponds to the effect of actions on the state dynamics. We chose to use a shared diffusion transformer for both video generation and action extraction (as in DiT4DiT) rather than separate models because it ensures the residual delta is grounded in the same dynamics representation, avoiding feature misalignment that would require additional training. We chose to sample actions uniformly from the full denoising trajectory instead of using only the final steps because early denoising steps capture coarse dynamics that inform high-level planning, while later steps refine details; the uniform sampling balances resolution. The trade-off for using a shared transformer is that the model must be trained end-to-end with a joint objective, which may require large datasets and careful balancing of video and action loss weights. The trade-off for using uniform sampling is that it may waste steps in regions of low action relevance, but it avoids manual scheduling. Additionally, we include a calibration step to verify the linearity assumption; if it fails, we fall back to an inverse dynamics model, which reduces risk.

(D) **Why it measures what we claim:** The quantity `delta = residual(t) - residual(t-1)` measures the **planning gradient** because it quantifies how the model's prediction of the future state changes as noise is removed; under the assumption that the denoising process approximates a linearized dynamics (the residual is the negative gradient of the log-density), delta is proportional to the action effect. This assumption fails when dynamics are highly nonlinear or the diffusion model is poorly calibrated, in which case delta reflects stochastic fluctuations. The `action_projection` layer (linear, calibrated via ridge regression) maps delta to action space, measuring the **minimal intervention** that explains the change in denoising direction; this is justified under the assumption that the action space is linearly related to delta (holds for continuous low-level actions). When this assumption fails (e.g., discrete actions), the projection may extract inaccurate actions. The iterative denoising loop over T steps measures **cumulative plan refinement** because each step provides a correction signal based on the current state estimate; the number of output actions K controls the planning horizon. This operationalization assumes that the denoising trajectory is a valid simulation of environment dynamics, which holds only if the diffusion model is trained on trajectories reflecting the true transitions. The calibration step (Pearson correlation >0.8 on held-out data) provides a quantitative check on this assumption.

## Contribution

(1) We introduce Iterative Diffusion Planning (IDP), a method that extracts actions from the difference between consecutive denoising residuals in a video diffusion transformer, enabling iterative planning without separate rollout models. (2) We propose an action projection head that maps the residual delta to action space, trained end-to-end with the diffusion model, and show that the residual delta provides a planning gradient that can be used to refine actions across denoising steps. (3) We demonstrate that IDP integrates planning and simulation in a single iterative process, addressing the limitation of single-pass action prediction.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Robosuite (Can, Lift, Stack) and synthetic LinearDynamics-v0 | Standardized manipulation + controlled linear validation |
| Primary metric | Task success rate (50 episodes) | Direct measure of task completion |
| Baseline 1 | DiT4DiT | Joint video-action model, no delta extraction |
| Baseline 2 | Cosmos-Predict1 + inverse dynamics | World model with separate action inference |
| Baseline 3 (new) | IDP with single-step residual (t=1 only) | Isolates benefit of multi-step delta |
| Baseline 4 (new) | IDP with intermediate features (layer 6) | Tests if delta is better than activations |
| Ablation-of-ours | IDP without delta (final residual only) | Isolates benefit of delta residual |

**Sanity check environment**: LinearDynamics-v0 (linear state transitions, action = linear function of state difference) to validate delta-action equivalence.

### Why this setup validates the claim

This design checks the core claim—that delta residuals capture action-relevant dynamics—through four contrasts. DiT4DiT jointly predicts video and actions but ignores the residual change, testing whether delta adds value beyond a shared representation. Cosmos-Predict1 uses a video predictor plus a learned inverse dynamics model, testing whether our iterative extraction from the denoising trajectory outperforms a separate two-stage pipeline. The two new baselines (single-step residual and intermediate features) isolate the delta mechanism: single-step residual uses only the residual at t=1, while intermediate features uses the output of the 6th transformer layer at each step. The ablation removes the delta computation, using only the final denoising residual for action extraction, directly assessing the contribution of the delta mechanism. Task success rate in Robosuite is appropriate because it reflects both planning accuracy and execution fidelity; an improvement in success signals that extracted actions are more aligned with true dynamics. The dataset covers short and long-horizon tasks (e.g., Can vs. Stack), enabling detection of differential effects. The linear sanity check environment provides a falsifiable test: if delta does not exactly match ground truth actions (up to linear transformation) on linear dynamics, our assumption is violated. Failure modes like dynamics mismatch or model misalignment would manifest as success drops, providing falsifiable evidence.

### Expected outcome and causal chain

**vs. DiT4DiT** — On a Stack task requiring precise sequential actions, DiT4DiT produces smooth video predictions but extracts actions from a single feature layer, which may conflate dynamics across steps; the method yields misaligned low-level actions, causing the stack to tip. Our method instead computes delta residuals at each denoising step, isolating the change in dynamics that corresponds to each action, so actions are more accurate and temporally consistent. We expect a noticeable gap (e.g., 20-30% higher success) on long-horizon tasks, but parity on simple tasks like Can.

**vs. Cosmos-Predict1** — On a Lift task with variable object pose, Cosmos-Predict1's video predictions drift over the planning horizon, leading the inverse dynamics module to infer incorrect actions from an inaccurate future state. Our method iteratively refines actions via the denoising trajectory, using delta residuals to correct drift at each step; the actions stay grounded in the updated state estimate. We expect our method to maintain >85% success while Cosmos-Predict1 degrades to ~60% with increasing task uncertainty.

**vs. IDP with single-step residual** — On PickPlace, single-step residual captures only the immediate change at low noise, missing multi-step dynamics; our delta across all steps accumulates planning gradient, yielding more consistent orientations. Expect 10-15% success improvement.

**vs. IDP with intermediate features** — Intermediate features encode high-level semantics but lack the directional change captured by delta; on Stack, intermediate features may confuse object identities, causing misalignment. Expect our method to be 10-20% better.

**vs. Ablation (IDP without delta)** — On the PickPlace task requiring exact gripper orientation, the ablation extracts actions from the final residual only, which captures the overall denoising direction but misses stepwise corrections; it often chooses a wrong grasp orientation. Our delta-based method uses the change in residual across steps to compute a planning gradient, yielding precise orientation adjustments. We expect the ablation to be ~15% lower in success, confirming that delta is critical for fine-grained control.

### What would falsify this idea
If our method shows no significant improvement over the ablation on tasks demanding precise or long-horizon actions, or if it underperforms DiT4DiT on any subset, the core claim that delta residuals are a superior planning signal would be falsified. Additionally, if on LinearDynamics-v0 the Pearson correlation between delta and ground-truth action is ≤0.8 or the linear projection yields >10% action error, the linearity assumption is invalid and the method's foundation is undermined.

## References

1. DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
2. World Simulation with Video Foundation Models for Physical AI

# Adaptive Diffusion World Models via Uncertainty-Guided Online Fine-Tuning

## Motivation

Existing diffusion world models, such as those used in World4RL, are frozen after offline training, causing policy rollouts to diverge as the refined policy produces out-of-distribution actions. This structural limitation arises because the world model has no mechanism to detect distribution shift nor to adapt to new policy behavior, leading to accumulating errors and degraded policy performance.

## Key Insight

The predictive variance of a diffusion world model across multiple denoising trajectories is a principled and efficient proxy for epistemic uncertainty, enabling selective fine-tuning on transitions that most impact model accuracy.

## Method

### (A) What it is
UDAPT (Uncertainty-Driven Adaptive Policy Training) continuously fine-tunes a frozen diffusion world model on policy rollout data, using an uncertainty estimate to select high-impact transitions and elastic weight consolidation (EWC) to prevent catastrophic forgetting. Inputs: base diffusion world model W_0, policy π, offline replay buffer D_off. Output: adapted world model W.

### (B) How it works
```python
# Hyperparameters
τ = 0.05          # uncertainty threshold, adjusted via calibration (see below)
α = 1e-4          # learning rate
λ = 0.1           # EWC regularization strength
k = 10            # fine-tuning steps per policy episode
N_var = 10        # number of denoising trajectories for variance estimation
calib_freq = 100  # calibration frequency (policy iterations)
calib_set = 512  # size of calibration set (held-out OOD transitions)

# Initialize
W = copy(W_0)
F = compute_fisher(W_0, D_off)  # Fisher information matrix for EWC
D_on = deque(maxlen=10000)       # online replay buffer

# Precompute calibration set: use a small set of OOD transitions from a perturbed policy
# to establish initial threshold τ0
D_calib = generate_ood_transitions(W_0, n=calib_set)  # e.g., using a random policy

for each policy refinement iteration:
    # 1. Collect rollout data from world model
    trajectories = rollout(π, W, horizon=50)
    
    # 2. Identify high-uncertainty transitions
    for (s, a, s_next) in transitions(trajectories):
        # Sample N_var denoising paths to estimate variance
        samples = [W.sample_next(s, a) for _ in range(N_var)]
        var = mean(samples, axis=0).var()
        if var > τ:
            D_on.append((s, a, s_next))
    
    # 3. Calibrate uncertainty threshold every calib_freq iterations
    if iteration % calib_freq == 0:
        # Compute correlation between variance and actual prediction error on D_calib
        vars_calib = []
        errors_calib = []
        for (s, a, s_next) in D_calib:
            samples = [W.sample_next(s, a) for _ in range(N_var)]
            var = mean(samples, axis=0).var()
            pred = mean(samples, axis=0)
            error = mse(pred, s_next)
            vars_calib.append(var)
            errors_calib.append(error)
        correlation = pearsonr(vars_calib, errors_calib)
        # Adjust τ to maintain a target false positive rate (e.g., 10%)
        # using a simple percentile-based heuristic: set τ such that only 10% of low-error transitions exceed τ
        low_error_mask = errors_calib < np.percentile(errors_calib, 50)  # lower half
        high_var_low_error = [v for v, e in zip(vars_calib, errors_calib) if e < np.percentile(errors_calib, 50) and v > τ]
        fpr = len(high_var_low_error) / sum(low_error_mask)
        if fpr > 0.1:  # increase threshold to reduce false positives
            τ *= 1.1
        elif fpr < 0.05:  # decrease threshold to capture more high-uncertainty transitions
            τ *= 0.9
        # Log correlation for monitoring
        log(f'Iter {iteration}: correlation={correlation:.2f}, τ={τ:.4f}, fpr={fpr:.2f}')
    
    # 4. Fine-tune W with EWC
    for step in range(k):
        batch = sample_mixed(D_on, D_off, online_ratio=0.5)
        loss = diffusion_loss(W, batch) + λ * ewc_loss(W, W_0, F)
        W.update(loss, lr=α)

# ewc_loss = sum_{i} (F_i * (W_i - W_0_i)^2)
```

### (C) Why this design
We chose predictive variance over alternative uncertainty measures (e.g., model ensemble or entropy) because variance from multiple denoising trajectories is computationally efficient for diffusion models and directly reflects the model's instability in generation. The trade-off is that multiple samples increase inference cost, but we limit to 10 samples to keep overhead manageable. We selected EWC over simpler replay because EWC explicitly protects important weights for offline data, which is critical since offline data covers a broad distribution; the cost is additional memory for Fisher information. We mix online and offline data in a 1:1 ratio to balance adaptation and retention; a higher online ratio would accelerate adaptation but risk forgetting basic dynamics. The uncertainty threshold τ is set initially based on a held-out calibration set of OOD transitions to minimize false positives (transitions that are actually predictable but have high variance due to noise), and is periodically updated based on the correlation between variance and actual prediction error, ensuring the threshold adapts to the model's calibration state.

### (D) Why it measures what we claim
The predictive variance computed from multiple denoising trajectories measures the diffusion model's epistemic uncertainty about the next observation because it reflects the conditional variance of the generative process given the training data; this equivalently estimates the model's expected prediction error under the assumption that the model is well-calibrated in its conditional distribution. This assumption fails when the model is overconfident on out-of-distribution transitions (e.g., due to mode collapse), in which case variance may underestimate uncertainty and lead to missed high-impact transitions. To mitigate this, we periodically compute the Pearson correlation between predictive variance and actual next-frame MSE error on the calibration set D_calib (see (B) step 3). If the correlation drops below a threshold (e.g., 0.3), we increase the number of denoising paths N_var to 20 or activate a separate calibration mechanism (e.g., temperature scaling of variance). Additionally, we monitor the false positive rate (FPR) of variance-based selection on low-error transitions and adjust τ to keep FPR below 10%. This quantitative verification ensures that variance remains a reliable proxy for uncertainty, even under distribution shift. The EWC regularization term measures the deviation from the offline-trained weights, preserving knowledge of the original data distribution; this operationalizes the concept of preventing catastrophic forgetting because it penalizes changes to parameters that are important for offline data, under the assumption that the Fisher information approximates parameter importance. This assumption fails if the offline data is not representative of all necessary dynamics, causing EWC to overly constrain adaptation; we address this by including online data in the fine-tuning loss and by only applying EWC to layers with high Fisher values (top 50% by magnitude).

## Contribution

(1) A novel online fine-tuning framework for diffusion world models that adapts to policy-induced distribution shift without requiring retraining from scratch. (2) An uncertainty-guided transition selection mechanism based on predictive variance that identifies high-impact transitions for efficient adaptation. (3) A demonstration that combining EWC regularization with mixed replay prevents catastrophic forgetting while enabling adaptation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Robosuite (6 tasks: Door, Lift, NutAssembly, PegInsert, PickPlace, Stack) | Standard robotics benchmark with diverse dynamics and long-horizon options |
| Primary metric | Success rate (%) | Direct measure of task completion |
| Secondary metrics | Episode length, average return, predictive variance–error correlation | Capture efficiency, reward, and uncertainty calibration |
| Baseline: World4RL | World4RL | Baseline diffusion world model without UDAPT |
| Baseline: DIAMOND | DIAMOND | Diffusion world model with RL from scratch |
| Baseline: DreamerV3 | DreamerV3 | State-of-the-art latent world model |
| Ablation-of-ours | UDAPT w/o uncertainty selection | Isolates effect of uncertainty-guided selection |
| Additional ablation | UDAPT with fixed threshold (no calibration) | Isolates effect of adaptive threshold calibration |

### Why this setup validates the claim
This setup directly tests the central claim that uncertainty-driven adaptive fine-tuning improves policy refinement by focusing on high-impact transitions while preserving offline knowledge. The dataset contains diverse manipulation tasks requiring varied dynamics, challenging the world model's generalization. NutAssembly and PegInsert are long-horizon tasks (≥100 steps) that expose accumulated prediction errors. The primary metric captures task completion, which reflects policy quality. World4RL tests the value of adaptation itself (no fine-tuning), DIAMOND tests a different training paradigm (RL from scratch in the world model), and DreamerV3 tests against a non-diffusion latent model. The ablation removes uncertainty selection, isolating its contribution. A second ablation with fixed threshold tests the benefit of periodic calibration. Together, these baselines and ablations create a falsifiable test: if UDAPT outperforms all, the uncertainty-guided adaptation is beneficial; if not, the core mechanism is ineffective.

### Expected outcome and causal chain

**vs. World4RL** — On a task requiring precise dynamics after policy change (e.g., peg insertion with new controller), World4RL's fixed world model produces inaccurate rollouts, leading to poor policy updates because it cannot adapt to distributional shifts. Our method instead fine-tunes on high-uncertainty transitions (where prediction variance is high), correcting the model, so we expect UDAPT to show a noticeable success rate improvement (e.g., +15-20%) on such tasks while matching World4RL on static tasks.

**vs. DIAMOND** — On a task with limited online data (e.g., 50 policy episodes), DIAMOND retrains the world model from scratch using only new data, causing catastrophic forgetting of basic dynamics and producing unstable rollouts. Our method uses EWC and replay mixing to retain offline knowledge, so we expect UDAPT to maintain higher success rates (e.g., +10-15%) when online data is scarce, while DIAMOND might degrade.

**vs. DreamerV3** — On a task requiring fine-grained visual detail (e.g., object texture variation), DreamerV3's discrete latent representation loses fidelity, leading to inaccurate predictions and suboptimal policy. Our diffusion world model operates directly in pixel space, preserving detail, and adaptation further refines predictions. Thus, we expect UDAPT to outperform DreamerV3 by a larger margin on visually detailed tasks (e.g., +20-30%) than on simple geometric tasks (e.g., +5-10%).

### What would falsify this idea
If UDAPT does not outperform UDAPT without uncertainty selection on the subset of tasks where prediction variance is high and the variance–error correlation remains above 0.5, then the uncertainty-guided selection mechanism is not the driver of improvement, falsifying the central claim. Additionally, if the periodic calibration does not maintain a positive correlation (ρ < 0.3) on the calibration set, the assumption that variance proxies epistemic uncertainty is invalidated.

## References

1. World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation
2. Diffusion for World Modeling: Visual Details Matter in Atari
3. Navigation World Models
4. STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning
5. GAIA-1: A Generative World Model for Autonomous Driving
6. Transformer-based World Models Are Happy With 100k Interactions
7. Learning Interactive Real-World Simulators
8. NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration

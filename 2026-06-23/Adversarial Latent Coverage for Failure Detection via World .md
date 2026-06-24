# Adversarial Latent Coverage for Failure Detection via World Model Uncertainty

## Motivation

Existing failure detection methods that rely on world model latents (e.g., Foresight) assume the learned representation captures all failure modes. However, world models trained only on successful trajectories have blind spots in latent regions corresponding to failure-inducing action sequences, because the training distribution does not cover these. This structural incompleteness leads to missed failures that deviate from the training manifold. We need a mechanism to actively expand coverage without requiring failure labels.

## Key Insight

A min-max game between a challenger that generates action sequences maximizing the world model's predictive uncertainty and the world model that minimizes that uncertainty on those sequences forces the world model to become a complete representation of all possible action outcomes, including failures.

## Method

### (A) What it is
ALC-FD (Adversarial Latent Coverage for Failure Detection) is a self-supervised training framework that, given only successful trajectories, iteratively trains a world model to reduce its predictive uncertainty on adversarial latent trajectories, thereby covering failure modes. Input: successful trajectories (observations & actions). Output: a fine-tuned world model whose latent representation is complete w.r.t. failure modes.

### (B) How it works (pseudocode)
```python
# Phase 1: pre-train world model W on successful trajectories
# W is a Gaussian RSSM with deterministic recurrent state (size 256) and stochastic latent (size 32)
W = pretrain_world_model(successful_trajectories)  # ELBO objective, learning rate 1e-3, Adam

# Phase 2: adversarial loop
buffer = []  # stores challenger-generated trajectories (max size 50000)
for iteration in range(iterations=10):
    # Train challenger C to maximize W's predictive entropy
    # C is a 2-layer MLP (hidden=256, GeLU) outputting action mean, variance fixed to 0.1
    for episode in range(E=1000):
        s = sample_initial_state(successful_trajectories)
        traj = []
        for t in range(T=32):
            a = C(s)  # sample from Gaussian(mean, 0.1)
            z_t, _ = W.encode(s)  # get deterministic+stochastic latent
            z_{t+1} = W.transition(z_t, a)  # deterministic+stochastic, predictive distribution N(μ, Σ)
            # predictive entropy = 0.5*log(det(Σ_{t+1})) + const
            u_t = 0.5 * W.logdet_cov(z_t, a)
            traj.append((z_t, a, z_{t+1}, u_t))
            s = W.decode(z_{t+1})  # or use actual env step (optional)
        total_uncertainty = sum(u_t)
        # REINFORCE: maximize expected total_uncertainty
        update C via policy gradient (REINFORCE with baseline, lr=1e-4, Adam)
        buffer.extend(traj)
        if len(buffer) > 50000: buffer = buffer[-50000:]
    # Fine-tune W on buffer to reduce uncertainty
    for (z_t, a, z_{t+1}, u_t) in buffer:
        # Standard ELBO on forward dynamics
        loss_elbo = -ELBO(z_t, a, z_{t+1})  # reconstruction + KL (β=1.0)
        # Entropy minimization: reduce predictive uncertainty on next latent
        loss_unc = u_t
        # Total loss
        loss_total = loss_elbo + 0.1 * loss_unc
    update W (lr=1e-4, Adam)
    # Verification: after each iteration, evaluate W's predictive log-likelihood on a held-out set of 512 successful trajectories to ensure no overconfidence
```
Hyperparameters: iterations=10, E=1000, T=32, λ=0.1, W learning rate=1e-4, C learning rate=1e-4, buffer size=50000.

### (C) Why this design
We chose a learned challenger policy (C) over random shooting because it can more efficiently explore the latent space, especially in high-dimensional action spaces, accepting the cost of additional training instability. We use entropy minimization on challenger trajectories (rather than maximizing a task-based adversarial loss) to directly measure and reduce knowledge gaps, which aligns with the objective of coverage. We pre-train the world model first to provide a stable latent space; without it, the adversarial loop may collapse as the challenger exploits undefined regions. Unlike standard adversarial training (e.g., AT for classification) which maximizes task loss, our challenger maximizes entropy, a self-supervised signal that does not require failure labels. This design forces the world model to become predictive on previously uncertain trajectories, expanding coverage without manual annotation.

### (D) Why it measures what we claim
The challenger's objective (maximizing predictive entropy of W) measures the world model's knowledge gaps because entropy quantifies uncertainty in latent predictions; this assumption fails when the challenger's optimization is insufficient (e.g., local optima in action space), in which case only a subset of blind spots are exposed. The world model's fine-tuning loss (minimizing entropy on challenger sequences) directly operationalizes completeness: by penalizing high uncertainty, the world model must learn to represent those state-action pairs; this assumption fails when the entropy loss dominates and forces overconfident predictions without actual representation improvement (e.g., if W learns to output Dirac deltas regardless of true dynamics). In that case the loss no longer reflects coverage but rather a collapsed distribution. We mitigate this by retaining the ELBO objective, which enforces reconstruction and prevents trivial solutions. Critically, for entropy minimization to translate to coverage of failure modes, we assume that the challenger explores all relevant failure-inducing state-action sequences. To verify this assumption, we periodically evaluate W's predictive log-likelihood on a held-out set of 512 successful trajectories; a significant drop would indicate overconfidence, prompting early stopping or adjustment of λ.

## Contribution

(1) ALC-FD, an adversarial training framework that iteratively expands the coverage of world model latent representations by having a challenger generate high-uncertainty trajectories and fine-tuning the world model to reduce that uncertainty. (2) The empirical finding that this self-supervised min-max game, using only successful demonstrations, enables failure detection on previously unseen failure modes without any failure labels. (3) Integration with conformal prediction (as in FAIL-Detect) to calibrate the detection threshold with rigorous false-positive control.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LIBERO-Long (simulation) | Long-horizon tasks with diverse failures |
| Primary metric | AUC-ROC for failure detection | Threshold-free ranking measure |
| Baseline 1 | Prediction error threshold (raw policy) | Tests deterministic uncertainty |
| Baseline 2 | MC Dropout uncertainty threshold | Tests stochastic uncertainty |
| Baseline 3 | RND-based uncertainty threshold (random network distillation, prediction error of random features) | Tests exploration-driven uncertainty without adversarial loop |
| Ablation-of-ours | Pre-trained WM with uncertainty threshold (no adversarial fine-tuning) | Isolates adversarial coverage benefit |

### Why this setup validates the claim
LIBERO-Long provides long-horizon manipulation tasks where failures often stem from subtle deviations, testing detection of gradual errors. Baselines capture three common unsupervised uncertainty estimates: deterministic reconstruction error (prediction error threshold), stochastic dropout (MC Dropout), and exploration-driven prediction error (RND). Our ablation isolates the effect of adversarial refinement. AUC-ROC evaluates ranking ability across thresholds, directly measuring whether our method assigns higher uncertainty to failures than successes. This combination tests the central claim: adversarial latent coverage expands the world model's awareness of failure modes without failure data. The comparison to baselines and ablation reveals whether the adversarial mechanism is necessary and effective. Adding RND baseline further tests if any exploration-based uncertainty signal suffices, or if the adversarial loop's targeted entropy maximization is uniquely beneficial.

### Expected outcome and causal chain

**vs. Prediction error threshold** — On a case where gripper position drifts slowly, reconstruction error stays low because the world model interpolates. The baseline misses the failure. Our challenger searches for high-entropy latent transitions, forcing the world model to become uncertain on those trajectories after fine-tuning. Thus, our model detects gradual failures that the baseline overlooks. We expect a noticeable AUC gap on gradual-failure tasks (e.g., +0.15 AUC) and parity on abrupt failures.

**vs. MC Dropout uncertainty threshold** — On a case with noisy lighting but safe task, MC Dropout yields high uncertainty due to stochastic forward passes, causing false positives. Our challenger maximizes entropy of the deterministic latent prediction, not sampling-based, so it is less sensitive to input noise. Fine-tuning further reduces uncertainty on predictable trajectories. We expect our method to have lower false positive rate (e.g., -20% FPR at same recall) under environmental variation.

**vs. RND-based uncertainty threshold** — RND uses a fixed random network and measures prediction error of a predictor on that feature space; it does not explicitly target world model uncertainty. The adversarial loop directly maximizes entropy in the world model's latent predictions, forcing fine-tuning on precisely those high-uncertainty regions. We expect our method to achieve higher AUC on failure types that require precise latent coverage (e.g., action sequences leading to unmodeled collisions), while RND may perform comparably on simple perceptual novelty. A gap of +0.10 AUC would support the adversarial loop's advantage.

### What would falsify this idea
If our method's AUC is not significantly better than the ablation (pre-trained world model with uncertainty threshold) on failures involving novel state-action combinations unseen in successful data, then adversarial coverage does not effectively expand awareness of failure modes.

## References

1. Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents
2. Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies
3. SAFE: Multitask Failure Detection for Vision-Language-Action Models
4. Revisiting Feature Prediction for Learning Visual Representations from Video
5. AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation
6. Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress
7. Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
8. VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking

# Self-Supervised Trajectory Quality Estimation via Progress-Consistent World Models

## Motivation

World Value Models (WVM) achieve state-of-the-art value estimation but rely on expensive task progression labels, limiting scalability. No existing work provides a self-supervised alternative that removes this label dependency. We address this gap by leveraging the temporal consistency of future predictions under a learned latent progress variable as a self-supervised proxy for trajectory quality.

## Key Insight

Successful manipulation trajectories exhibit temporally consistent dynamics that can be accurately predicted under a smoothly varying latent progress variable, enabling unsupervised quality estimation via prediction error.

## Method

### (A) What it is
**SPC-WM** (Self-supervised Progress-Consistent World Model) is a framework that learns a latent progress variable and a dynamics predictor from unlabeled robot trajectories, and uses the prediction consistency to estimate trajectory quality without any task progression labels.

**Input:** Offline dataset of unlabeled trajectories $\mathcal{D} = \{\tau_i = (o_1, a_1, \dots, o_T)\}$ where $o_t$ are observations (images) and $a_t$ are actions.
**Output:** For each trajectory, a calibrated quality score $q(\tau) \in [0,1]$ indicating likelihood of task success.

### (B) How it works
```python
# Architecture
# progress_encoder: 2-layer MLP, hidden=256, GeLU activation, applied to DINOv2 patch features (dim=768) -> output dim 1
progress_encoder: ϕ(o_1, …, o_t) → z_t ∈ ℝ (e.g., using causal masking on DINOv2 patch features)
# predictor: 2-layer MLP, hidden=512, ReLU activation, output dim=768 (DINOv2 patch features)
predictor: ψ(o_t, a_t, z_t) → ō_{t+1}  # predicts next observation features (DINO patch features)

# Training loop
for epoch in range(num_epochs=200):
    for batch of trajectories τ in D (batch_size=32):
        for each trajectory:
            # 1. Encode progress at each timestep
            Z = [ϕ(o_1:t) for t in 1..T]
            # 2. Prediction consistency loss (multi-step, K=5)
            L_pred = 0
            for t in 1..T-1:
                for k in 1..K:  # K=5
                    if t+k <= T:
                        # Rollout using teacher forcing: use observed o_t as features
                        ē_{t+1} = ψ(feat(o_t), a_t, z_t)
                        for step in 2..k:
                            ē_{t+step} = ψ(ē_{t+step-1}, a_{t+step-1}, z_{t+step-1})
                        L_pred += MSE(ē_{t+k}, feat(o_{t+k}))
            # 3. Progress smoothness loss
            L_smooth = Σ_t ||z_{t+1} - z_t||^2
            # 4. Weak monotonicity: encourage z_t to increase on average
            L_mono = max(0, mean(z_t - z_{t-1}) - ε)  # ε = -0.01
            # Total loss
            loss = L_pred + λ_smooth * L_smooth + λ_mono * L_mono  # λ_smooth=0.1, λ_mono=0.01
            update ϕ, ψ (Adam, lr=1e-4)

# Quality estimation (inference)
# After training, calibrate using a small labeled validation set (512 trajectories).
# Compute avg_error (mean MSE over all steps and horizons) and monotonicity (mean(z_t - z_{t-1})) for each trajectory.
# Fit logistic regression model: p(success|avg_error, monotonicity) = sigmoid(w0 + w1*avg_error + w2*monotonicity).
# Calibrated quality score q(τ) = p(success)
```

**Load-bearing assumption:** Successful trajectories have lower multi-step prediction error under a progress-consistent dynamics model than failed trajectories, even for predictable failures. This assumption is tested via a preliminary calibration plot (see experiment) and a small labeled validation set is used to adjust scores if the assumption holds weakly.

### (C) Why this design
We chose to learn a latent progress variable $z_t$ rather than using a hand-crafted distance-to-goal because it can adapt to diverse tasks without supervision; the trade-off is that the learned variable may not align with true progress if smoothness constraints are too weak, but we rely on prediction consistency to provide an implicit gradient. We chose to predict observation features (e.g., DINOv2 patch features) rather than raw pixels because features are more robust to visual noise and reduce computational cost; the trade-off is that fine-grained spatial details may be lost, but semantic consistency suffices for quality estimation. We chose a multi-step prediction horizon ($K=5$) over single-step because long-term consistency is a stronger indicator of trajectory quality; the trade-off is increased prediction difficulty and potential error accumulation, but this is mitigated by using feature space and teacher forcing during training. We chose to enforce monotonicity via a weak penalty rather than a hard constraint because occasional regressions (e.g., in grasp-and-lift tasks) should not fully disqualify a trajectory; the trade-off is that the progress variable might be non-monotonic for some successful trajectories, reducing discriminability.

### (D) Why it measures what we claim
The average prediction error measures trajectory quality because it quantifies how well the world model can forecast future observations given the learned progress variable; this assumes that successful trajectories follow predictable dynamics (e.g., consistent gripper closing, object motion) while unsuccessful ones are erratic. This assumption fails when a failed trajectory is also predictable (e.g., a robot repeatedly missing a grasp in the same way), in which case the error would incorrectly indicate high quality. The monotonicity of $z_t$ measures task progression because it assumes that progress should increase over time; this assumption fails in tasks with inherent regress (e.g., assembly where a part is temporarily placed and then moved), which would penalize successful trajectories. The combination of both signals mitigates these failure modes: high prediction error catches erratic failures, while low monotonicity catches predictable failures that lack progression, but neither is perfect alone. To address predictable failures, we introduce a calibration step using a labeled validation set to adjust the score threshold.

## Contribution

(1) A self-supervised framework (SPC-WM) that learns a latent progress variable and uses prediction consistency to estimate trajectory quality without any task progression labels. (2) The insight that temporal consistency of future predictions under a learned progress variable correlates with trajectory success, enabling scalable data filtering for policy learning. (3) A practical method combining feature-space prediction, multi-step forecasting, and weak monotonicity constraints to produce robust quality scores.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Suboptimal-Value-Bench (WVM) | Contains suboptimal trajectories with known outcomes. |
| Primary metric | Value-Order Correlation (VOC) | Measures ranking quality of trajectory scores. |
| Baseline | WVM (World Value Model) | Supervised value function trained on optimal data. |
| Baseline | V-JEPA 2 (video prediction) | Self-supervised video prediction without progress variable. |
| Baseline | DataMIL (datamodel selection) | Uses datamodels to predict trajectory utility. |
| Baseline | Standard video prediction (no progress variable) | Same architecture but without progress encoding; uses only single-step prediction error. |
| Ablation | SPC-WM without monotonicity | Tests importance of progress smoothness constraint. |
| Downstream task | Policy learning on filtered vs unfiltered data | Evaluate if filtering with SPC-WM improves RL performance (e.g., TRPO on Robosuite pick-place). |

### Why this setup validates the claim

Suboptimal-Value-Bench directly tests trajectory quality estimation on non-expert data, which is the target domain. VOC evaluates ranking of trajectories by predicted success likelihood, aligning with our goal. WVM represents supervised methods that require task labels; V-JEPA-2 and the standard video prediction baseline test raw prediction consistency without a progress variable; DataMIL tests a different unsupervised approach. The ablation isolates the monotonicity loss. The downstream task demonstrates practical utility. Additionally, we conduct a preliminary experiment on a small labeled subset (100 trajectories) to plot prediction error vs. ground-truth success probability, visually verifying the core assumption that successful trajectories yield lower error. Together, they allow falsification: if SPC-WM does not outperform baselines on suboptimal successes (where WVM fails) or predictable failures (where V-JEPA-2 fails), the central claim is unsupported.

### Expected outcome and causal chain

**vs. WVM** — On a suboptimal trajectory with erratic actions but eventual success, WVM assigns a low value because it was trained only on optimal demonstrations, lacking generalization to suboptimal paths. Our method uses prediction consistency across multi-step horizons; despite erratic actions, if the progression is predictable (e.g., object moves toward goal), the error remains low, yielding a high score. We expect a noticeable gap on this subset, with SPC-WM achieving >0.2 higher VOC than WVM.

**vs. V-JEPA 2** — On a visually predictable but task-failing trajectory (e.g., repeatedly missing a grasp), V-JEPA 2 assigns low prediction error because it predicts features without a progress variable; thus it incorrectly rates the trajectory highly. Our method includes a monotonicity penalty, so low progress increase (e.g., no object displacement) reduces the score. We expect SPC-WM to have >0.3 lower VOC on such failures than V-JEPA-2, but higher on suboptimal successes.

**vs. DataMIL** — On a trajectory that is informative for training but not successful (e.g., diverse random actions), DataMIL may assign a high score because datamodels capture learning influence rather than task success. Our method directly measures task consistency via prediction error and monotonicity, which correlates with actual success. We expect SPC-WM to outperform DataMIL by >0.1 VOC overall, especially on trajectories that are uninformative yet successful (where DataMIL undervalues) and on informative failures (where DataMIL overvalues).

**vs. Standard video prediction (no progress variable)** — Without the progress variable, prediction error alone cannot distinguish predictable failures. SPC-WM with monotonicity penalizes such failures, thus we expect >0.15 VOC improvement on the predictable failure subset.

**Downstream task** — Training a policy (e.g., TRPO) on filtered data (top 20% trajectories by SPC-WM score) will achieve higher success rate than training on the full dataset or on data filtered by baselines. We expect a 10% absolute improvement in task success.

### What would falsify this idea

If SPC-WM does not achieve a higher VOC than all baselines on the Suboptimal-Value-Bench, especially on the subset of predictable failures (where monotonicity should be decisive) or on suboptimal successes (where prediction consistency should shine), or if the downstream policy training does not improve over unfiltered data, then the claim that progress-consistency and monotonicity improve trajectory quality estimation is falsified.

## References

1. World Value Models for Robotic Manipulation
2. Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning
3. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
4. DataMIL: Selecting Data for Robot Imitation Learning with Datamodels
5. Stop Regressing: Training Value Functions via Classification for Scalable Deep RL
6. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
7. Scaling 4D Representations
8. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

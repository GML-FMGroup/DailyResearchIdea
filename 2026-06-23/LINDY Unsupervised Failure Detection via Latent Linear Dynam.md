# LINDY: Unsupervised Failure Detection via Latent Linear Dynamics

## Motivation

Existing failure detection methods for robotic manipulation (e.g., Foresight) require supervision—either final task outcomes or a calibration set of successful executions—to define normal behavior. This structural reliance on labels limits applicability where human feedback is expensive or infeasible. The root cause is that these methods treat detection as a supervised problem, needing explicit examples of normal or abnormal behavior.

## Key Insight

Normal task execution follows a structured progression approximable as a linear dynamical system in a learned latent space, while failures cause abrupt deviations that break this linear predictability, enabling detection solely from unlabeled trajectory data.

## Method

(A) **What it is**: LINDY (Linear Dynamics for Unsupervised Failure Detection) trains a latent linear dynamics model on unlabeled robot trajectories and flags failures when the linear prediction error exceeds a distribution-derived threshold. Input: stream of observations and actions. Output: per-step failure flag.

(B) **How it works**:
```pseudocode
# Stage 1: Learn latent linear dynamics (unsupervised)
Initialize encoder E (obs->latent z), linear dynamics model (A,B), decoder D (latent->obs)
for each unlabeled trajectory (o_1:T, a_1:T):
    for t=1...T-1:
        z_t = E(o_t)
        z_{t+1_pred} = A * z_t + B * a_t
        z_{t+1_actual} = E(o_{t+1})  # stop gradient through z_{t+1_actual} for dynamics training
        loss = MSE(o_{t+1}, D(z_{t+1_pred})) + beta * MSE(z_{t+1_pred}, z_{t+1_actual}) + KL(encoder)
        # beta=0.1 to balance reconstruction and linearity

# Stage 2: Compute threshold from training error distribution
Compute abs prediction error e_t = |z_{t+1_pred} - z_{t+1_actual}| for all training steps
Set threshold = 99th percentile of {e_t} (or robust alternative like median + 3*MAD)

# Stage 3: Deployment
for each new trajectory step:
    compute e_t as above
    if e_t > threshold: flag as failure
```

(C) **Why this design**: We chose a linear dynamics model over a nonlinear RNN because failures cause nonlinear jumps that linear models cannot capture, making prediction error a sensitive indicator (trade-off: linearity may underfit complex dynamics, but for manipulation tasks normal transitions are often locally smooth; we accept that highly agile motions may produce false positives). We trained the encoder and dynamics jointly end-to-end rather than separately to ensure the latent space is shaped by the linearity constraint (trade-off: joint training couples representation and dynamics, risking mode collapse; we used KL regularization to avoid degenerate latents). We selected the 99th percentile threshold over a learned detector because it avoids any label requirement (trade-off: percentile depends on the assumption that failures are rare in training; if failures constitute >1%, the threshold becomes too lenient; we accept this limitation and note that practitioners can audit training data for failure frequency). The MSE loss on both prediction and reconstruction balances two objectives—predictive error measures linearity, reconstruction preserves scene information; the balance β=0.1 was chosen empirically on a small held-out set of normal trajectories (note: no labels used; only to tune hyperparameter). Without β, the model might ignore latent linearity and rely on reconstruction alone.

(D) **Why it measures what we claim**: The quantity |z_{t+1_pred} - z_{t+1_actual}| measures the deviation from linear predictability because it operationalizes the assumption that normal transitions are well-approximated by z_{t+1} = A z_t + B a_t; this assumption fails when the dynamics are inherently nonlinear or when failures are indistinguishable from normal execution in the latent space, in which case the metric reflects modeling error rather than failure presence. The threshold derived from the training set percentile measures the typical linear prediction error because we assume failures are rare and thus the error distribution is dominated by normal transitions; this assumption fails when failures are frequent during training, causing the threshold to inflate and miss actual failures. The joint training loss with linearity penalty measures the degree to which latent dynamics are linear because the penalty explicitly minimizes the squared error between predicted and encoded latent; this assumption fails if the encoder is degenerate (e.g., constant output), in which case the loss is low but latents meaningless—we prevent this by the reconstruction term that forces latents to carry scene information.

## Contribution

(1) LINDY, a fully unsupervised failure detection framework that trains a latent linear dynamics model without any success/failure labels, enabling detection in settings where labeled data is unavailable. (2) A principled threshold selection method based on the distribution of linear prediction errors from normal trajectories, eliminating the need for a calibration set or human-specified thresholds.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LIBERO-Long (simulation) | Common long-horizon manipulation benchmark |
| Primary metric | Failure detection AUC | Balances precision and recall |
| Baseline 1 | Threshold on raw policy features | Tests if linear latent improves over raw |
| Baseline 2 | Monte Carlo dropout uncertainty | Tests if prediction error beats uncertainty |
| Baseline 3 | Task-specific detector (trained with failures) | Measures gap to supervised upper bound |
| Ablation-of-ours | LINDY without linearity penalty | Isolates effect of linear dynamics constraint |

### Why this setup validates the claim

The dataset LIBERO-Long provides diverse long-horizon tasks where normal transitions are locally smooth but failures cause abrupt nonlinearities—ideal for testing LINDY's core assumption that linear prediction error signals failures. The baselines isolate the sub-claims: thresholding raw features tests whether the learned latent space adds value; Monte Carlo dropout compares to a common unsupervised uncertainty measure; the supervised detector quantifies the cost of avoiding failure labels. The ablation removes the linearity penalty to check if the linear dynamics constraint is crucial or if reconstruction alone suffices. AUC is chosen because it thresholds across operating points, reflecting the practical need to balance detection and false alarms without manual threshold tuning. If LINDY outperforms baselines on AUC, especially on tasks where failures are distinct from normal dynamics, the claim is supported.

### Expected outcome and causal chain

**vs. Threshold on raw policy features** — On a case where a gripper slips during a pick, raw policy features (e.g., action logits) may not show a distinct anomaly because the policy still outputs similar probabilities. The baseline thus fails to detect the slip (low recall). Our method instead encodes the observation into a linear latent; the slip causes a large prediction error, exceeding the threshold, so we flag the failure. We expect our AUC to be >0.1 higher on tasks with sudden dynamical changes (e.g., dropping objects) but similar on smooth failures (e.g., slow drift).

**vs. Monte Carlo dropout uncertainty** — On a case where a robot approaches a table edge, dropout uncertainty may rise due to out-of-distribution visual inputs even though no failure occurs, causing false positives. Our method relies on prediction error in latent space, which remains low if the dynamics are still linear despite novel visuals. Thus we expect our false positive rate to be half that of the baseline on tasks with visual variation. AUC gain is concentrated on non-failure OOD inputs.

**vs. Task-specific detector (trained with failures)** — On a case where a new failure type (e.g., cable snag) occurs that was never seen during training, the supervised detector cannot recognize it (low recall). Our unsupervised method does not require failure labels; prediction error will spike for any deviation from linear dynamics, including novel failures. However, on known failure types, the supervised detector may have higher precision. We expect overall AUC to be competitive (within 0.05) on known failures, but significantly higher (≥0.15) on novel ones.

### What would falsify this idea

If LINDY's AUC is lower than the raw-feature threshold baseline across all task subsets, or if the ablation without linearity penalty performs similarly, then the central claim that linear prediction error in a learned latent space is a sensitive and specific failure indicator is false.

## References

1. Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents
2. Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies
3. SAFE: Multitask Failure Detection for Vision-Language-Action Models
4. Revisiting Feature Prediction for Learning Visual Representations from Video
5. AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation
6. Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress
7. Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
8. VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking

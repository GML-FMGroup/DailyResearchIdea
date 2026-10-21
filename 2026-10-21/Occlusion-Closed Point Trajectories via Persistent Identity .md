# Occlusion-Closed Point Trajectories via Persistent Identity Memory

## Motivation

Learning the 4D World through Point Trajectory Forecasting (VPM) assumes that every scene point can be tracked across all views and time, yet provides no mechanism for closing a trajectory when a point is occluded and later re-enters the field of view. This structural gap recurs across cross-view completion, HOT3D, and MEt3R: all treat correspondence as per-pair or per-capture, so the predicted trajectory set is open under realistic visibility dynamics. The root cause is that identity is never represented as a persistent memory that survives occlusion; it is re-derived independently at each frame-pair, making re-identification impossible when the point disappears.

## Key Insight

A point's identity is a latent variable that persists through occlusion, and the only observable evidence of its persistence is the geometric consistency of its 3D reappearance under the camera motion—so identity closure and trajectory forecasting must be optimized as a single differentiable constraint rather than as separate matching and prediction stages.

## Method

**What it is.** We introduce PIVOT (Persistent Identity Verification for Occlusion-closed Trajectories), a differentiable module that sits between the multi-view video encoder and VPM's trajectory head. It maintains an identity memory over the multi-view history, re-associates occluded points when they reappear, and feeds a closure loss back into the trajectory head.

**How it works.**

```pseudocode
# Inputs: multi-view frames {I_v^t}, camera poses {T_v^t}, VPM trajectory features f^t
# Outputs: closed point trajectories {p_i^t}, identity embeddings {z_i}

1. For each visible point i at time t, extract appearance descriptor a_i^t from multi-view features
   (128-d, from the encoder's penultimate layer) and geometric descriptor
   g_i^t = (T_v^t)^{-1} * p_i^t (point in world frame).
2. Identity memory M: store (z_i, last_seen_t, last_world_pos_i, last_velocity_i, last_cov_i)
   for every point ever observed. z_i = MLP_identity([a_i^t ; g_i^t]) with L2 normalization;
   MLP_identity is a 2-layer MLP (hidden=256, GeLU). Update M by EMA (momentum 0.9).
3. Occlusion detection: point i is occluded at t if its projected depth is behind a rendered depth buffer
   from any view by margin tau=0.05 m. Occluded points are removed from the active set but retained in M.
   Fallback (when rendered depth is unavailable): use a z-buffer of the trajectory head's own predicted
   point cloud; a point is occluded if its predicted depth is behind the nearest predicted surface by tau.
4. Re-identification at time t for a newly visible point j:
   a. Candidate retrieval: top-k=8 memory entries by cosine similarity of z_j to stored z_i.
   b. Geometric verification: for each candidate i, the trajectory head emits a predicted world position
      p_hat_i^t and a full 3x3 position covariance Sigma_i^t that widens with gap length
      g = t - last_seen_t and with predicted acceleration magnitude ||a_i||:
      Sigma_i^t = Sigma_0 + g^2 * diag(sigma_v^2) + (0.5 g^2)^2 * diag(sigma_a^2 * ||a_i||^2),
      with Sigma_0 = (0.02 m)^2 I, sigma_v = 0.05 m/s, sigma_a = 0.1 m/s^2 (calibrated per scene scale;
      see calibration below). Accept candidate if the Mahalanobis distance
      D_M = sqrt((p_hat_i^t - p_j^t)^T (Sigma_i^t)^{-1} (p_hat_i^t - p_j^t)) < chi2_thresh = 3.0
      (corresponding to ~95% confidence for 3 DoF). The residual used for gating is DETACHED from the
      closure gradient (stop_gradient on p_hat_i^t and Sigma_i^t in the gate computation) so the head
      cannot satisfy L_close by flattening velocities.
   c. If exactly one candidate passes, assign identity i to j and mark tracklet closed.
   d. If multiple pass, choose the one minimizing Mahalanobis distance under the predicted covariance;
      if none pass, create a new identity.
5. Closure loss L_close = -log softmax(sim(z_j, z_i*) / T) over the verified identity i*,
   with temperature T=0.07, applied only to re-identified points. Additionally, an acceleration
   regularizer L_acc = ||a_i||^2 is added with weight lambda_acc = 0.01 to prevent velocity suppression.
6. Joint training: total loss = L_video + lambda_traj * L_traj + lambda_close * L_close + lambda_acc * L_acc,
   with lambda_traj=1.0, lambda_close=0.5, lambda_acc=0.01. Gradients flow through z_j and through p_hat_i^t
   into the trajectory head (but NOT through the gate residual, which is detached), so identity closure
   shapes the predicted trajectories without a degenerate smoothing incentive. Warm-up: freeze lambda_close=0
   for the first 5 epochs, then linearly ramp to 0.5 over epochs 5-10.
```

**Calibration of the gate.** The covariance parameters (Sigma_0, sigma_v, sigma_a) are calibrated on a held-out set of 512 reappearance events from the simulated split of VP-WorldBench by maximizing the likelihood of the true reappearance position under the predicted Gaussian, and then scaled per scene by the median depth of the scene (scale factor = median_depth / 5 m). The chi2_thresh=3.0 is fixed a priori and not tuned on test.

**Why this design.** We chose a persistent memory with EMA updates over re-computing descriptors per frame because identity must survive the frames where the point is unobservable, accepting the cost that stale memory entries can drift and require geometric verification to prune. We chose geometric verification with a gap-length-conditioned predictive distribution over pure appearance matching because appearance changes under occlusion and viewpoint shift, and because constant-velocity extrapolation error grows with gap length and acceleration; the covariance widening makes the gate a valid identity test rather than a motion-smoothness test, accepting the cost that the covariance parameters must be calibrated per scene scale. We chose to back-propagate the closure loss into the trajectory head over treating re-identification as a post-hoc matching step because the trajectory head's predicted velocity is exactly the quantity that makes geometric verification work, accepting the cost that early training with noisy trajectories produces unstable closure gradients (mitigated by the warm-up schedule and the detached gate). A domain expert would not describe this as a variant of VPM: VPM has no identity memory and no closure loss, and its trajectory head is supervised only by ground-truth trajectories; PIVOT adds a differentiable re-identification pathway whose gradients reshape the trajectory head to predict velocities consistent with reappearance.

**Why it measures what we claim.** The cosine similarity sim(z_j, z_i*) measures identity persistence because the assumption that appearance and geometry descriptors are sufficient statistics for point identity holds when lighting and occlusion are the only nuisance factors; this assumption fails when two distinct points share appearance and geometry (e.g., repeated texture on a planar surface), in which case the similarity reflects surface similarity rather than identity. The Mahalanobis distance D_M measures trajectory consistency because the assumption that the trajectory head's gap-length-conditioned predictive distribution is calibrated (unbiased mean, covariance widening with gap and acceleration) holds when the calibration set is representative; this assumption fails during abrupt motion beyond the calibrated acceleration range, in which case the residual reflects prediction error rather than identity mismatch. The closure loss L_close measures the soft constraint that re-identified points should have consistent identities because the assumption that the softmax over verified candidates is calibrated when exactly one candidate passes geometric verification holds; this assumption fails when multiple candidates pass, in which case the loss reflects ambiguity rather than incorrect closure.

## Contribution

['A differentiable occlusion-aware re-identification module (PIVOT) that maintains a persistent point-identity memory across multi-view video history and enforces tracklet closure through learned identity embeddings plus geometric verification.', 'A joint training scheme in which the closure loss back-propagates into the trajectory forecasting head, making identity closure a soft constraint that reshapes predicted point velocities to be consistent with reappearance after occlusion.', 'An empirical design principle: geometric verification with constant-velocity extrapolation from the trajectory head is necessary to prune stale memory entries, and without it identity drift accumulates under long occlusions.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VP-WorldBench (simulated + real) | Contains occlusion-heavy multi-view scenes with trajectory ground truth |
| Primary metric | Identity-switch rate on reappearance subsets | Directly measures whether occluded points regain correct identity |
| Secondary metric | Trajectory consistency error (MSE) on reappearance frames | Tests whether closure improves predicted trajectories |
| Baseline | VPM (no identity memory) | Tests whether closure pathway is needed at all |
| Baseline | VPM + post-hoc Hungarian matching | Tests whether differentiable closure beats post-hoc association |
| Baseline | VPM + appearance-only re-ID | Tests whether geometric verification adds value over appearance |
| Baseline | VPM + constant-velocity delta gate (original PIVOT gate) | Tests whether gap-conditioned covariance gate is necessary |
| Ablation-of-ours | PIVOT without closure loss (frozen trajectory head) | Isolates gradient contribution of closure loss |
| Ablation-of-ours | PIVOT with detached gate residual (full model) vs. non-detached | Tests whether gradient through gate causes velocity suppression |
| Ablation-of-ours | PIVOT without acceleration regularizer | Tests whether L_acc prevents velocity flattening |
| Diagnostic subset | Repeated-texture subset (planar surfaces with repeated patterns) | Exposes descriptor-sufficiency failure mode F |

### Why this setup validates the claim

The central claim is that a persistent identity memory with geometric verification, trained via a closure loss that back-propagates into the trajectory head, reduces identity switches on reappearance and improves trajectory consistency after occlusion. VP-WorldBench is the right testbed because it supplies multi-view sequences with explicit occlusions and ground-truth trajectories, so reappearance events can be isolated from ordinary prediction error. Identity-switch rate on the reappearance subset is the correct primary metric because it is the only quantity that directly counts the failure the method targets; aggregate video-prediction metrics would dilute the effect across frames where no occlusion occurs. The baseline ladder tests each sub-claim in turn: plain VPM establishes that identity memory is necessary, post-hoc Hungarian matching establishes that differentiability (not just matching) is necessary, appearance-only re-ID establishes that geometric verification is necessary, and the constant-velocity delta gate baseline establishes that the gap-conditioned covariance gate is necessary rather than a motion-smoothness test. The frozen-head ablation is the sharpest test: if the gain persists without closure gradients, the differentiable pathway is decorative, and the design's core rationale collapses. The detached-gate and acceleration-regularizer ablations directly test the adversarial concern that the closure loss could be minimized by flattening predicted velocities rather than by improving identity. The repeated-texture diagnostic exposes the descriptor-sufficiency failure mode F, where appearance and geometry are not sufficient statistics for identity.

### Expected outcome and causal chain

**vs. VPM (no identity memory)** — On a sequence where a point is occluded for several frames behind a moving foreground object and then reappears, plain VPM has no stored descriptor or last-seen world position, so it treats the reappearing point as a new tracklet and the trajectory head restarts from the new observation. Our method instead retains the identity in memory, verifies the reappearance geometrically against the gap-conditioned predictive distribution, and closes the tracklet, so we expect a clear reduction in identity-switch rate concentrated on the long-occlusion subset, with near-parity on frames where nothing is occluded.

**vs. VPM + post-hoc Hungarian matching** — On a case where two points with similar appearance are occluded simultaneously and reappear near each other, post-hoc matching has no gradient signal tying predicted velocity to reappearance, so its assignment is driven purely by descriptor similarity and can swap the two identities. Our method instead back-propagates the closure loss into the trajectory head, which reshapes predicted velocities to be consistent with where each identity actually reappears, so we expect fewer swaps specifically on the crowded-similar-appearance subset while matching the baseline on isolated single-point occlusions.

**vs. VPM + appearance-only re-ID** — On a case where a point's appearance drifts across a long occlusion due to viewpoint change or lighting shift, appearance-only re-ID retrieves the wrong memory entry or none at all, spawning a duplicate identity. Our method instead gates candidates by the Mahalanobis residual against the gap-conditioned predictive distribution, rejecting appearance matches that are physically implausible, so we expect the gain to concentrate on long-gap and viewpoint-shift subsets, with parity on short-gap cases where appearance is stable.

**vs. VPM + constant-velocity delta gate (original PIVOT gate)** — On long-occlusion gaps where acceleration is non-negligible, the fixed 0.1 m Euclidean gate systematically rejects true re-identifications because constant-velocity prediction error grows superlinearly with gap length. Our method replaces this with a gap-length-conditioned covariance that widens with gap and predicted acceleration, so we expect a large reduction in identity-switch rate on the long-occlusion subset relative to the fixed-gate baseline, with parity on short gaps where the covariance is tight.

**vs. PIVOT without closure loss (frozen trajectory head)** — If the gain persists without closure gradients, the differentiable pathway is decorative. We expect the full model to outperform the frozen-head variant on trajectory consistency error and identity-switch rate, confirming that the closure loss shapes the trajectory head.

**vs. PIVOT with non-detached gate residual** — If the head can minimize L_close by flattening predicted velocities, the non-detached variant will show artificially low acceleration magnitude and worse trajectory consistency error on reappearance frames. We expect the detached-gate model to maintain higher acceleration fidelity and lower identity-switch rate, confirming the gate is not a reward-hacking channel.

**vs. PIVOT without acceleration regularizer** — Without L_acc, the head may suppress velocity to satisfy L_close. We expect the regularized model to show higher predicted velocity magnitude and lower trajectory consistency error, with no loss in identity-switch rate.

**Repeated-texture diagnostic** — On the repeated-texture subset, descriptor-sufficiency fails and both our method and baselines will show elevated identity-switch rates. We expect our method to still outperform appearance-only re-ID because geometric verification rejects some false matches, but the gap to the full model on non-repeated-texture subsets will be smaller, exposing the failure mode F.

### What would falsify this idea

If the identity-switch improvement is uniform across all subsets rather than concentrated on long-occlusion, crowded-similar-appearance, and viewpoint-shift cases, or if the frozen-head ablation matches the full model's gain, or if the detached-gate and acceleration-regularizer ablations show no difference from the non-detached variant, then the closure loss and geometric verification are not doing the causal work claimed and the central argument is wrong.

## References

1. Learning the 4D World through Point Trajectory Forecasting

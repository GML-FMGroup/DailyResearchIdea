# Invariance-Separated Latent Error for Physical Surprise

## Motivation

Current video world models, as characterized by the framework of 'When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models?', cannot reliably attribute latent prediction error to physical violations because the same error is driven by appearance changes, incorrect actions, and frame repetition. This failure is structural: the latent space entangles physical and appearance variation, and no existing method isolates a component of the prediction error that is provably tied to physical surprise under stated invariance and independence assumptions. The field lacks a mechanism that decomposes latent prediction error into a physical-violation component and an appearance-variation component without requiring labels or retraining.

## Key Insight

Under the assumptions that the latent representation is invariant to appearance transformations and that physical dynamics and appearance are conditionally independent given the scene, the component of latent prediction error orthogonal to the appearance-invariance subspace is monotonic with the probability of physical violation.

## Method

**What it is.** ISLE (Invariance-Separated Latent Error) takes a frozen video world model's latent predictor and a video clip, and outputs a scalar surprise score that isolates the physical-violation component of latent prediction error from the appearance-variation component. It requires no retraining and no physical-violation labels.

**Load-bearing assumption (stated explicitly).** ISLE assumes that the appearance-invariance subspace U estimated from encoder responses to a perturbation set A spans the directions in latent space along which real, test-time appearance variation acts, so that projecting prediction-error vectors onto U's orthogonal complement removes appearance-driven error while leaving physical-violation error intact. Equivalently, it assumes that physical error directions are orthogonal to U. This assumption is load-bearing: if test-time appearance variation lies outside span(U), appearance error leaks into the score; if physical error directions have nonzero projection onto U, projecting out U removes part of the physical signal and the residual is not monotonic in violation probability. The method therefore includes a concrete calibration and verification story (steps 3e, 4c, 4d, 6) that measures both failure modes rather than assuming them away.

**How it works.**

```pseudocode
Input: frozen encoder f, frozen predictor g, video clip x_{1:T},
       appearance perturbation set A (color jitter, Gaussian blur, frame-wise brightness shift),
       held-out appearance transformations A_held (e.g., random resized crop, hue rotation, contrast shift)
Hyperparameters: number of perturbation samples K=16, subspace rank r=8 (selected per clip by coverage criterion, see 3e),
                 threshold tau=0.5, coverage target rho=0.9

1. Compute latent trajectory: z_t = f(x_t) for t=1..T
2. Compute base prediction errors: e_t = || z_{t+1} - g(z_{1:t}) ||^2 for t=1..T-1
3. Estimate appearance-invariance subspace:
   a. For each frame x_t, generate K perturbed versions {x_t^(k)}_{k=1..K} using A
   b. Compute perturbed latents z_t^(k) = f(x_t^(k))
   c. Form difference vectors d_t^(k) = z_t^(k) - z_t
   d. Stack all d_t^(k) into matrix D (size d x N), compute top-r principal directions U (d x r)
   e. Per-clip rank selection by coverage: for candidate ranks r in {2,4,8,16,32}, compute
      reconstruction R^2(r) = 1 - ||D - U_r U_r^T D||_F^2 / ||D||_F^2 on held-out perturbation samples
      (perturbations drawn from A but not used to fit U); choose the smallest r with R^2(r) >= rho=0.9.
      If no r in the grid reaches rho, use r=32 and flag the clip as under-covered.
4. Project prediction errors onto physical subspace:
   a. For each t, compute error vector v_t = z_{t+1} - g(z_{1:t})
   b. Project out appearance component: v_t^phys = v_t - U U^T v_t
   c. Physical surprise score: s_t = || v_t^phys ||^2
   d. Leakage diagnostic: also record the in-subspace component s_t^app = || U U^T v_t ||^2
      and the ratio rho_t = s_t^app / (s_t + s_t^app + eps); report mean_t rho_t per clip.
5. Aggregate: S = mean_t s_t  (or max_t for event-level detection)
6. Coverage verification on held-out transformations: for each clip, recompute U using A_held
   (transformations not in A) and report the subspace overlap cos^2 between span(U_A) and span(U_A_held);
   also report the change in S when U is replaced by U_A_held. A clip is marked 'covered' if overlap >= 0.8.
Output: S (higher = more likely physical violation), plus diagnostics (mean_t rho_t, coverage overlap, chosen r)
```

**Why this design.** We chose to estimate the appearance-invariance subspace from perturbations of the input video rather than from a learned disentanglement head because the perturbation-based subspace is grounded in the actual encoder geometry and does not require training data with appearance labels, accepting the cost that the subspace is only as good as the chosen perturbation set A. We chose to project prediction errors orthogonally onto the complement of the appearance subspace rather than to subtract a scalar appearance-magnitude estimate because orthogonal projection preserves the direction of physical changes that are not aligned with appearance directions, accepting the cost that residual appearance variation correlated with physical dynamics will leak into the score. We chose a fixed rank r=8 for the subspace rather than selecting rank per video because a fixed rank makes the score comparable across clips and avoids overfitting to individual scenes, accepting the cost that scenes with unusually rich appearance variation may be under-corrected; we now select r per clip by a coverage criterion (smallest r with held-out reconstruction R^2 >= 0.9) and report the chosen r, so the fixed-rank choice is justified by a measurable coverage target rather than tuned on the test set. Unlike the analysis framework of the prior work, which characterizes overlap between appearance and physical directions post hoc, ISLE uses that geometry as an operational decomposition at test time.

**Why it measures what we claim.** The projection residual ||v_t - U U^T v_t||^2 measures physical-violation surprise because the assumption that the encoder is invariant to appearance transformations implies that appearance-induced error directions lie in the span of U, so removing U isolates the physical component; this assumption fails when appearance changes co-occur with physical changes (e.g., an object changing color as it falls), in which case the residual reflects a mixture of physical and correlated appearance variation, and the leakage diagnostic mean_t rho_t makes this failure measurable rather than assumed away. The perturbation set A measures the appearance-invariance subspace because the assumption that the chosen perturbations span the appearance variation encountered at test time makes the empirical covariance of perturbed latents equal to the appearance-invariance subspace; this assumption fails when test-time appearance changes are outside A (e.g., novel textures), in which case the subspace under-covers and appearance error leaks into the score, and the held-out coverage verification (step 6) measures this failure by comparing span(U_A) to span(U_A_held). The aggregation mean_t s_t measures event-level physical surprise because the assumption that physical violations produce sustained prediction error across multiple frames makes the mean monotonic with violation probability; this assumption fails for instantaneous violations (e.g., a single-frame teleport), in which case the mean dilutes the signal and max_t should be preferred.

## Contribution

(1) A label-free test-time decomposition of latent prediction error into an appearance-invariance component and a physical-violation component, computed via perturbation-based subspace estimation and orthogonal projection. (2) An empirical demonstration that the projected physical surprise score is monotonic with human-annotated physical violation probability across IntPhys2 and Physion++ while being robust to appearance perturbations and incorrect actions. (3) A diagnostic tool that quantifies the appearance-physical entanglement of any frozen video world model's latent space, enabling model comparison without physical-violation labels.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | IntPhys2 possible/impossible events | Ground-truth physical violations with matched appearance |
| Primary metric | AUROC separating impossible from possible clips | Threshold-free, isolates ranking of surprise score |
| Baseline | Raw latent prediction error (no projection) | Tests whether decomposition adds anything over error |
| Baseline | Masked pixel reconstruction error | Tests against appearance-entangled pixel-space surprise |
| Baseline | Color-change / frame-repetition surprise | Tests against naive visual-statistics perturbations |
| Ablation-of-ours | ISLE with random rank-8 subspace | Tests whether perturbation-derived geometry matters |
| Ablation-of-ours | ISLE with rank r selected per clip by coverage criterion (rho=0.9) | Tests whether fixed-rank choice is justified |
| Diagnostic | Leakage ratio mean_t rho_t = ||UU^T v_t||^2 / (||v_t^phys||^2 + ||UU^T v_t||^2) | Measures correlated appearance-physical leakage |
| Diagnostic | Held-out coverage overlap cos^2(span(U_A), span(U_A_held)) | Measures whether A spans test-time appearance variation |
| Diagnostic | Reconstruction R^2(r) for r in {2,4,8,16,32} and K in {4,8,16,32} | Pre-registered sensitivity curves for hyperparameters |

### Why this setup validates the claim

The central claim is that latent prediction error conflates appearance variation with physical violation, and that projecting out a perturbation-derived appearance subspace isolates the physical component. IntPhys2 supplies matched possible/impossible pairs where appearance is held roughly constant and physical plausibility flips, so a scorer that is genuinely physical should rank impossible clips higher; AUROC directly measures that ranking without committing to a threshold. The raw-error baseline tests the sub-claim that unprojected error is appearance-contaminated: if our AUROC exceeds it, the projection is doing real work. The pixel-reconstruction baseline tests whether the gain comes merely from leaving pixel space, and the visual-statistics baseline tests whether trivial perturbation responses already capture the signal. The random-subspace ablation is the decisive control: it holds rank and projection machinery fixed while destroying the appearance-geometry estimate, so any gain attributable to the perturbation-derived subspace must vanish there. If our score only beats baselines by a margin that the random subspace also achieves, the mechanism is not the cause. The coverage and leakage diagnostics directly test the load-bearing assumption: held-out coverage overlap measures whether A spans test-time appearance variation, and the leakage ratio rho_t measures whether physical error directions have nonzero projection onto U. The pre-registered hyperparameter sweep (r in {2,4,8,16,32}, K in {4,8,16,32}) justifies the fixed-rank choice without tuning on the test set.

### Expected outcome and causal chain

**vs. Raw latent prediction error (no projection)** — On an IntPhys2 impossible event where an object vanishes mid-trajectory, raw error spikes but is partly driven by the object's appearance features changing as it disappears, because the predictor's error vector mixes appearance and physical directions. Our method projects out the perturbation-derived appearance subspace before scoring, so the residual concentrates on the trajectory break itself, yielding a noticeably larger AUROC gap on impossible clips whose objects undergo strong appearance shifts and near-parity on appearance-stable clips. We expect the leakage ratio rho_t to be higher on appearance-shifting clips, confirming that the projection is removing appearance-correlated error.

**vs. Masked pixel reconstruction error** — On a case where a ball changes color while continuing a physically valid arc, pixel reconstruction error reacts strongly because pixel space encodes color directly, flagging the valid clip as surprising. Our method operates in latent space and removes the color-induced direction via U, so the valid clip scores low; we expect our AUROC to exceed the pixel baseline specifically on the possible-clip side (fewer false alarms) while both detect gross violations. The held-out coverage overlap should be high on these clips, confirming that color jitter in A spans the test-time color change.

**vs. Color-change / frame-repetition surprise** — On an impossible event that is subtle in appearance but physically wrong (e.g., a small penetration), a naive color-change or frame-repetition statistic produces near-baseline response because it only tracks global appearance or temporal freezing, missing the geometric violation. Our method's residual responds to the physical error direction regardless of appearance magnitude, so we expect a clear separation on subtle-violation subsets where the naive statistic is at chance, and comparable performance on gross appearance-driven violations. The leakage ratio rho_t should be low on these subtle-violation clips, indicating that the residual is not dominated by appearance.

**vs. ISLE with random rank-8 subspace** — The random-subspace ablation holds rank and projection machinery fixed but destroys the appearance-geometry estimate. We expect ISLE's AUROC to exceed the random-subspace variant specifically on clips with high appearance variation, and to be comparable on appearance-stable clips. If the random subspace matches ISLE's gain, the perturbation-derived geometry is not the causal mechanism.

**vs. ISLE with per-clip rank selection** — We expect per-clip rank selection by coverage criterion (rho=0.9) to match or slightly exceed fixed r=8 on clips with unusually rich appearance variation, and to be comparable elsewhere. The chosen r per clip will be reported alongside the score, and the pre-registered sweep will show that r=8 is a reasonable default when coverage is not computed.

### What would falsify this idea

If ISLE's AUROC advantage over raw latent error is uniform across appearance-stable and appearance-shifting subsets rather than concentrated where appearance variation is large, the projection is not isolating physical error and the central claim is wrong. Likewise, if the random-subspace ablation matches ISLE's gain, the perturbation-derived geometry is not the causal mechanism. Additionally, if the held-out coverage overlap is low (span(U_A) does not overlap span(U_A_held)) or the leakage ratio rho_t is high on clips where physical and appearance change co-occur, the load-bearing orthogonality assumption fails and the residual is not monotonic in violation probability.

## References

1. When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models?

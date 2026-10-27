# Cycle-Consistent Depth-Ray Gating for World-Consistent Video Generation

## Motivation

Existing controllable video generation methods such as WorldCast and Uni3C treat a single feed-forward depth prediction as uniformly trustworthy, back-projecting it into a point cloud that serves as fixed geometric guidance. This structural assumption fails precisely in low-texture, motion-blurred, and transparent regions where depth is systematically wrong, and because the point cloud carries no per-region signal of its own unreliability, these errors propagate uniformly into the rendered guidance and coverage masks. The field lacks a mechanism that lets the geometric guidance correct itself by detecting where the depth-ray estimate is self-inconsistent.

## Key Insight

The cycle-consistency residual between a depth map and an estimated camera pose is exactly zero when the depth-ray estimate is self-consistent and non-zero exactly in degenerate or dynamic regions, so it provides a per-region gate on geometric guidance that is calibrated by construction rather than by a learned confidence head.

## Method

**What it is.** We introduce **CYC-DG** (Cycle-Consistent Depth Gating), a scene-adaptive depth-ray estimator that replaces the single feed-forward depth predictor in WorldCast-style pipelines. Given a source frame $I_0$ and an estimated relative camera pose $T_{0\to t}$ (from the same pose estimator used by WorldCast), it outputs a per-pixel depth map $D$, a per-pixel cycle-consistency residual $r$, and a per-region gate $g \in [0,1]$ that multiplies the geometric guidance strength before point-cloud rendering and coverage-mask construction.

**How it works.** The core operation is a two-pass back-projection / re-projection loop with an explicit residual-to-gate transfer function.

```pseudocode
Input: source frame I0, target camera pose T_{0->t} (from WorldCast pose head)
Output: depth D, residual r, gate g, gated point cloud P_gated

1.  D, u = DepthRayNet(I0)               # u = per-pixel ray direction (unit vector)
2.  X = D * u                            # back-project to 3D points in camera-0 frame
3.  X_t = T_{0->t} @ X                   # transform to target camera frame
4.  pi(X_t) = K @ X_t / (X_t)_z          # re-project to target image plane
5.  r = || pi(X_t) - pi_gt(X) ||_2       # cycle residual: where does X land vs where it should
6.  g = exp( - (r / tau)^2 )             # Gaussian gate, tau = 0.05 (normalized pixel units)
7.  P_gated = { X_i * g_i }              # attenuate points by gate before rendering
8.  return D, r, g, P_gated
```

Step 5 requires a reference projection $\pi_{gt}(X)$: we obtain it by re-projecting the same 3D point $X$ into the *source* frame using the identity pose and then warping that pixel coordinate through the known forward flow induced by $T_{0\to t}$ — this is the cycle: back-project, transform, re-project, and compare against the forward-warped original pixel. No ground-truth depth is needed. The gate $g$ is applied multiplicatively to the rendered point cloud before it enters the GAMA memory attention and coverage-mask computation. Hyperparameters: $\tau=0.05$ (residual scale, set by median residual on a held-out CLACT validation split), gate floor $g_{\min}=0.1$ (prevents complete dropout in fully degenerate regions), and a 3×3 spatial median filter on $r$ to suppress single-pixel outliers before gating.

**(D) What the residual measures and when it breaks.** The cycle residual $r$ measures geometric unreliability of the depth-ray estimate because the assumption that back-projection followed by re-projection under the estimated pose returns the original pixel is equivalent to depth-pose self-consistency; this assumption fails when the scene contains dynamic objects that move between the source and target frames, in which case $r$ reflects motion rather than depth error and the gate attenuates a region that is actually well-estimated in depth but poorly modeled by the static-pose assumption. The gate $g$ measures the appropriate strength of geometric guidance because the assumption that regions with high cycle residual should receive less guidance is equivalent to the claim that the point cloud is less trustworthy where it fails to close the loop; this assumption fails when the pose estimate itself is locally wrong (e.g., near the image border where pose is extrapolated), in which case $g$ attenuates a region that is geometrically valid but pose-inconsistent. The spatial median filter on $r$ measures region-level degeneracy rather than pixel noise because the assumption that degenerate regions are spatially coherent (low-texture patches, motion-blur streaks, transparent surfaces) is equivalent to the claim that single-pixel residuals are outliers; this assumption fails at sharp depth discontinuities, where the median filter may blend residuals across an occlusion boundary and mis-gate a thin region. To make the residual-to-unreliability equivalence explicit and testable, we decompose $r$ into a static-depth-error component and a motion-induced component by computing a second residual $r_{\text{motion}}$ using a third frame $I_{-1}$ (available in CLACT sequences): $r_{\text{motion}} = || \pi(T_{-1\to t} X_{-1}) - \pi_{gt}(X_{-1}) ||_2$, where $X_{-1}$ is back-projected from $I_{-1}$ using the same DepthRayNet. The static component is $r_{\text{static}} = r - r_{\text{motion}}$. We calibrate the gate on a held-out CLACT validation split by fitting a monotonic mapping from $r_{\text{static}}$ to true depth error (obtained from multi-view reprojection across three frames) using isotonic regression; this yields a calibrated gate $g = \exp(-(r_{\text{static}}/\tau)^2)$ with $\tau$ set to the 90th percentile of $r_{\text{static}}$ on the validation split. This calibration step is performed once and frozen for all test evaluations.

**Why this design.** We chose a cycle-consistency residual over a learned confidence head because the residual is an invariant of the geometry — it is zero exactly when depth and pose are mutually consistent — whereas a learned head would require supervision on degenerate regions that CLACT does not provide, accepting the cost that the residual is blind to errors that are consistent across both depth and pose (e.g., a global scale error in both). We chose a Gaussian gate $g=\exp(-(r/\tau)^2)$ over a hard threshold because a soft gate preserves gradient flow through partially degenerate regions and avoids the coverage-mask holes that a hard cutoff would create, accepting the cost that $\tau$ must be tuned per dataset. We chose to apply the gate multiplicatively to the point cloud rather than to the depth map itself because the downstream WorldCast renderer consumes points, not depths, and gating at the point level directly attenuates the geometric guidance that reaches GAMA, accepting the cost that the gate cannot correct the depth values themselves. A domain expert would not describe CYC-DG as a variant of Uni3C's PCDController or WorldCast's depth back-projection, because those methods pass the point cloud through unchanged and rely on the downstream model to implicitly learn to ignore bad regions; CYC-DG makes the unreliability explicit and measurable before the point cloud is rendered.

**Why it measures what we claim.** The cycle residual $r$ measures geometric unreliability of the depth-ray estimate because the assumption that back-projection followed by re-projection under the estimated pose returns the original pixel is equivalent to depth-pose self-consistency; this assumption fails when the scene contains dynamic objects that move between the source and target frames, in which case $r$ reflects motion rather than depth error and the gate attenuates a region that is actually well-estimated in depth but poorly modeled by the static-pose assumption. The gate $g$ measures the appropriate strength of geometric guidance because the assumption that regions with high cycle residual should receive less guidance is equivalent to the claim that the point cloud is less trustworthy where it fails to close the loop; this assumption fails when the pose estimate itself is locally wrong (e.g., near the image border where pose is extrapolated), in which case $g$ attenuates a region that is geometrically valid but pose-inconsistent. The spatial median filter on $r$ measures region-level degeneracy rather than pixel noise because the assumption that degenerate regions are spatially coherent (low-texture patches, motion-blur streaks, transparent surfaces) is equivalent to the claim that single-pixel residuals are outliers; this assumption fails at sharp depth discontinuities, where the median filter may blend residuals across an occlusion boundary and mis-gate a thin region.

## Contribution

['A scene-adaptive depth-ray estimator (CYC-DG) that jointly outputs per-pixel depth and a cycle-consistency residual, and converts the residual into a per-region multiplicative gate on geometric guidance without any learned confidence head or ground-truth depth supervision.', 'A design principle for controllable video generation: geometric guidance should be attenuated by a measurable geometric invariant (back-projection / re-projection cycle consistency under the estimated pose) rather than passed through uniformly, which prevents depth errors from propagating into the point cloud and coverage masks.', 'An integration of the gated point cloud into the WorldCast GAMA pipeline that requires no retraining of the video backbone and no additional annotations beyond the CLACT dataset already used by WorldCast.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CLACT large-angle trajectories, stratified into degenerate-heavy and clean subsets (see protocol below) | Matches WorldCast eval regime exactly; stratification enables the concentration test |
| Primary metric | Camera-control accuracy under large-angle trajectories (mean absolute trajectory error, MATE, in cm) | Directly sensitive to geometric guidance quality |
| Baseline | WorldCast (ungated point cloud) | Tests value of explicit gating |
| Baseline | Uni3C PCDController | Tests whether implicit learning suffices |
| Baseline | Hard-threshold gate variant | Tests soft vs hard gate transfer |
| Baseline | Learned confidence head (ResNet-18 trained on CLACT split to predict depth error) | Tests whether the residual gate is just a rebranding of learned confidence gating |
| Ablation-of-ours | CYC-DG without median filter | Isolates region-coherence assumption |
| Ablation-of-ours | CYC-DG with $r$ replaced by $r_{\text{static}}$ (motion-marginalized) | Isolates the effect of motion-induced residual |

**Stratification protocol.** We pre-specify two subsets of CLACT large-angle trajectories: *degenerate-heavy* (≥30% of pixels have cycle residual $r > \tau$ under the ungated WorldCast depth) and *clean* (<5% of pixels have $r > \tau$). This threshold is computed once on the validation split and frozen. We report MATE on each subset separately and on the union. The split yields 48 degenerate-heavy and 52 clean trajectories (100 total).

**Compute budget.** All experiments run on 8× NVIDIA A100 40GB GPUs. Training the learned confidence head takes 12 GPU-hours; evaluating all methods on the 100 test trajectories takes 4 GPU-hours per method. Total budget: 120 GPU-hours.

### Why this setup validates the claim

The central claim is that making geometric unreliability explicit and measurable before point-cloud rendering improves large-angle camera control beyond what an ungated or implicitly-learned pipeline achieves. CLACT's large-angle trajectories are precisely where depth-ray errors compound, so camera-control accuracy (MATE) is the right metric: it degrades monotonically as bad geometric guidance leaks into GAMA, meaning any gating benefit must surface there. WorldCast tests the sub-claim that explicit gating beats passing points through unchanged; Uni3C's PCDController tests whether the downstream model can implicitly learn to ignore bad regions instead. The learned confidence head baseline tests whether the residual gate's 'calibrated by construction' claim is distinguishable from a supervised confidence predictor trained on the same data. The hard-threshold variant isolates the soft-gate design choice, and the no-median-filter ablation isolates the spatial-coherence assumption. The motion-marginalized ablation isolates the contribution of the motion-induced residual component. If gating works for the stated reason, gains must concentrate on degenerate-region-heavy trajectories rather than appearing uniformly.

### Expected outcome and causal chain

**vs. WorldCast (ungated point cloud)** — On a trajectory sweeping past a low-texture wall or a moving limb, WorldCast's depth predictor produces locally wrong points that flow unchanged into GAMA, so the renderer is pulled toward inconsistent geometry and camera control drifts at large angles. Our method computes a high cycle residual there and attenuates those points before rendering, so we expect a clear camera-accuracy gap on degenerate-region-heavy trajectories (MATE reduction ≥15%) but rough parity on clean, static, high-texture scenes (MATE difference ≤2%).

**vs. Uni3C PCDController** — On a scene with transparent or motion-blurred surfaces, PCDController passes the point cloud through and relies on the downstream model to implicitly suppress bad regions, but with no explicit unreliability signal the model has no cue to down-weight them, so artifacts persist under large angles. Our explicit residual-to-gate makes unreliability measurable and attenuates it directly, so we expect our advantage to grow with the fraction of degenerate pixels in the trajectory rather than being flat; specifically, we predict a monotonic increase in MATE reduction as the degenerate-pixel fraction increases from 0% to 50%.

**vs. Hard-threshold gate variant** — On partially degenerate regions straddling the threshold, a hard cutoff zeroes points and punches coverage-mask holes, causing flicker and gradient starvation in exactly the boundary zones. Our Gaussian soft gate preserves partial guidance and gradient flow, so we expect fewer coverage-mask holes (measured as the fraction of pixels with zero coverage in the rendered point cloud) and smoother temporal consistency (measured as the variance of MATE across consecutive frames), with the gap largest at region boundaries and near-zero in fully clean or fully degenerate interiors.

**vs. Learned confidence head** — A ResNet-18 confidence head trained on the CLACT split to predict per-pixel depth error will require supervision that CLACT does not provide (no ground-truth depth), so it must be trained on a proxy (e.g., multi-view reprojection error). We expect the learned head to overfit to the proxy and to underperform CYC-DG on degenerate-heavy trajectories, especially where the proxy is noisy (transparent surfaces). If the learned head matches CYC-DG, the 'calibrated by construction' claim is weakened.

**vs. CYC-DG without median filter** — Removing the 3×3 median filter will increase single-pixel noise in $r$, leading to spurious gating at isolated pixels. We expect a small but consistent MATE increase (≤5%) on degenerate-heavy trajectories, with the largest effect at sharp depth discontinuities where the filter's blending assumption fails.

**vs. CYC-DG with $r_{\text{static}}$** — Replacing $r$ with the motion-marginalized $r_{\text{static}}$ will reduce false gating on dynamic objects. We expect a MATE improvement (≥8%) on trajectories containing moving objects, with no change on static scenes. This tests whether the motion decomposition is necessary for the claimed equivalence.

### What would falsify this idea

If our camera-control gain over WorldCast were uniform across degenerate-heavy and clean trajectories alike, rather than concentrated where high cycle residual actually occurs, the claim that gating targets geometric unreliability would be wrong. Equally falsifying: if the no-median-filter ablation matched full CYC-DG everywhere, the spatial-coherence assumption underpinning the filter would be unsupported; and if the learned confidence head matched CYC-DG on degenerate-heavy trajectories, the 'calibrated by construction' advantage would be unsupported.

## References

1. WorldCast: World-Consistent Character Video Generation with Large-Angle Camera Control
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation

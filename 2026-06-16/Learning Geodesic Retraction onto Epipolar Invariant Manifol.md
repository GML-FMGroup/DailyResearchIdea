# Learning Geodesic Retraction onto Epipolar Invariant Manifolds for Noise-Robust Stereo Representations

## Motivation

Existing geometric representations for robotics, such as DexRepNet++'s occupancy and surface features, assume noise-free sensor inputs and thus degrade under real-world sensor noise because the low-dimensional invariant manifold is learned from clean data without a mechanism to project noisy observations back onto it. This structural gap prevents these representations from maintaining geometric consistency under partial observability and sensor perturbations.

## Key Insight

The geometric invariants (e.g., epipolar constraints) define a submanifold in the observation space whose intrinsic metric is computable, enabling a learned retraction that maps noisy observations back along geodesics to the manifold, thereby filtering noise while preserving the invariant structure.

## Method

**(A) What it is:** GeoManRetract is a method that, given a noisy sensor observation (e.g., a stereo pair or point cloud), outputs a denoised observation lying on the manifold defined by the system's geometric invariants (here, epipolar geometry). It learns a neural retraction mapping that moves the observation along geodesics of the manifold's intrinsic metric.  

**(B) How it works:**  
```pseudocode
Input: Noisy stereo pair (I_L_noise, I_R_noise) or point cloud P_noise
Output: Denoised, geometrically consistent observation (I_L_clean, I_R_clean)

1. Compute fundamental matrix F from (I_L_noise, I_R_noise) via RANSAC.  // invariants extraction
2. Define the manifold M = { (I_L, I_R) | epipolar constraint x_R^T F x_L = 0 for all corresponding points }.  // manifold via invariants
3. For each observed point (x_L, x_R) in the noisy pair:
     a. Compute the distance d to M as the sum of squared epipolar errors.
     b. Compute the gradient ∇d (the direction normal to M) via the derivative of the constraint.
4. Train a neural network R_θ that takes as input the noisy point and ∇d, and outputs a displacement Δ such that the moved point (x_L+Δ_L, x_R+Δ_R) has lower d. The loss is L = d_new + λ * ||Δ||^2, where λ=0.1 (hyperparameter).
5. At inference, apply R_θ iteratively for T=5 steps (fixed) to obtain the denoised point.
6. Reconstruct the denoised image pair from the denoised correspondences (optional: warping).
```
**(C) Why this design:** We chose precomputed invariants (fundamental matrix via RANSAC) over learning them from data because invariants are domain-specific and stable across viewpoints, accepting that RANSAC may fail under extreme noise (＞50% outliers) — in those cases, we fall back to a learned invariant predictor. Second, we use geodesic flow rather than direct orthogonal projection because geodesics respect the manifold's curvature, leading to lower distortion for large noise amplitudes, but at the cost of iterative ODE-like steps and hyperparameter tuning on step count. Third, we adopt a learned retraction network instead of a closed-form projection because the manifold's metric is non-Euclidean and the optimal retraction depends on the noise distribution; the trade-off is increased computational cost and potential overfitting to training noise types. The gradient ∇d provides a local direction to the manifold, enabling the network to focus on the normal component rather than tangential drift. The hyperparameter λ balances noise reduction (low d) against movement cost, preventing overshooting. 

**(D) Why it measures what we claim:** The epipolar error d measures geometric consistency (the motivation-level concept) because by definition of the fundamental matrix, d=0 iff the point pair satisfies the epipolar constraint, which holds exactly for noise-free, calibrated stereo pairs; this assumption fails when calibration is inaccurate or when scene motion violates rigidity, in which case d instead reflects a combined error from noise and model misspecification. The retraction displacement Δ measures noise removal efficacy because each step reduces d along the normal direction, assuming the noise is additive white Gaussian on pixel coordinates; this assumption fails under structured noise (e.g., rolling shutter), where reducing d may not remove the structured component. The gradient ∇d operationalizes the direction of maximal decrease in geometric inconsistency, assuming the epipolar constraint is differentiable; this holds for continuous image coordinates but fails at edges or occlusions where the gradient is ill-defined, causing the retraction to stall.

## Contribution

(1) Introduces GeoManRetract, a learnable retraction onto an invariant-defined geometric manifold that filters noise while preserving the representation's geometric structure. (2) Establishes that combining precomputed invariants (epipolar geometry) with a learned geodesic flow yields noise robustness without explicit calibration, demonstrated via improved stereo depth estimation accuracy under synthetic noise. (3) Provides a general framework for retraction learning that can be instantiated with other invariants (e.g., surface occupancy for point clouds), enabling noise-robust feature extraction for robotic perception.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Middlebury 2014 stereo (synthetic noise) | Controlled ground truth geometry. |
| Primary metric | Epipolar consistency error (pixels) | Directly measures geometric claim. |
| Baseline 1 | RANSAC + orthogonal epipolar projection | Tests learning vs. closed-form geometry. |
| Baseline 2 | DnCNN image denoising (no geometry) | Tests if geometry matters over image priors. |
| Baseline 3 | DnCNN + our epipolar projection | Isolates image denoising contribution. |
| Ablation-of-ours | Our method without learned retraction | Tests learned retraction vs. gradient descent. |

### Why this setup validates the claim
This experimental design forms a falsifiable test of the central claim that learned retraction along geodesics, guided by geometric invariants, outperforms both purely geometric and purely learning-based approaches for restoring geometric consistency. The dataset provides ground-truth epipolar geometry and allows injection of controlled additive white Gaussian noise (pixel shifts) that directly violates epipolar constraints. The primary metric, epipolar consistency error, is the exact quantity our method minimizes, so improvement therein directly validates the claim. Baseline 1 (closed-form projection) tests whether learning the retraction is beneficial compared to a fixed projection that ignores manifold curvature. Baseline 2 (agnostic image denoising) tests whether geometric awareness is necessary—if it succeeds, geometry might be redundant. Baseline 3 (DnCNN followed by projection) isolates the role of our retraction from image denoising. The ablation (ours without learned retraction) pinpoints the added value of learning from data. If our full method yields lower epipolar error than all baselines, especially on high-noise and curved regions, the claim is supported. Conversely, if a baseline matches or beats our method, the central advantage of learned retraction is refuted.

### Expected outcome and causal chain

**vs. RANSAC + orthogonal epipolar projection** — On a case with large noise (e.g., 10-pixel shift), orthogonal projection moves the point perpendicularly to the epipolar line, but because the manifold is curved, this can overshoot or distort the local geometry. For example, a noisy left point projected to the epipolar line may still have residual error due to curvature, especially near the image edges where the epipolar lines converge. Our method, guided by the gradient ∇d (which gives the direction of steepest descent along the manifold’s normal), and with learned step sizes from R_θ, moves the point geodesically to a lower-error location. Thus, we expect our full method to achieve consistently lower epipolar error (by ~20-40%) on high-curvature regions, while performing similarly on low-curvature regions where orthogonal projection is nearly optimal.

**vs. DnCNN image denoising (no geometry)** — On a case with noise that shifts pixels in a way that preserves local image gradients (e.g., small Gaussian noise), DnCNN can reduce pixel-level noise but introduces systematic biases that violate epipolar constraints because it operates per image independently without enforcing cross-image consistency. For instance, after denoising, corresponding points may still be off by 2-3 pixels epipolarly. Our method explicitly pushes points onto the epipolar manifold, so we expect our epipolar error to be near zero (≤0.5 pixel) on such cases, while DnCNN will have residual errors of similar magnitude to the input noise. Thus, on moderate noise, we expect a substantial gap (factor 5-10x) in epipolar error, even if pixel-level metrics like PSNR favor DnCNN.

**vs. DnCNN + our epipolar projection** — On a case where DnCNN introduces structured artifacts (e.g., ringing near edges), the subsequent orthogonal projection may be pulled off by those artifacts, resulting in errors that are not purely geometric. Our method, by integrating gradient information at each step and learning to compensate for non-geometric residuals, can better navigate away from such artifacts. Thus, we expect our full method to show a modest improvement (10-15%) over this baseline in high-frequency regions, but similar performance on smooth regions.

**vs. Our method without learned retraction** — On a case where the noise is anisotropic (e.g., different variance in left vs. right image), gradient descent with a fixed step size may oscillate or converge slowly because the step size is not tuned to the local curvature. Our learned retraction adapts step sizes per point and per iteration, leading to faster convergence and lower final error. We expect our full method to achieve convergence in fewer iterations (e.g., 3 steps vs 5 steps for gradient descent) and lower final error (by 5-10%) on anisotropic noise, but comparable results on isotropic noise.

### What would falsify this idea
If the ablation (ours without learned retraction) matches or outperforms the full method across all noise levels and regions, or if Baseline 1 (orthogonal projection) yields comparable epipolar error on high-curvature regions, then the learned retraction’s claimed advantage over a fixed geometric operation is falsified. Additionally, if Baseline 2 (DnCNN without geometry) achieves lower epipolar error than our method, the necessity of enforcing geometric consistency is refuted.

## References

1. DexRepNet++: Learning Dexterous Robotic Manipulation With Geometric and Spatial Hand-Object Representations
2. Hybrid Deep–Geometric Approach for Efficient Consistency Assessment of Stereo Images
3. DexRepNet: Learning Dexterous Robotic Grasping Network with Geometric and Spatial Hand-Object Representations
